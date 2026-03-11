"""
SAE (Sparse Autoencoder) comparison between two model architectures.

Trains simple SAEs on residual stream activations from two checkpoints and
compares feature quality metrics: reconstruction loss, L0 sparsity, dead
features, explained variance, and feature density distributions.

Usage:
    python sae_comparison.py checkpoint_best.pt checkpoint_baseline.pt
    python sae_comparison.py ckpt_a.pt ckpt_b.pt --layers 0,2,4,6 --n-activations 50000
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader

# ---------------------------------------------------------------------------
# Model loading (from interpret.py, architecture-independent via AST)
# ---------------------------------------------------------------------------


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint using AST-based source extraction.
    Reconstructs the model architecture from the saved train.py source code.
    """
    import ast

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_source = checkpoint["train_source"]

    tree = ast.parse(train_source)
    safe_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant)):
                safe_nodes.append(node)

    safe_tree = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_tree)
    safe_source = ast.unparse(safe_tree)

    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

    preamble = "\n".join(
        [
            "import os, sys, math, gc, time, contextlib",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from dataclasses import dataclass, asdict",
            f"CACHE_DIR = os.path.expanduser('~/.cache/autoresearch')",
            f"TOKENIZER_DIR = os.path.join(CACHE_DIR, 'tokenizer')",
        ]
    )
    full_source = preamble + "\n" + safe_source

    namespace = {"__builtins__": __builtins__}
    exec(full_source, namespace)

    GPTConfig_cls = namespace["GPTConfig"]
    GPT_cls = namespace["GPT"]
    config = GPTConfig_cls(**checkpoint["config"])
    model = GPT_cls(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    """
    Simple SAE: encoder (Linear + ReLU) -> decoder (Linear).
    Loss = MSE reconstruction + L1 sparsity on encoder activations.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)
        # Tie decoder bias to zero (reconstruct centered activations)
        self.decoder.bias.data.zero_()
        # Initialize encoder/decoder with small weights
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, encoder_activations, loss_dict)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    device,
    layer_indices: List[int],
    n_activations: int = 100_000,
    batch_size: int = 4,
) -> Dict[int, torch.Tensor]:
    """
    Collect residual stream activations at specified layers.
    Returns dict mapping layer_idx -> tensor of shape (n_activations, d_model).
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    layer_buffers: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
    layer_counts: Dict[int, int] = {i: 0 for i in layer_indices}
    target = n_activations

    # Register hooks on transformer blocks to capture residual stream output
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            captured[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for idx in layer_indices:
        h = model.transformer.h[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    done = False
    while not done:
        x, y, _ = next(val_loader)
        x = x.to(device)
        captured.clear()
        _ = model(x)

        for idx in layer_indices:
            if layer_counts[idx] >= target:
                continue
            # output shape: (B, T, d_model)
            acts = captured[idx].float().reshape(-1, captured[idx].shape[-1])
            # Subsample to avoid collecting way too much
            remaining = target - layer_counts[idx]
            if acts.shape[0] > remaining:
                acts = acts[:remaining]
            layer_buffers[idx].append(acts.cpu())
            layer_counts[idx] += acts.shape[0]

        # Check if all layers collected enough
        if all(c >= target for c in layer_counts.values()):
            done = True

    for h in hooks:
        h.remove()

    result = {}
    for idx in layer_indices:
        result[idx] = torch.cat(layer_buffers[idx], dim=0)[:target]
    return result


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------


@dataclass
class SAETrainConfig:
    expansion_factor: int = 4
    n_steps: int = 5000
    batch_size: int = 256
    lr: float = 3e-4
    l1_coeff: float = 1e-3


def train_sae(
    activations: torch.Tensor,
    device: torch.device,
    config: SAETrainConfig = SAETrainConfig(),
    quiet: bool = False,
) -> Tuple[SparseAutoencoder, Dict]:
    """
    Train a sparse autoencoder on collected activations.
    Returns (trained_sae, metrics_dict).
    """
    n_samples, d_model = activations.shape
    d_sae = d_model * config.expansion_factor

    sae = SparseAutoencoder(d_model, d_sae).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)

    # Precompute mean for centering (helps SAE training)
    act_mean = activations.mean(dim=0).to(device)

    losses = []
    for step in range(config.n_steps):
        # Sample random batch
        indices = torch.randint(0, n_samples, (config.batch_size,))
        batch = activations[indices].to(device)
        batch = batch - act_mean  # center activations

        x_hat, z = sae(batch)

        mse_loss = F.mse_loss(x_hat, batch)
        l1_loss = z.abs().mean()
        loss = mse_loss + config.l1_coeff * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if not quiet and (step + 1) % 1000 == 0:
            print(
                f"    step {step + 1}/{config.n_steps} | loss={loss.item():.5f} "
                f"| mse={mse_loss.item():.5f} | l1={l1_loss.item():.4f}"
            )

    # --- Compute final metrics on full dataset (in batches) ---
    sae.eval()
    metrics = compute_sae_metrics(sae, activations, act_mean, device, config.batch_size)
    sae.train()

    return sae, metrics


@torch.no_grad()
def compute_sae_metrics(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    act_mean: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Dict:
    """Compute SAE quality metrics over full activation dataset."""
    n_samples = activations.shape[0]
    d_sae = sae.d_sae

    total_mse = 0.0
    total_ss_res = 0.0
    total_ss_tot = 0.0
    total_l0 = 0.0
    feature_fire_counts = torch.zeros(d_sae, device=device)
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = activations[start:end].to(device)
        batch_centered = batch - act_mean

        x_hat, z = sae(batch_centered)

        # MSE
        mse = F.mse_loss(x_hat, batch_centered, reduction="sum")
        total_mse += mse.item()

        # Explained variance components
        total_ss_res += ((batch_centered - x_hat) ** 2).sum().item()
        total_ss_tot += (
            ((batch_centered - batch_centered.mean(dim=0, keepdim=True)) ** 2)
            .sum()
            .item()
        )

        # L0: count of active features per sample
        active = (z > 0).float()
        total_l0 += active.sum(dim=1).sum().item()  # sum of per-sample L0

        # Feature fire counts
        feature_fire_counts += (z > 0).float().sum(dim=0)

        n_batches += 1

    # Aggregate
    avg_mse = total_mse / n_samples
    avg_l0 = total_l0 / n_samples
    r_squared = 1.0 - (total_ss_res / total_ss_tot) if total_ss_tot > 0 else 0.0

    # Dead features: features that never fired across entire dataset
    feature_freqs = feature_fire_counts / n_samples
    dead_frac = (feature_fire_counts == 0).float().mean().item()

    return {
        "reconstruction_mse": avg_mse,
        "l0_sparsity": avg_l0,
        "dead_features_pct": dead_frac * 100.0,
        "explained_variance_r2": r_squared,
        "feature_freqs": feature_freqs.cpu(),  # for histogram
    }


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------


def print_comparison_table(
    name_a: str,
    name_b: str,
    results_a: Dict[int, Dict],
    results_b: Dict[int, Dict],
    layer_indices: List[int],
):
    """Print a formatted comparison table."""
    metrics = [
        "reconstruction_mse",
        "l0_sparsity",
        "dead_features_pct",
        "explained_variance_r2",
    ]
    metric_labels = {
        "reconstruction_mse": "Recon MSE",
        "l0_sparsity": "L0 Sparsity",
        "dead_features_pct": "Dead Feats %",
        "explained_variance_r2": "Explained Var (R2)",
    }
    metric_formats = {
        "reconstruction_mse": ".5f",
        "l0_sparsity": ".1f",
        "dead_features_pct": ".1f",
        "explained_variance_r2": ".4f",
    }
    # Which direction is "better" for each metric
    # lower_is_better: True means lower value = better
    metric_lower_better = {
        "reconstruction_mse": True,
        "l0_sparsity": True,
        "dead_features_pct": True,
        "explained_variance_r2": False,
    }

    header = (
        f"{'Layer':>6} | {'Metric':<20} | {name_a:>14} | {name_b:>14} | {'Winner':>10}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SAE COMPARISON RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)

    for layer_idx in layer_indices:
        for m in metrics:
            val_a = results_a[layer_idx][m]
            val_b = results_b[layer_idx][m]
            fmt = metric_formats[m]
            label = metric_labels[m]

            if metric_lower_better[m]:
                winner = name_a if val_a < val_b else name_b if val_b < val_a else "tie"
            else:
                winner = name_a if val_a > val_b else name_b if val_b > val_a else "tie"

            print(
                f"{layer_idx:>6} | {label:<20} | {val_a:>{14}{fmt}} | {val_b:>{14}{fmt}} | {winner:>10}"
            )
        print(sep)

    # Averages across layers
    print(f"\n{'AVERAGE ACROSS LAYERS':^{len(header)}}")
    print(sep)
    for m in metrics:
        vals_a = [results_a[l][m] for l in layer_indices]
        vals_b = [results_b[l][m] for l in layer_indices]
        avg_a = sum(vals_a) / len(vals_a)
        avg_b = sum(vals_b) / len(vals_b)
        fmt = metric_formats[m]
        label = metric_labels[m]

        if metric_lower_better[m]:
            winner = name_a if avg_a < avg_b else name_b if avg_b < avg_a else "tie"
        else:
            winner = name_a if avg_a > avg_b else name_b if avg_b > avg_a else "tie"

        print(
            f"{'avg':>6} | {label:<20} | {avg_a:>{14}{fmt}} | {avg_b:>{14}{fmt}} | {winner:>10}"
        )
    print(sep)


def save_comparison_figure(
    name_a: str,
    name_b: str,
    results_a: Dict[int, Dict],
    results_b: Dict[int, Dict],
    layer_indices: List[int],
    output_path: str,
):
    """Save a figure with per-layer reconstruction loss, L0, and feature density histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Reconstruction MSE per layer ---
    ax = axes[0, 0]
    mse_a = [results_a[l]["reconstruction_mse"] for l in layer_indices]
    mse_b = [results_b[l]["reconstruction_mse"] for l in layer_indices]
    x_pos = range(len(layer_indices))
    width = 0.35
    ax.bar(
        [p - width / 2 for p in x_pos],
        mse_a,
        width,
        label=name_a,
        color="#2196F3",
        alpha=0.85,
    )
    ax.bar(
        [p + width / 2 for p in x_pos],
        mse_b,
        width,
        label=name_b,
        color="#FF9800",
        alpha=0.85,
    )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Loss per Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: L0 Sparsity per layer ---
    ax = axes[0, 1]
    l0_a = [results_a[l]["l0_sparsity"] for l in layer_indices]
    l0_b = [results_b[l]["l0_sparsity"] for l in layer_indices]
    ax.bar(
        [p - width / 2 for p in x_pos],
        l0_a,
        width,
        label=name_a,
        color="#2196F3",
        alpha=0.85,
    )
    ax.bar(
        [p + width / 2 for p in x_pos],
        l0_b,
        width,
        label=name_b,
        color="#FF9800",
        alpha=0.85,
    )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_xlabel("Layer")
    ax.set_ylabel("L0 (avg active features)")
    ax.set_title("L0 Sparsity per Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: Explained variance (R2) per layer ---
    ax = axes[1, 0]
    r2_a = [results_a[l]["explained_variance_r2"] for l in layer_indices]
    r2_b = [results_b[l]["explained_variance_r2"] for l in layer_indices]
    ax.bar(
        [p - width / 2 for p in x_pos],
        r2_a,
        width,
        label=name_a,
        color="#2196F3",
        alpha=0.85,
    )
    ax.bar(
        [p + width / 2 for p in x_pos],
        r2_b,
        width,
        label=name_b,
        color="#FF9800",
        alpha=0.85,
    )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_xlabel("Layer")
    ax.set_ylabel("R-squared")
    ax.set_title("Explained Variance per Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 4: Feature density histogram (aggregated across layers) ---
    ax = axes[1, 1]
    all_freqs_a = torch.cat([results_a[l]["feature_freqs"] for l in layer_indices])
    all_freqs_b = torch.cat([results_b[l]["feature_freqs"] for l in layer_indices])
    # Filter out dead features for histogram clarity
    live_a = all_freqs_a[all_freqs_a > 0].log10().numpy()
    live_b = all_freqs_b[all_freqs_b > 0].log10().numpy()
    bins = 50
    if len(live_a) > 0:
        ax.hist(
            live_a, bins=bins, alpha=0.6, label=name_a, color="#2196F3", density=True
        )
    if len(live_b) > 0:
        ax.hist(
            live_b, bins=bins, alpha=0.6, label=name_b, color="#FF9800", density=True
        )
    ax.set_xlabel("log10(feature activation frequency)")
    ax.set_ylabel("Density")
    ax.set_title("Feature Density Distribution (live features)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train SAEs on two model checkpoints and compare feature quality."
    )
    parser.add_argument(
        "checkpoint_a", type=str, help="Path to first checkpoint (e.g., best)"
    )
    parser.add_argument(
        "checkpoint_b", type=str, help="Path to second checkpoint (e.g., baseline)"
    )
    parser.add_argument(
        "--name-a", type=str, default="best", help="Label for checkpoint A"
    )
    parser.add_argument(
        "--name-b", type=str, default="baseline", help="Label for checkpoint B"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: auto-select 5 layers)",
    )
    parser.add_argument(
        "--n-activations",
        type=int,
        default=100_000,
        help="Number of activation vectors to collect per layer",
    )
    parser.add_argument(
        "--sae-steps", type=int, default=5000, help="Number of SAE training steps"
    )
    parser.add_argument(
        "--sae-batch-size", type=int, default=256, help="Batch size for SAE training"
    )
    parser.add_argument(
        "--sae-lr", type=float, default=3e-4, help="Learning rate for SAE training"
    )
    parser.add_argument(
        "--l1-coeff", type=float, default=1e-3, help="L1 sparsity coefficient"
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=4,
        help="SAE expansion factor (d_sae = expansion_factor * d_model)",
    )
    parser.add_argument(
        "--collect-batch-size",
        type=int,
        default=4,
        help="Batch size for activation collection forward passes",
    )
    parser.add_argument(
        "--output", type=str, default="sae_comparison.png", help="Output figure path"
    )
    args = parser.parse_args()

    # --- Device ---
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_type)
    print(f"Device: {device_type}")

    # --- Load models ---
    print(f"\nLoading checkpoint A: {args.checkpoint_a}")
    model_a, ckpt_a = load_model(args.checkpoint_a, device)
    config_a = model_a.config
    print(
        f"  Model A ({args.name_a}): {config_a.n_layer} layers, {config_a.n_embd} dim, "
        f"val_bpb={ckpt_a.get('val_bpb', 'N/A')}"
    )

    print(f"\nLoading checkpoint B: {args.checkpoint_b}")
    model_b, ckpt_b = load_model(args.checkpoint_b, device)
    config_b = model_b.config
    print(
        f"  Model B ({args.name_b}): {config_b.n_layer} layers, {config_b.n_embd} dim, "
        f"val_bpb={ckpt_b.get('val_bpb', 'N/A')}"
    )

    # --- Determine layers to analyze ---
    if args.layers is not None:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    else:
        # Auto-select: layers 0, n//4, n//2, 3n//4, n-1 using the smaller model
        n = min(config_a.n_layer, config_b.n_layer)
        if n <= 4:
            layer_indices = list(range(n))
        else:
            layer_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    # Validate layer indices against both models
    for idx in layer_indices:
        if idx >= config_a.n_layer:
            print(
                f"Error: layer {idx} exceeds model A's depth ({config_a.n_layer} layers)"
            )
            sys.exit(1)
        if idx >= config_b.n_layer:
            print(
                f"Error: layer {idx} exceeds model B's depth ({config_b.n_layer} layers)"
            )
            sys.exit(1)

    print(f"\nAnalyzing layers: {layer_indices}")

    # --- Tokenizer ---
    tokenizer = Tokenizer.from_directory()

    # --- SAE training config ---
    sae_config = SAETrainConfig(
        expansion_factor=args.expansion_factor,
        n_steps=args.sae_steps,
        batch_size=args.sae_batch_size,
        lr=args.sae_lr,
        l1_coeff=args.l1_coeff,
    )

    results_a: Dict[int, Dict] = {}
    results_b: Dict[int, Dict] = {}

    # --- Collect activations and train SAEs ---
    print(f"\n{'=' * 60}")
    print(
        f"Collecting {args.n_activations:,} activations from model A ({args.name_a})..."
    )
    print(f"{'=' * 60}")
    acts_a = collect_activations(
        model_a,
        tokenizer,
        device,
        layer_indices,
        n_activations=args.n_activations,
        batch_size=args.collect_batch_size,
    )
    for idx in layer_indices:
        print(f"  Layer {idx}: {acts_a[idx].shape}")

    # Free model A from GPU
    del model_a
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()

    print(f"\n{'=' * 60}")
    print(
        f"Collecting {args.n_activations:,} activations from model B ({args.name_b})..."
    )
    print(f"{'=' * 60}")
    acts_b = collect_activations(
        model_b,
        tokenizer,
        device,
        layer_indices,
        n_activations=args.n_activations,
        batch_size=args.collect_batch_size,
    )
    for idx in layer_indices:
        print(f"  Layer {idx}: {acts_b[idx].shape}")

    # Free model B from GPU
    del model_b
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()

    # --- Train SAEs per layer ---
    for idx in layer_indices:
        print(f"\n{'=' * 60}")
        print(f"LAYER {idx}")
        print(f"{'=' * 60}")

        print(
            f"\n  Training SAE on {args.name_a} activations "
            f"(d_model={acts_a[idx].shape[1]}, d_sae={acts_a[idx].shape[1] * sae_config.expansion_factor})..."
        )
        _, metrics_a = train_sae(acts_a[idx], device, sae_config)
        results_a[idx] = metrics_a
        print(
            f"    -> MSE={metrics_a['reconstruction_mse']:.5f}, "
            f"L0={metrics_a['l0_sparsity']:.1f}, "
            f"Dead={metrics_a['dead_features_pct']:.1f}%, "
            f"R2={metrics_a['explained_variance_r2']:.4f}"
        )

        print(
            f"\n  Training SAE on {args.name_b} activations "
            f"(d_model={acts_b[idx].shape[1]}, d_sae={acts_b[idx].shape[1] * sae_config.expansion_factor})..."
        )
        _, metrics_b = train_sae(acts_b[idx], device, sae_config)
        results_b[idx] = metrics_b
        print(
            f"    -> MSE={metrics_b['reconstruction_mse']:.5f}, "
            f"L0={metrics_b['l0_sparsity']:.1f}, "
            f"Dead={metrics_b['dead_features_pct']:.1f}%, "
            f"R2={metrics_b['explained_variance_r2']:.4f}"
        )

    # --- Print comparison ---
    print_comparison_table(
        args.name_a, args.name_b, results_a, results_b, layer_indices
    )

    # --- Save figure ---
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), output_path
        )
    save_comparison_figure(
        args.name_a, args.name_b, results_a, results_b, layer_indices, output_path
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
