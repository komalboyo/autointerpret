"""
Linear probe analysis for autoresearch models.
Trains linear probes on intermediate layer activations to validate
interpretability differences across architectures.

Three probe tasks:
  1. Next-token prediction (tuned lens): linear map from hidden state -> vocab logits
  2. Bigram frequency probe: binary classifier for high-frequency bigram continuations
  3. Control tasks (Hewitt & Liang 2019): random label permutation baselines
     Selectivity = real accuracy - control accuracy (filters out linear separability artifacts)

Usage:
    python linear_probes.py checkpoint_best.pt checkpoint_baseline.pt
    python linear_probes.py checkpoint_best.pt  # single checkpoint analysis
"""

import os
import sys
import ast
import math
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader

# ---------------------------------------------------------------------------
# Checkpoint loading (mirrored from interpret.py)
# ---------------------------------------------------------------------------


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint. Uses AST-based source extraction to
    reconstruct model architecture from saved train.py source.
    """
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


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_activations(model, tokenizer, device, num_tokens=5000, batch_size=4):
    """
    Run model on validation data and collect residual stream activations
    at every layer, plus the target tokens.

    Returns:
        layer_acts: dict mapping layer_idx -> tensor of shape (N, n_embd)
        targets: tensor of shape (N,) with next-token ids
        input_tokens: tensor of shape (N,) with current token ids
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    n_layers = len(model.transformer.h)

    # Storage for residual stream at each layer
    layer_residuals = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            layer_residuals[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    all_layer_acts = {i: [] for i in range(n_layers)}
    all_targets = []
    all_inputs = []
    collected = 0

    while collected < num_tokens:
        x, y, _ = next(val_loader)
        x, y = x.to(device), y.to(device)
        layer_residuals.clear()
        _ = model(x)

        B, T = x.shape
        # Flatten batch and sequence dims
        for i in range(n_layers):
            # Apply final norm like the model does before lm_head
            residual = layer_residuals[i].float()
            flat = residual.reshape(B * T, -1)
            all_layer_acts[i].append(flat.cpu())

        all_targets.append(y.reshape(-1).cpu())
        all_inputs.append(x.reshape(-1).cpu())
        collected += B * T

    for h in hooks:
        h.remove()

    # Concatenate and truncate to num_tokens
    layer_acts = {}
    for i in range(n_layers):
        layer_acts[i] = torch.cat(all_layer_acts[i], dim=0)[:num_tokens]
    targets = torch.cat(all_targets, dim=0)[:num_tokens]
    input_tokens = torch.cat(all_inputs, dim=0)[:num_tokens]

    return layer_acts, targets, input_tokens


# ---------------------------------------------------------------------------
# Task 1: Next-token prediction probe (tuned lens)
# ---------------------------------------------------------------------------


def train_next_token_probe(
    layer_acts, targets, model, device, epochs=3, lr=1e-2, batch_size=512
):
    """
    Train a linear probe: hidden_state -> vocab logits.
    Uses a single nn.Linear layer trained with cross-entropy.

    Returns top-1 accuracy on a held-out split.
    """
    n_embd = layer_acts.shape[1]
    vocab_size = model.config.vocab_size

    # Train/test split (80/20)
    N = layer_acts.shape[0]
    split = int(0.8 * N)
    X_train, X_test = layer_acts[:split], layer_acts[split:]
    y_train, y_test = targets[:split], targets[split:]

    # Linear probe
    probe = nn.Linear(n_embd, vocab_size, bias=False).to(device)
    # Initialize from the model's lm_head as a warm start
    with torch.no_grad():
        probe.weight.copy_(model.lm_head.weight.float())

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Training
    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_train.shape[0])
        for start in range(0, X_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            # Apply norm like the model does before lm_head
            xb_normed = norm(xb)
            logits = probe(xb_normed)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, X_test.shape[0], batch_size):
            xb = X_test[start : start + batch_size].to(device)
            yb = y_test[start : start + batch_size].to(device)
            xb_normed = norm(xb)
            logits = probe(xb_normed)
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.shape[0]

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# ---------------------------------------------------------------------------
# Task 2: Bigram frequency probe (binary classification)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_bigram_labels(input_tokens, targets, top_k=500):
    """
    Build binary labels: 1 if (input_token, target_token) is a top-k
    frequent bigram, 0 otherwise.

    Returns:
        labels: tensor of shape (N,) with 0/1 values
        bigram_rate: fraction of positives (for baseline reference)
    """
    N = input_tokens.shape[0]
    # Count bigram frequencies
    bigram_counts = Counter()
    inp_np = input_tokens.numpy()
    tgt_np = targets.numpy()
    for i in range(N):
        bigram_counts[(int(inp_np[i]), int(tgt_np[i]))] += 1

    # Get top-k bigrams
    top_bigrams = set(bg for bg, _ in bigram_counts.most_common(top_k))

    # Build labels
    labels = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        if (int(inp_np[i]), int(tgt_np[i])) in top_bigrams:
            labels[i] = 1

    bigram_rate = labels.float().mean().item()
    return labels, bigram_rate


def train_bigram_probe(layer_acts, labels, device, epochs=5, lr=1e-3, batch_size=512):
    """
    Train a logistic regression (linear probe) for binary bigram classification.
    Uses sklearn if available, otherwise falls back to a simple PyTorch linear layer.

    Returns accuracy on held-out split.
    """
    n_embd = layer_acts.shape[1]
    N = layer_acts.shape[0]
    split = int(0.8 * N)

    X_train, X_test = layer_acts[:split], layer_acts[split:]
    y_train, y_test = labels[:split], labels[split:]

    try:
        from sklearn.linear_model import LogisticRegression

        # Normalize features for sklearn
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0).clamp(min=1e-6)
        X_train_np = ((X_train - mean) / std).numpy()
        X_test_np = ((X_test - mean) / std).numpy()
        y_train_np = y_train.numpy()
        y_test_np = y_test.numpy()

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_train_np, y_train_np)
        accuracy = clf.score(X_test_np, y_test_np)
        return accuracy

    except ImportError:
        # Fallback: PyTorch binary probe
        probe = nn.Linear(n_embd, 2, bias=True).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        probe.train()
        for epoch in range(epochs):
            perm = torch.randperm(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                idx = perm[start : start + batch_size]
                xb = X_train[idx].to(device)
                yb = y_train[idx].to(device)
                logits = probe(xb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, X_test.shape[0], batch_size):
                xb = X_test[start : start + batch_size].to(device)
                yb = y_test[start : start + batch_size].to(device)
                logits = probe(xb)
                preds = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.shape[0]

        accuracy = correct / total if total > 0 else 0.0
        return accuracy


# ---------------------------------------------------------------------------
# Control tasks (Hewitt & Liang 2019)
# ---------------------------------------------------------------------------


def train_control_next_token_probe(
    layer_acts, targets, model, device, epochs=3, lr=1e-2, batch_size=512, seed=0
):
    """
    Control task: train next-token probe with randomly permuted labels.
    If the probe achieves high accuracy on random labels, the representation
    is trivially linearly separable and real accuracy is not meaningful.
    Selectivity = real_accuracy - control_accuracy.
    """
    rng = np.random.RandomState(seed)
    shuffled_targets = targets[torch.from_numpy(rng.permutation(len(targets)))]
    return train_next_token_probe(
        layer_acts,
        shuffled_targets,
        model,
        device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )


def train_control_bigram_probe(
    layer_acts, labels, device, epochs=5, lr=1e-3, batch_size=512, seed=0
):
    """
    Control task: train bigram probe with randomly permuted labels.
    """
    rng = np.random.RandomState(seed)
    shuffled_labels = labels[torch.from_numpy(rng.permutation(len(labels)))]
    return train_bigram_probe(
        layer_acts, shuffled_labels, device, epochs=epochs, lr=lr, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# Probe interpretability score (area under layer-accuracy curve)
# ---------------------------------------------------------------------------


def compute_probe_auc(layer_accuracies):
    """
    Compute area under the layer-wise accuracy curve using the trapezoidal rule.
    Normalized to [0, 1] by dividing by the number of layers.
    Higher = information accessible earlier = more interpretable.
    """
    n = len(layer_accuracies)
    if n < 2:
        return layer_accuracies[0] if n == 1 else 0.0
    # Normalize layer indices to [0, 1]
    xs = [i / (n - 1) for i in range(n)]
    area = 0.0
    for i in range(n - 1):
        area += (
            0.5 * (layer_accuracies[i] + layer_accuracies[i + 1]) * (xs[i + 1] - xs[i])
        )
    return area


# ---------------------------------------------------------------------------
# Full analysis for one checkpoint
# ---------------------------------------------------------------------------


def analyze_checkpoint(checkpoint_path, device, tokenizer, num_tokens=5000):
    """
    Load checkpoint, extract activations, train probes at every layer.
    Returns a results dict.
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {os.path.basename(checkpoint_path)}")
    print(f"{'=' * 60}")

    model, ckpt = load_model(checkpoint_path, device)
    config = model.config
    n_layers = config.n_layer
    val_bpb = ckpt.get("val_bpb", "N/A")

    print(f"  Model: {n_layers} layers, {config.n_embd} dim, {config.n_head} heads")
    print(f"  val_bpb: {val_bpb}")

    print(f"\n  Extracting activations from {num_tokens} tokens...")
    layer_acts, targets, input_tokens = extract_activations(
        model, tokenizer, device, num_tokens=num_tokens
    )

    # Build bigram labels once (shared across layers)
    print("  Building bigram frequency labels...")
    bigram_labels, bigram_rate = build_bigram_labels(input_tokens, targets)
    print(
        f"  Bigram positive rate: {bigram_rate:.3f} (baseline accuracy = {max(bigram_rate, 1 - bigram_rate):.3f})"
    )

    # Train probes at each layer (real + control)
    ntp_accuracies = []
    ntp_control_accuracies = []
    bigram_accuracies = []
    bigram_control_accuracies = []

    for layer_idx in range(n_layers):
        acts = layer_acts[layer_idx]
        print(f"\n  Layer {layer_idx}/{n_layers - 1}:")

        # Next-token probe (real)
        print(f"    Training next-token probe...", end=" ", flush=True)
        ntp_acc = train_next_token_probe(acts, targets, model, device)
        ntp_accuracies.append(ntp_acc)
        print(f"acc = {ntp_acc:.4f}")

        # Next-token probe (control — random labels)
        print(f"    Training NTP control task...", end=" ", flush=True)
        ntp_ctrl = train_control_next_token_probe(acts, targets, model, device)
        ntp_control_accuracies.append(ntp_ctrl)
        ntp_sel = ntp_acc - ntp_ctrl
        print(f"control = {ntp_ctrl:.4f}, selectivity = {ntp_sel:+.4f}")

        # Bigram probe (real)
        print(f"    Training bigram probe...", end=" ", flush=True)
        bg_acc = train_bigram_probe(acts, bigram_labels, device)
        bigram_accuracies.append(bg_acc)
        print(f"acc = {bg_acc:.4f}")

        # Bigram probe (control — random labels)
        print(f"    Training bigram control task...", end=" ", flush=True)
        bg_ctrl = train_control_bigram_probe(acts, bigram_labels, device)
        bigram_control_accuracies.append(bg_ctrl)
        bg_sel = bg_acc - bg_ctrl
        print(f"control = {bg_ctrl:.4f}, selectivity = {bg_sel:+.4f}")

    # Compute selectivity = real - control (Hewitt & Liang 2019)
    ntp_selectivities = [r - c for r, c in zip(ntp_accuracies, ntp_control_accuracies)]
    bigram_selectivities = [
        r - c for r, c in zip(bigram_accuracies, bigram_control_accuracies)
    ]

    # Compute AUC scores (on both raw accuracy and selectivity)
    ntp_auc = compute_probe_auc(ntp_accuracies)
    ntp_control_auc = compute_probe_auc(ntp_control_accuracies)
    ntp_selectivity_auc = compute_probe_auc(ntp_selectivities)
    bigram_auc = compute_probe_auc(bigram_accuracies)
    bigram_control_auc = compute_probe_auc(bigram_control_accuracies)
    bigram_selectivity_auc = compute_probe_auc(bigram_selectivities)

    # Combined score uses SELECTIVITY (not raw accuracy) — this is the credible metric
    combined_selectivity_auc = (ntp_selectivity_auc + bigram_selectivity_auc) / 2.0
    combined_auc = (ntp_auc + bigram_auc) / 2.0

    results = {
        "name": os.path.basename(checkpoint_path),
        "n_layer": n_layers,
        "n_embd": config.n_embd,
        "val_bpb": val_bpb,
        # Raw accuracies
        "ntp_accuracies": ntp_accuracies,
        "bigram_accuracies": bigram_accuracies,
        "bigram_baseline": max(bigram_rate, 1 - bigram_rate),
        # Control accuracies (Hewitt & Liang 2019)
        "ntp_control_accuracies": ntp_control_accuracies,
        "bigram_control_accuracies": bigram_control_accuracies,
        # Selectivity = real - control
        "ntp_selectivities": ntp_selectivities,
        "bigram_selectivities": bigram_selectivities,
        # AUC scores
        "ntp_auc": ntp_auc,
        "ntp_control_auc": ntp_control_auc,
        "ntp_selectivity_auc": ntp_selectivity_auc,
        "bigram_auc": bigram_auc,
        "bigram_control_auc": bigram_control_auc,
        "bigram_selectivity_auc": bigram_selectivity_auc,
        # Combined scores
        "probe_interpretability_score": combined_selectivity_auc,  # PRIMARY: selectivity-based
        "probe_raw_score": combined_auc,  # secondary: raw accuracy (for reference)
    }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results_table(results_list):
    """Print a clear comparison table."""
    print(f"\n{'=' * 70}")
    print("PROBE INTERPRETABILITY ANALYSIS — COMPARISON")
    print(f"{'=' * 70}")

    # Header
    names = [r["name"] for r in results_list]
    max_layers = max(r["n_layer"] for r in results_list)

    # Model summary
    print(f"\n{'Model Summary':}")
    print(f"  {'':20s}", end="")
    for r in results_list:
        print(f"  {r['name']:>20s}", end="")
    print()
    print(f"  {'Layers':20s}", end="")
    for r in results_list:
        print(f"  {r['n_layer']:>20d}", end="")
    print()
    print(f"  {'Embedding dim':20s}", end="")
    for r in results_list:
        print(f"  {r['n_embd']:>20d}", end="")
    print()
    print(f"  {'val_bpb':20s}", end="")
    for r in results_list:
        bpb = r["val_bpb"]
        if isinstance(bpb, float):
            print(f"  {bpb:>20.4f}", end="")
        else:
            print(f"  {str(bpb):>20s}", end="")
    print()

    # Next-token probe table (with selectivity)
    print(f"\n{'--- Next-Token Prediction Probe (Tuned Lens) ---':}")
    col_w = 12
    ncols = len(results_list)
    print(f"  {'Layer':>8s}", end="")
    for r in results_list:
        name = r["name"][:10]
        print(f"  {'acc':>{col_w}s}  {'ctrl':>{col_w}s}  {'sel':>{col_w}s}", end="")
    print()
    print(f"  {'':>8s}", end="")
    for r in results_list:
        name = r["name"][:10]
        print(f"  {name:>{col_w}s}  {name:>{col_w}s}  {name:>{col_w}s}", end="")
    print()
    print(f"  {'-' * 8}", end="")
    for _ in results_list:
        print(f"  {'-' * col_w}  {'-' * col_w}  {'-' * col_w}", end="")
    print()

    for layer_idx in range(max_layers):
        print(f"  {layer_idx:>8d}", end="")
        for r in results_list:
            if layer_idx < len(r["ntp_accuracies"]):
                acc = r["ntp_accuracies"][layer_idx]
                ctrl = r["ntp_control_accuracies"][layer_idx]
                sel = r["ntp_selectivities"][layer_idx]
                print(
                    f"  {acc:>{col_w}.4f}  {ctrl:>{col_w}.4f}  {sel:>{col_w + 1}.4f}",
                    end="",
                )
            else:
                print(f"  {'—':>{col_w}s}  {'—':>{col_w}s}  {'—':>{col_w}s}", end="")
        print()

    print(f"\n  {'AUC':>8s}", end="")
    for r in results_list:
        print(
            f"  {r['ntp_auc']:>{col_w}.4f}  {r['ntp_control_auc']:>{col_w}.4f}  {r['ntp_selectivity_auc']:>{col_w + 1}.4f}",
            end="",
        )
    print()

    # Bigram probe table (with selectivity)
    print(f"\n{'--- Bigram Frequency Probe (Binary) ---':}")
    print(f"  {'Layer':>8s}", end="")
    for r in results_list:
        name = r["name"][:10]
        print(f"  {'acc':>{col_w}s}  {'ctrl':>{col_w}s}  {'sel':>{col_w}s}", end="")
    print()
    print(f"  {'-' * 8}", end="")
    for _ in results_list:
        print(f"  {'-' * col_w}  {'-' * col_w}  {'-' * col_w}", end="")
    print()

    for layer_idx in range(max_layers):
        print(f"  {layer_idx:>8d}", end="")
        for r in results_list:
            if layer_idx < len(r["bigram_accuracies"]):
                acc = r["bigram_accuracies"][layer_idx]
                ctrl = r["bigram_control_accuracies"][layer_idx]
                sel = r["bigram_selectivities"][layer_idx]
                print(
                    f"  {acc:>{col_w}.4f}  {ctrl:>{col_w}.4f}  {sel:>{col_w + 1}.4f}",
                    end="",
                )
            else:
                print(f"  {'—':>{col_w}s}  {'—':>{col_w}s}  {'—':>{col_w}s}", end="")
        print()

    print(f"\n  {'AUC':>8s}", end="")
    for r in results_list:
        print(
            f"  {r['bigram_auc']:>{col_w}.4f}  {r['bigram_control_auc']:>{col_w}.4f}  {r['bigram_selectivity_auc']:>{col_w + 1}.4f}",
            end="",
        )
    print()
    print(f"  {'Baseline':>8s}", end="")
    for r in results_list:
        print(f"  {r['bigram_baseline']:>{col_w}.4f}", end="")
    print()

    # Final scores — uses SELECTIVITY (the credible metric per Hewitt & Liang 2019)
    print(f"\n{'=' * 70}")
    print("PROBE INTERPRETABILITY SCORES (Selectivity = Real - Control)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':>30s}", end="")
    for r in results_list:
        print(f"  {r['name'][:18]:>18s}", end="")
    print()
    print(f"  {'-' * 30}", end="")
    for _ in results_list:
        print(f"  {'-' * 18}", end="")
    print()
    print(f"  {'NTP Selectivity AUC':>30s}", end="")
    for r in results_list:
        print(f"  {r['ntp_selectivity_auc']:>18.4f}", end="")
    print()
    print(f"  {'Bigram Selectivity AUC':>30s}", end="")
    for r in results_list:
        print(f"  {r['bigram_selectivity_auc']:>18.4f}", end="")
    print()
    print(f"  {'Combined Selectivity':>30s}", end="")
    for r in results_list:
        print(f"  {r['probe_interpretability_score']:>18.4f}", end="")
    print()
    print(f"  {'(Raw accuracy, ref only)':>30s}", end="")
    for r in results_list:
        print(f"  {r['probe_raw_score']:>18.4f}", end="")
    print()

    # Delta if two models
    if len(results_list) == 2:
        r0, r1 = results_list
        d_ntp = r0["ntp_selectivity_auc"] - r1["ntp_selectivity_auc"]
        d_bg = r0["bigram_selectivity_auc"] - r1["bigram_selectivity_auc"]
        d_comb = r0["probe_interpretability_score"] - r1["probe_interpretability_score"]
        print(f"\n  {'Delta (A - B)':>30s}  {d_ntp:>+18.4f}  (NTP selectivity)")
        print(f"  {'':>30s}  {d_bg:>+18.4f}  (bigram selectivity)")
        print(f"  {'':>30s}  {d_comb:>+18.4f}  (combined)")

        if d_comb > 0:
            pct = (
                abs(d_comb / r1["probe_interpretability_score"]) * 100
                if r1["probe_interpretability_score"] > 0
                else 0
            )
            print(
                f"\n  >> {r0['name']} is MORE interpretable by probing (+{pct:.1f}% selectivity)"
            )
        elif d_comb < 0:
            pct = (
                abs(d_comb / r0["probe_interpretability_score"]) * 100
                if r0["probe_interpretability_score"] > 0
                else 0
            )
            print(
                f"\n  >> {r1['name']} is MORE interpretable by probing (+{pct:.1f}% selectivity)"
            )
        else:
            print(f"\n  >> Both models have equal probe interpretability")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe interpretability analysis for autoresearch models"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Path(s) to model checkpoint(s). Provide two for comparison.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=5000,
        help="Number of tokens to collect for probe training (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation extraction (default: 4)",
    )
    args = parser.parse_args()

    # Validate checkpoint paths
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"Error: checkpoint not found at {path}")
            sys.exit(1)

    # Device selection
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_type)
    print(f"Device: {device_type}")

    # Load tokenizer
    tokenizer = Tokenizer.from_directory()
    print(f"Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")

    # Analyze each checkpoint
    all_results = []
    for ckpt_path in args.checkpoints:
        results = analyze_checkpoint(
            ckpt_path, device, tokenizer, num_tokens=args.num_tokens
        )
        all_results.append(results)

    # Print comparison
    print_results_table(all_results)
