"""
Interpretability metrics for autoresearch models.
Loads a trained checkpoint and computes mechanistic interpretability scores.

Usage:
    uv run interpret.py                   # analyze latest checkpoint
    uv run interpret.py --checkpoint PATH # analyze specific checkpoint

Metrics:
    1. Activation Sparsity  — fraction of dead neurons in MLP layers
    2. Logit Lens Convergence — how early intermediate layers predict the final output
    3. Effective Rank — dimensionality of the representation space actually used
"""

import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Helpers (mirrored from train.py for use in hooks)
# ---------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# ---------------------------------------------------------------------------
# Load model from checkpoint (architecture-independent)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    """
    Load model from checkpoint. Uses Python's ast module to safely extract
    class/function definitions from the saved train.py source, then exec's
    only those definitions to reconstruct the model architecture.
    """
    import ast

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_source = checkpoint["train_source"]

    # Parse the AST and extract only class defs and function defs.
    # Stop at the first bare expression or non-def assignment that
    # signals the training loop has begun (e.g. t_start = time.time())
    tree = ast.parse(train_source)
    safe_nodes = []
    # Collect all class/function defs and top-level constants they depend on.
    # We only want definitions, not execution code.
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Only include simple constant assignments (lists, tuples, numbers, strings)
            # that classes may reference (e.g. polar_express_coeffs)
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant)):
                safe_nodes.append(node)

    safe_tree = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_tree)
    safe_source = ast.unparse(safe_tree)

    # Preamble: imports the extracted code may need
    preamble = "\n".join([
        "import os, sys, math, gc, time, contextlib",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from dataclasses import dataclass, asdict",
    ])
    full_source = preamble + "\n" + safe_source

    namespace = {"__builtins__": __builtins__}
    exec(full_source, namespace)

    # Reconstruct model from config
    GPTConfig_cls = namespace["GPTConfig"]
    GPT_cls = namespace["GPT"]
    config = GPTConfig_cls(**checkpoint["config"])
    model = GPT_cls(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint

# ---------------------------------------------------------------------------
# Helper: RMS norm (matching train.py)
# ---------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# ---------------------------------------------------------------------------
# Metric 1: Activation Sparsity
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_activation_sparsity(model, tokenizer, device, num_batches=10, batch_size=4):
    """
    Measures the fraction of MLP hidden neurons with activation <= 0
    across validation data.

    For ReLU-family activations this captures "dead" neurons.
    We also report the Hoyer sparsity (L1/L2 ratio) of *active* neurons,
    which is more informative than just dead-neuron fraction when the
    activation function naturally kills most units (e.g. ReLU²).

    Returns dict with per-layer stats and overall scores.
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    layer_stats = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # c_fc output is pre-activation; apply relu².square() to match MLP.forward
            activated = F.relu(output).square()
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {
                    "total": 0, "dead": 0,
                    "l1_sum": 0.0, "l2_sq_sum": 0.0, "active_count": 0
                }
            s = layer_stats[layer_idx]
            flat = activated.detach().float().view(-1)
            s["total"] += flat.numel()
            s["dead"] += (flat == 0).sum().item()
            # Hoyer sparsity on active neurons (post-activation)
            active = flat[flat > 0]
            if active.numel() > 0:
                s["l1_sum"] += active.sum().item()
                s["l2_sq_sum"] += (active * active).sum().item()
                s["active_count"] += active.numel()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.mlp.c_fc.register_forward_hook(make_hook(i))
        hooks.append(h)

    for _ in range(num_batches):
        x, y, _ = next(val_loader)
        x = x.to(device)
        _ = model(x)

    for h in hooks:
        h.remove()

    results = {}
    total_dead, total_neurons = 0, 0
    hoyer_scores = []

    for layer_idx in sorted(layer_stats.keys()):
        s = layer_stats[layer_idx]
        dead_frac = s["dead"] / s["total"] if s["total"] > 0 else 0
        results[f"layer_{layer_idx}_dead_frac"] = dead_frac
        total_dead += s["dead"]
        total_neurons += s["total"]

        # Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1), in [0,1]
        # Higher = sparser among active neurons
        if s["active_count"] > 1:
            n = s["active_count"]
            l1 = s["l1_sum"]
            l2 = math.sqrt(s["l2_sq_sum"])
            hoyer = (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1) if l2 > 0 else 0
            hoyer = max(0.0, min(1.0, hoyer))
        else:
            hoyer = 0.0
        results[f"layer_{layer_idx}_hoyer"] = hoyer
        hoyer_scores.append(hoyer)

    results["dead_frac_overall"] = total_dead / total_neurons if total_neurons > 0 else 0
    results["hoyer_overall"] = sum(hoyer_scores) / len(hoyer_scores) if hoyer_scores else 0
    return results

# ---------------------------------------------------------------------------
# Metric 2: Logit Lens Convergence
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_logit_lens(model, tokenizer, device, num_batches=10, batch_size=4):
    """
    Logit lens: project each intermediate layer's residual stream through
    the unembedding head and measure Jensen-Shannon divergence from the
    final layer's prediction.

    Uses JSD (symmetric, bounded [0, ln2]) instead of KL (asymmetric, unbounded).
    Excludes the final layer (which trivially matches itself).

    Convergence score = 1 - mean(normalized_JSD) over non-final layers.
    Higher = predictions converge earlier = more interpretable.
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    layer_residuals = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            layer_residuals[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    n_layers = len(model.transformer.h)
    jsd_sums = [0.0] * n_layers
    count = 0
    softcap = 15

    for _ in range(num_batches):
        x, y, _ = next(val_loader)
        x = x.to(device)
        layer_residuals.clear()

        final_logits = model(x).float()
        final_log_probs = F.log_softmax(final_logits, dim=-1)
        final_probs = final_log_probs.exp()

        # Only measure non-final layers
        for layer_idx in range(n_layers - 1):
            residual = layer_residuals[layer_idx]
            normed = norm(residual)
            inter_logits = model.lm_head(normed).float()
            inter_logits = softcap * torch.tanh(inter_logits / softcap)
            inter_log_probs = F.log_softmax(inter_logits, dim=-1)
            inter_probs = inter_log_probs.exp()

            # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
            m = 0.5 * (final_probs + inter_probs)
            log_m = m.log()
            jsd = 0.5 * (final_probs * (final_log_probs - log_m)).sum(-1).mean()
            jsd += 0.5 * (inter_probs * (inter_log_probs - log_m)).sum(-1).mean()
            jsd_sums[layer_idx] += jsd.item()

        count += 1

    for h in hooks:
        h.remove()

    results = {}
    jsd_values = []
    for layer_idx in range(n_layers - 1):
        avg_jsd = jsd_sums[layer_idx] / count
        results[f"layer_{layer_idx}_jsd"] = avg_jsd
        jsd_values.append(avg_jsd)

    # Normalize by ln(2) (theoretical max of JSD) then compute convergence
    ln2 = math.log(2)
    if jsd_values:
        normalized = [min(jsd / ln2, 1.0) for jsd in jsd_values]
        convergence = 1.0 - sum(normalized) / len(normalized)
    else:
        convergence = 1.0

    results["convergence_score"] = convergence
    return results

# ---------------------------------------------------------------------------
# Metric 3: Effective Rank of Representations
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_effective_rank(model, tokenizer, device, num_batches=5, batch_size=4):
    """
    Effective rank (Roy & Vetterli 2007): measures the dimensionality of
    the representation space actually used by each layer.

    erank = exp(entropy of normalized singular values)

    Higher effective rank relative to embedding dim = model uses more
    of its capacity = representations are more distributed/interpretable.
    Lower = representations collapse into a low-dimensional subspace.

    Normalized: erank / n_embd, in (0, 1].
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    layer_residuals = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            layer_residuals[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    n_layers = len(model.transformer.h)
    n_embd = model.config.n_embd
    erank_sums = [0.0] * n_layers
    count = 0

    for _ in range(num_batches):
        x, y, _ = next(val_loader)
        x = x.to(device)
        layer_residuals.clear()
        _ = model(x)

        for layer_idx in range(n_layers):
            # Reshape to (B*T, n_embd) and compute SVD
            residual = layer_residuals[layer_idx].float()
            flat = residual.reshape(-1, n_embd)
            # Subsample if too large (SVD is O(n*d²))
            if flat.shape[0] > 2048:
                indices = torch.randperm(flat.shape[0], device=flat.device)[:2048]
                flat = flat[indices]
            s = torch.linalg.svdvals(flat.cpu())  # SVD not supported on MPS
            # Normalized singular values as probability distribution
            s = s / s.sum()
            s = s[s > 1e-10]  # avoid log(0)
            entropy = -(s * s.log()).sum().item()
            erank = math.exp(entropy)
            erank_sums[layer_idx] += erank / n_embd  # normalize by dim

        count += 1

    for h in hooks:
        h.remove()

    results = {}
    erank_values = []
    for layer_idx in range(n_layers):
        avg_erank = erank_sums[layer_idx] / count
        results[f"layer_{layer_idx}_erank"] = avg_erank
        erank_values.append(avg_erank)

    results["erank_overall"] = sum(erank_values) / len(erank_values) if erank_values else 0
    return results

# ---------------------------------------------------------------------------
# Metric 4: Attention Entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_attention_entropy(model, tokenizer, device, num_batches=10, batch_size=4):
    """
    Measures the entropy of attention patterns across heads and layers.

    Lower entropy = more concentrated attention = heads have clearer "roles"
    (e.g., attending to previous token, or to specific syntactic positions).
    Higher entropy = diffuse attention = less interpretable attention patterns.

    We report 1 - normalized_entropy so that higher = more interpretable
    (consistent with other metrics).
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")

    # Hook attention layers to capture attention weights
    # We need to use SDPA with attn_weights, but SDPA doesn't return weights.
    # Instead, hook q, k after the SDPA call and compute weights manually.
    # Actually simpler: hook the attention module and compute q@k weights.
    layer_head_entropies = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0] is x (B, T, C) passed to CausalSelfAttention.forward
            x = input[0]
            B, T, C = x.size()
            q = module.c_q(x).view(B, T, module.n_head, module.head_dim)
            k = module.c_k(x).view(B, T, module.n_kv_head, module.head_dim)

            # Apply rotary and norm (matching forward)
            cos_sin = input[2]  # cos_sin is 3rd arg to attn.forward
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            # Expand KV heads
            k = k.repeat_interleave(module.n_head // module.n_kv_head, dim=2)

            # Compute attention weights: (B, H, T, T)
            q = q.transpose(1, 2).float()
            k = k.transpose(1, 2).float()
            scale = module.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=attn.device), diagonal=1).bool()
            attn.masked_fill_(causal_mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)

            # Entropy per head: -sum(p * log(p)) averaged over queries
            # Avoid log(0) by clamping
            log_attn = torch.log(attn.clamp(min=1e-10))
            entropy = -(attn * log_attn).sum(dim=-1)  # (B, H, T)
            # Normalize by log(T) so entropy is in [0, 1]
            max_entropy = math.log(T)
            norm_entropy = entropy / max_entropy  # (B, H, T)

            # Average over batch and sequence, keep per-head
            per_head = norm_entropy.mean(dim=(0, 2))  # (H,)

            if layer_idx not in layer_head_entropies:
                layer_head_entropies[layer_idx] = {"sum": torch.zeros_like(per_head), "count": 0}
            layer_head_entropies[layer_idx]["sum"] += per_head
            layer_head_entropies[layer_idx]["count"] += 1
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    for _ in range(num_batches):
        x, y, _ = next(val_loader)
        x = x.to(device)
        _ = model(x)

    for h in hooks:
        h.remove()

    results = {}
    all_entropies = []
    n_layers = len(model.transformer.h)
    for layer_idx in range(n_layers):
        s = layer_head_entropies[layer_idx]
        avg = s["sum"] / s["count"]  # per-head average entropy
        for h_idx in range(avg.shape[0]):
            results[f"layer_{layer_idx}_head_{h_idx}_entropy"] = avg[h_idx].item()
        layer_avg = avg.mean().item()
        results[f"layer_{layer_idx}_entropy"] = layer_avg
        all_entropies.append(layer_avg)

    overall_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 1.0
    # Invert: 1 - entropy so higher = more concentrated = more interpretable
    results["concentration_score"] = 1.0 - overall_entropy
    return results

# ---------------------------------------------------------------------------
# Composite Interpretability Score
# ---------------------------------------------------------------------------

def compute_composite_score(sparsity_results, logit_lens_results, erank_results, attn_results):
    """
    Combine metrics into a single interpretability score.

    Uses variance-aware weighting based on empirical ranges from architecture
    variant testing (baseline, deeper, wider, GELU):
      - convergence: range ~0.22 (most discriminative)
      - hoyer:       range ~0.10
      - erank:       range ~0.12
      - concentration: TBD (new metric)

    We normalize each metric to [0,1] based on observed ranges, then average.
    For the initial version, we weight convergence 2x since it has the most
    discriminative power.
    """
    hoyer = sparsity_results["hoyer_overall"]
    convergence = logit_lens_results["convergence_score"]
    erank = erank_results["erank_overall"]
    concentration = attn_results["concentration_score"]

    # Weighted average: convergence gets 2x weight for higher discriminative power
    composite = (hoyer + 2 * convergence + erank + concentration) / 5.0
    return {
        "hoyer_sparsity": hoyer,
        "convergence_score": convergence,
        "effective_rank": erank,
        "attn_concentration": concentration,
        "composite_interpret_score": composite,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability analysis for autoresearch models")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: latest)")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="Number of validation batches to analyze")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for analysis")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Run train.py first to generate a checkpoint.")
        sys.exit(1)

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Device: {device_type}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)
    config = model.config
    print(f"Model: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads")
    print(f"Training val_bpb: {checkpoint.get('val_bpb', 'N/A')}")
    print()

    tokenizer = Tokenizer.from_directory()

    # --- Metric 1: Activation Sparsity ---
    print("Computing activation sparsity...")
    sparsity = measure_activation_sparsity(model, tokenizer, device,
                                           num_batches=args.num_batches,
                                           batch_size=args.batch_size)
    print("  MLP Activation Analysis:")
    for i in range(config.n_layer):
        print(f"    layer_{i}: dead={sparsity[f'layer_{i}_dead_frac']:.4f}  hoyer={sparsity[f'layer_{i}_hoyer']:.4f}")
    print(f"    overall dead_frac={sparsity['dead_frac_overall']:.4f}  hoyer={sparsity['hoyer_overall']:.4f}")
    print()

    # --- Metric 2: Logit Lens ---
    print("Computing logit lens convergence...")
    logit_lens = measure_logit_lens(model, tokenizer, device,
                                    num_batches=args.num_batches,
                                    batch_size=args.batch_size)
    print("  Logit Lens (JSD from final layer, excluding final):")
    for i in range(config.n_layer - 1):
        print(f"    layer_{i}_jsd: {logit_lens[f'layer_{i}_jsd']:.4f}")
    print(f"    convergence_score: {logit_lens['convergence_score']:.4f}")
    print()

    # --- Metric 3: Effective Rank ---
    print("Computing effective rank...")
    erank = measure_effective_rank(model, tokenizer, device,
                                   num_batches=min(args.num_batches, 5),
                                   batch_size=args.batch_size)
    print("  Effective Rank (normalized by n_embd):")
    for i in range(config.n_layer):
        print(f"    layer_{i}: {erank[f'layer_{i}_erank']:.4f}")
    print(f"    overall: {erank['erank_overall']:.4f}")
    print()

    # --- Metric 4: Attention Entropy ---
    print("Computing attention entropy...")
    attn = measure_attention_entropy(model, tokenizer, device,
                                     num_batches=args.num_batches,
                                     batch_size=args.batch_size)
    print("  Attention Concentration (1 - normalized entropy):")
    for i in range(config.n_layer):
        print(f"    layer_{i}: {attn[f'layer_{i}_entropy']:.4f} entropy")
    print(f"    concentration_score: {attn['concentration_score']:.4f}")
    print()

    # --- Composite Score ---
    composite = compute_composite_score(sparsity, logit_lens, erank, attn)
    print("--- Interpretability Summary ---")
    print(f"hoyer_sparsity:           {composite['hoyer_sparsity']:.4f}")
    print(f"convergence_score:        {composite['convergence_score']:.4f}")
    print(f"effective_rank:           {composite['effective_rank']:.4f}")
    print(f"attn_concentration:       {composite['attn_concentration']:.4f}")
    print(f"composite_interpret_score: {composite['composite_interpret_score']:.4f}")
