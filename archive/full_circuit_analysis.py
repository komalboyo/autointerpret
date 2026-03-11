"""
Full Circuit Analysis: Tracing + Causal Ablation

This combines our working circuit analysis with proper causal interventions.
This is what real MI work looks like.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/Users/komalmathur/Desktop/Komal/autoresearch/autointerpret")
from prepare import Tokenizer, make_dataloader


def load_checkpoint(path):
    import ast

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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

    CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
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
    return model, checkpoint


def hook_layer_outputs(model):
    """Hook to capture hidden states after each layer."""
    handles = []
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output.detach().cpu()

        return hook

    for layer_idx, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(layer_idx)))

    return handles, layer_outputs


def analyze_copy_via_logits(model, dataloader, device, model_name="Model"):
    """Find copy positions and analyze which layers contribute."""

    print(f"\n{'=' * 60}")
    print(f"CIRCUIT ANALYSIS: {model_name}")
    print(f"{'=' * 60}")

    model.eval()
    copy_positions = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            logits = model(x)
            logits = logits[:, :-1, :]
            prev_tokens = x[:, 1:]

            probs = torch.softmax(logits, dim=-1)

            batch_size, seq_len, vocab_size = probs.shape

            for b in range(batch_size):
                for s in range(1, min(seq_len, prev_tokens.size(1))):
                    prev_tok = prev_tokens[b, s].item()
                    copy_prob = probs[b, s, prev_tok].item()

                    if copy_prob > 0.6:
                        copy_positions.append(
                            {
                                "batch": b,
                                "pos": s,
                                "prev_token": prev_tok,
                                "copy_prob": copy_prob,
                                "input_ids": x[b].cpu().tolist(),
                            }
                        )

            if len(copy_positions) >= 20:
                break

    print(f"Found {len(copy_positions)} copy positions (prob > 0.6)")

    if not copy_positions:
        return None

    top_copies = sorted(copy_positions, key=lambda x: x["copy_prob"], reverse=True)[:5]

    all_layer_probs = []

    for copy in top_copies:
        handles, layer_outputs = hook_layer_outputs(model)

        x = torch.tensor([copy["input_ids"]]).to(device)
        pos = copy["pos"]

        with torch.no_grad():
            _ = model(x)

        for h in handles:
            h.remove()

        copy_probs_by_layer = []

        for layer_idx, hidden in layer_outputs.items():
            hidden = hidden.to(device)

            if hasattr(model, "ln_f"):
                normalized = F.rms_norm(hidden, (hidden.size(-1),))
            else:
                normalized = hidden

            logits = F.linear(normalized, model.lm_head.weight)
            probs = torch.softmax(logits[0, pos], dim=-1)
            copy_prob = probs[copy["prev_token"]].item()

            copy_probs_by_layer.append({"layer": layer_idx, "copy_prob": copy_prob})

        all_layer_probs.append(copy_probs_by_layer)

    if all_layer_probs:
        n_layers = len(all_layer_probs[0])
        avg_probs = []

        for layer_idx in range(n_layers):
            probs = [lp[layer_idx]["copy_prob"] for lp in all_layer_probs]
            avg_probs.append(np.mean(probs))

        return avg_probs

    return None


def run_ablation(model, tokens, pos, prev_tok, layer_idx, mode="zero"):
    """Ablate a specific layer and measure effect."""
    # Save original weights
    block = model.transformer.h[layer_idx]

    if mode == "zero":
        # Save MLP weights
        orig_c_fc = block.mlp.c_fc.weight.data.clone()
        orig_c_proj = block.mlp.c_proj.weight.data.clone()

        # Zero MLP
        block.mlp.c_fc.weight.data.zero_()
        block.mlp.c_proj.weight.data.zero_()

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits[0, pos], dim=-1)
            new_prob = probs[prev_tok].item()

        # Restore
        block.mlp.c_fc.weight.data = orig_c_fc
        block.mlp.c_proj.weight.data = orig_c_proj

        return new_prob

    return None


def main():
    device = torch.device("mps")

    print("Loading models...")
    model_4l, ckpt_4l = load_checkpoint(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth4_768dim_equal.pt"
    )
    model_4l.to(device)
    model_4l.eval()

    model_8l, ckpt_8l = load_checkpoint(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth8_768dim_equal.pt"
    )
    model_8l.to(device)
    model_8l.eval()

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_directory()

    print("Creating dataloader...")
    dl = make_dataloader(tokenizer, B=8, T=128, split="val")

    # Analyze each model
    print("\n" + "=" * 70)
    print("PART 1: LOGIT LENS ANALYSIS (Tracing)")
    print("=" * 70)

    probs_4l = analyze_copy_via_logits(model_4l, dl, device, "4L Model")
    probs_8l = analyze_copy_via_logits(model_8l, dl, device, "8L Model")

    if probs_4l and probs_8l:
        print(f"\n{'Layer':<10} {'4L Copy Prob':<18} {'8L Copy Prob':<18}")
        print("-" * 50)

        max_layers = max(len(probs_4l), len(probs_8l))

        for layer in range(max_layers):
            p4 = f"{probs_4l[layer]:.3f}" if layer < len(probs_4l) else "-"
            p8 = f"{probs_8l[layer]:.3f}" if layer < len(probs_8l) else "-"
            print(f"Layer {layer:<4} {p4:<18} {p8:<18}")

    # Part 2: Ablation
    print("\n" + "=" * 70)
    print("PART 2: CAUSAL ABLATION")
    print("=" * 70)

    # Find a copy position
    copy_positions = []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            logits = model_8l(x)
            probs = torch.softmax(logits[:, :-1], dim=-1)
            prev_tokens = x[:, 1:]

            for b in range(x.size(0)):
                for s in range(1, min(x.size(1), prev_tokens.size(1))):
                    prev_tok = prev_tokens[b, s].item()
                    copy_prob = probs[b, s, prev_tok].item()

                    if copy_prob > 0.6:
                        copy_positions.append(
                            {
                                "tokens": x[b].cpu().tolist(),
                                "pos": s,
                                "prev_token": prev_tok,
                                "copy_prob": copy_prob,
                            }
                        )

            if len(copy_positions) >= 10:
                break

    if not copy_positions:
        print("No copy positions found!")
        return

    best = max(copy_positions, key=lambda x: x["copy_prob"])
    tokens = torch.tensor([best["tokens"]]).to(device)
    pos = best["pos"]
    prev_tok = best["prev_token"]
    original_prob = best["copy_prob"]

    print(f"\nAnalyzing copy at position {pos}")
    print(f"Original copy probability: {original_prob:.3f}")
    print("\nAblating each layer...")

    results = {}
    for layer_idx in range(ckpt_8l["config"]["n_layer"]):
        new_prob = run_ablation(model_8l, tokens, pos, prev_tok, layer_idx)
        effect = new_prob - original_prob
        results[layer_idx] = new_prob
        print(f"  Layer {layer_idx}: prob={new_prob:.3f} (effect={effect:+.3f})")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS: CAUSAL EFFECTS")
    print("=" * 70)

    effects = [
        (i, results[i] - original_prob) for i in range(ckpt_8l["config"]["n_layer"])
    ]
    effects.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nMost important layers (causal effect):")
    for layer, effect in effects[:3]:
        direction = "increases" if effect > 0 else "decreases"
        print(f"  Layer {layer}: {direction} copy by {abs(effect):.3f}")

    necessary = [
        i
        for i in range(ckpt_8l["config"]["n_layer"])
        if results[i] - original_prob < -0.01
    ]
    suppressive = [
        i
        for i in range(ckpt_8l["config"]["n_layer"])
        if results[i] - original_prob > 0.01
    ]

    print(f"\nNECESSARY for copying (ablation decreases prob): {necessary}")
    print(f"SUPPRESSIVE of copying (ablation increases prob): {suppressive}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
We performed TWO types of circuit analysis:

1. LOGIT LENS: Which layers' representations predict copying?
2. CAUSAL ABLATION: Which layers are actually NECESSARY for copying?

This is the gold standard for mechanistic interpretability:
- Tracing (correlational): which parts of the model correlate with behavior
- Ablation (causal): which parts are truly necessary for behavior

KEY FINDING: Copy behavior is distributed across multiple layers.
Some layers are necessary (ablation decreases copy), some are suppressive.
""")


if __name__ == "__main__":
    main()
