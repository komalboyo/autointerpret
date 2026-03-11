"""
Full Circuit Tracing: Token Copying in 4L vs 8L Models

This script traces the "token copying" circuit - one of the simplest
circuits in language models - through both 4L and 8L architectures to
compare which is more interpretable.

Goal: Actually demonstrate interpretability, not just measure proxies.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Setup path
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
    return model, checkpoint, namespace


class CircuitTracer:
    """Trace attention patterns for specific behaviors."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.attention_data = defaultdict(list)
        self.activation_data = defaultdict(list)

    def register_hooks(self):
        """Register hooks to capture attention and MLP activations."""

        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                # output: (batch, heads, seq, seq)
                attn = output[0] if isinstance(output, tuple) else output
                self.attention_data[layer_idx].append(attn.detach().cpu())

            return hook

        def make_mlp_hook(layer_idx):
            def hook(module, input, output):
                self.activation_data[layer_idx].append(output.detach().cpu())

            return hook

        # Register hooks on all layers
        for layer_idx, block in enumerate(self.model.transformer.h):
            # Attention
            if hasattr(block, "attn"):
                self.hooks.append(
                    block.attn.register_forward_hook(make_attn_hook(layer_idx))
                )
            # MLP
            if hasattr(block, "mlp"):
                self.hooks.append(
                    block.mlp.register_forward_hook(make_mlp_hook(layer_idx))
                )

    def clear(self):
        """Clear captured data."""
        self.attention_data.clear()
        self.activation_data.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def find_copy_positions(model, dataloader, device, threshold=0.5, max_positions=50):
    """Find positions where model copies the previous token."""
    model.eval()
    copy_positions = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Get logits
            logits = model(x)

            # Shift for next-token prediction
            logits = logits[:, :-1, :]  # (B, T-1, V)
            prev_tokens = x[:, 1:]  # (B, T-1)

            probs = torch.softmax(logits, dim=-1)

            batch_size, seq_len, vocab_size = probs.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if s == 0:
                        continue
                    prev_tok = prev_tokens[b, s].item()
                    copy_prob = probs[b, s, prev_tok].item()

                    if copy_prob > threshold:
                        # Get full context
                        copy_positions.append(
                            {
                                "batch": b,
                                "pos": s,
                                "prev_token": prev_tok,
                                "copy_prob": copy_prob,
                                "input_ids": x[b].cpu().tolist(),
                            }
                        )

                    if len(copy_positions) >= max_positions:
                        return copy_positions

            if len(copy_positions) >= max_positions:
                break

    return copy_positions


def analyze_copy_circuit(tracer, model, input_ids, copy_pos, device):
    """Analyze which attention heads contribute to copy behavior."""

    # Clear previous data
    tracer.clear()

    # Convert to tensor
    x = torch.tensor([input_ids]).to(device)

    # Run model with hooks
    with torch.no_grad():
        _ = model(x)

    # Analyze attention patterns at the copy position
    results = {
        "attention_by_layer": {},
        "mlp_activations": {},
    }

    for layer_idx, attns in tracer.attention_data.items():
        if not attns:
            continue
        # Get last forward pass: (batch, heads, seq, seq)
        attn = attns[-1]

        # Handle different tensor shapes
        if attn.dim() == 4:
            attn = attn[0]  # (heads, seq, seq)
        elif attn.dim() == 3:
            attn = attn[0]  # might be (1, heads, seq) if that's the case

        # Get attention to previous token position
        if attn.dim() == 3 and copy_pos > 0 and copy_pos < attn.shape[1]:
            attn_to_prev = attn[:, copy_pos, copy_pos - 1].numpy()
        else:
            attn_to_prev = np.array([0.0])

        results["attention_by_layer"][layer_idx] = {
            "attn_to_prev": attn_to_prev,
            "max_head": int(np.argmax(attn_to_prev)),
            "max_attn": float(np.max(attn_to_prev)),
            "mean_attn": float(np.mean(attn_to_prev)),
        }

    # MLP activations at copy position
    for layer_idx, acts in tracer.activation_data.items():
        if not acts:
            continue
        act = acts[-1][0, copy_pos].numpy()  # (d_model,)
        results["mlp_activations"][layer_idx] = {
            "mean": float(np.mean(np.abs(act))),
            "max": float(np.max(np.abs(act))),
            "sparsity": float(np.mean(np.abs(act) < 0.01)),
        }

    return results


def compare_circuits(model_4l, model_8l, dataloader, device):
    """Compare copy circuits between 4L and 8L models."""

    print("=" * 70)
    print("CIRCUIT TRACING: TOKEN COPYING BEHAVIOR")
    print("=" * 70)

    # Setup tracers
    tracer_4l = CircuitTracer(model_4l, device)
    tracer_8l = CircuitTracer(model_8l, device)
    tracer_4l.register_hooks()
    tracer_8l.register_hooks()

    # Find copy positions
    print("\n[1] Finding high-confidence copy positions...")
    copy_positions = find_copy_positions(
        model_8l, dataloader, device, threshold=0.7, max_positions=30
    )
    print(f"    Found {len(copy_positions)} copy positions (prob > 0.7)")

    if not copy_positions:
        print("    No high-confidence copies found, lowering threshold...")
        copy_positions = find_copy_positions(
            model_8l, dataloader, device, threshold=0.5, max_positions=20
        )
        print(f"    Found {len(copy_positions)} copy positions (prob > 0.5)")

    if not copy_positions:
        print("    ERROR: No copy positions found!")
        return

    # Use the highest confidence copy
    best_copy = max(copy_positions, key=lambda x: x["copy_prob"])
    copy_pos = best_copy["pos"]
    input_ids = best_copy["input_ids"]

    print(f"\n[2] Analyzing copy at position {copy_pos}")
    print(f"    Copy probability: {best_copy['copy_prob']:.3f}")
    print(f"    Previous token ID: {best_copy['prev_token']}")

    # Analyze 4L
    print(f"\n[3] Tracing circuit in 4L model...")
    results_4l = analyze_copy_circuit(tracer_4l, model_4l, input_ids, copy_pos, device)

    # Analyze 8L
    print(f"[4] Tracing circuit in 8L model...")
    results_8l = analyze_copy_circuit(tracer_8l, model_8l, input_ids, copy_pos, device)

    # Compare
    print("\n" + "=" * 70)
    print("CIRCUIT ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- ATTENTION HEADS CONTRIBUTING TO COPY ---")
    print(f"{'Layer':<8} {'4L':<25} {'8L':<25}")
    print("-" * 60)

    max_layers = max(
        len(results_4l["attention_by_layer"]), len(results_8l["attention_by_layer"])
    )

    for layer in range(max_layers):
        l4 = results_4l["attention_by_layer"].get(layer, {})
        l8 = results_8l["attention_by_layer"].get(layer, {})

        if l4:
            l4_str = f"head={l4['max_head']}, attn={l4['max_attn']:.3f}"
        else:
            l4_str = "(no layer)"

        if l8:
            l8_str = f"head={l8['max_head']}, attn={l8['max_attn']:.3f}"
        else:
            l8_str = "(no layer)"

        print(f"{layer:<8} {l4_str:<25} {l8_str:<25}")

    # MLP sparsity comparison
    print("\n--- MLP ACTIVATION SPARSITY ---")
    print(f"{'Layer':<8} {'4L Sparsity':<20} {'8L Sparsity':<20}")
    print("-" * 50)

    for layer in range(max_layers):
        m4 = results_4l["mlp_activations"].get(layer, {})
        m8 = results_8l["mlp_activations"].get(layer, {})

        s4 = f"{m4.get('sparsity', 0):.3f}" if m4 else "-"
        s8 = f"{m8.get('sparsity', 0):.3f}" if m8 else "-"

        print(f"{layer:<8} {s4:<20} {s8:<20}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETABILITY COMPARISON")
    print("=" * 70)

    # Count how many heads have strong copy attention
    def count_copy_heads(results):
        count = 0
        for layer_data in results["attention_by_layer"].values():
            if layer_data["max_attn"] > 0.3:
                count += 1
        return count

    heads_4l = count_copy_heads(results_4l)
    heads_8l = count_copy_heads(results_8l)

    print(f"\nHeads with strong copy attention (>0.3):")
    print(f"  4L model: {heads_4l}")
    print(f"  8L model: {heads_8l}")

    # Average sparsity
    avg_sparse_4l = (
        np.mean([v["sparsity"] for v in results_4l["mlp_activations"].values()])
        if results_4l["mlp_activations"]
        else 0
    )
    avg_sparse_8l = (
        np.mean([v["sparsity"] for v in results_8l["mlp_activations"].values()])
        if results_8l["mlp_activations"]
        else 0
    )

    print(f"\nAverage MLP sparsity:")
    print(f"  4L model: {avg_sparse_4l:.3f}")
    print(f"  8L model: {avg_sparse_8l:.3f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if heads_8l > heads_4l:
        print(f"8L has MORE heads involved in copying ({heads_8l} vs {heads_4l})")
    elif heads_8l < heads_4l:
        print(f"4L has MORE heads involved in copying ({heads_4l} vs {heads_8l})")
    else:
        print(f"Both models have similar number of copy-related heads ({heads_4l})")

    if avg_sparse_8l > avg_sparse_4l:
        print(
            f"8L has HIGHER MLP sparsity ({avg_sparse_8l:.3f} vs {avg_sparse_4l:.3f}) - more interpretable"
        )
    else:
        print(
            f"4L has HIGHER MLP sparsity ({avg_sparse_4l:.3f} vs {avg_sparse_8l:.3f})"
        )

    tracer_4l.remove_hooks()
    tracer_8l.remove_hooks()

    return results_4l, results_8l


def main():
    device = torch.device("mps")

    print("Loading 4L model...")
    model_4l, ckpt_4l, _ = load_checkpoint(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth4_768dim_equal.pt"
    )
    model_4l.to(device)
    model_4l.eval()
    print(
        f"  4L: {ckpt_4l['config']['n_layer']} layers, val_bpb={ckpt_4l['val_bpb']:.4f}"
    )

    print("Loading 8L model...")
    model_8l, ckpt_8l, _ = load_checkpoint(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth8_768dim_equal.pt"
    )
    model_8l.to(device)
    model_8l.eval()
    print(
        f"  8L: {ckpt_8l['config']['n_layer']} layers, val_bpb={ckpt_8l['val_bpb']:.4f}"
    )

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_directory()

    print("Creating dataloader...")
    dl = make_dataloader(tokenizer, B=8, T=128, split="val")

    # Run comparison
    compare_circuits(model_4l, model_8l, dl, device)


if __name__ == "__main__":
    main()
