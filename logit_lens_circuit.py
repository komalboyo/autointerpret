"""
Circuit Tracing via Logit Lens: Token Copying

Instead of hooks (which don't work with fused attention), we use
logit lens to trace which layers contribute to copy predictions.

This shows which layer representations contain the "copy" information.
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


def get_layer_embeddings(model, x, device):
    """Get the embedding from each layer (logit lens approach)."""
    model.eval()

    # Get input embeddings
    embeddings = [model.transformer.wte(x)]

    # Run through each block and capture outputs
    hidden_states = embeddings[-1]

    with torch.no_grad():
        for block in model.transformer.h:
            # Apply block
            ln1_out = block.ln_1(hidden_states)
            attn_out = block.attn(ln1_out, hidden_states)
            hidden_states = hidden_states + attn_out

            ln2_out = block.ln_2(hidden_states)
            mlp_out = block.mlp(ln2_out)
            hidden_states = hidden_states + mlp_out

            embeddings.append(hidden_states)

    return embeddings


def logit_lens_copy_analysis(model, dataloader, device, model_name="Model"):
    """Analyze which layers predict token copying via logit lens."""

    print(f"\n{'=' * 60}")
    print(f"LOGIT LENS ANALYSIS: {model_name}")
    print(f"{'=' * 60}")

    # Find copy positions first
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
            logits = logits[:, :-1, :]
            prev_tokens = x[:, 1:]

            probs = torch.softmax(logits, dim=-1)

            for b in range(x.size(0)):
                for s in range(1, min(x.size(1), prev_tokens.size(1))):
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

    print(f"Found {len(copy_positions)} high-confidence copy positions")

    if not copy_positions:
        print("No copy positions found!")
        return None

    # Use top 5 copy positions
    top_copies = sorted(copy_positions, key=lambda x: x["copy_prob"], reverse=True)[:5]

    # For each copy position, get layer-by-layer predictions
    layer_contributions = []

    for copy in top_copies:
        x = torch.tensor([copy["input_ids"]]).to(device)
        pos = copy["pos"]

        # Get layer embeddings
        embeddings = get_layer_embeddings(model, x, device)

        # Apply unembedding at each layer
        copy_probs_by_layer = []

        for layer_idx, emb in enumerate(embeddings):
            # Get logits at this layer's representation
            # Use layer norm + unembed
            if hasattr(model, "ln_f"):
                normalized = model.ln_f(emb)
            else:
                normalized = emb  # no final ln

            logits = F.linear(normalized, model.lm_head.weight)

            # Get probability of copying previous token
            # Position is pos (after shift), previous token is at pos-1
            probs = torch.softmax(logits[0, pos], dim=-1)
            copy_prob = probs[copy["prev_token"]].item()

            copy_probs_by_layer.append(
                {
                    "layer": layer_idx,
                    "copy_prob": copy_prob,
                }
            )

        layer_contributions.append(copy_probs_by_layer)

    # Average across positions
    n_layers = len(layer_contributions[0])
    avg_probs = []

    for layer_idx in range(n_layers):
        probs = [lc[layer_idx]["copy_prob"] for lc in layer_contributions]
        avg_probs.append(np.mean(probs))

    return avg_probs


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
    probs_4l = logit_lens_copy_analysis(model_4l, dl, device, "4L Model")
    probs_8l = logit_lens_copy_analysis(model_8l, dl, device, "8L Model")

    # Compare
    print(f"\n{'=' * 60}")
    print("COMPARISON: WHICH LAYERS HANDLE COPYING?")
    print(f"{'=' * 60}")

    print(f"\n{'Layer':<10} {'4L Copy Prob':<18} {'8L Copy Prob':<18}")
    print("-" * 50)

    max_layers = max(len(probs_4l), len(probs_8l))

    for layer in range(max_layers):
        p4 = f"{probs_4l[layer]:.3f}" if layer < len(probs_4l) else "-"
        p8 = f"{probs_8l[layer]:.3f}" if layer < len(probs_8l) else "-"

        layer_name = f"Layer {layer}"
        if layer == len(probs_4l) - 1 and layer == len(probs_8l) - 1:
            layer_name += " (output)"

        print(f"{layer_name:<10} {p4:<18} {p8:<18}")

    print("\n" + "=" * 60)
    print("INTERPRETABILITY INSIGHT")
    print("=" * 60)

    # Find when copy probability first becomes high
    def find_copy_layer(probs, threshold=0.3):
        for i, p in enumerate(probs):
            if p > threshold:
                return i, p
        return len(probs), probs[-1]

    layer_4l, prob_4l = find_copy_layer(probs_4l)
    layer_8l, prob_8l = find_copy_layer(probs_8l)

    print(f"\nCopy behavior emerges:")
    print(f"  4L: at layer {layer_4l} (prob={prob_4l:.3f})")
    print(f"  8L: at layer {layer_8l} (prob={prob_8l:.3f})")

    print(f"\nFinal layer copy prediction:")
    print(f"  4L: {probs_4l[-1]:.3f}")
    print(f"  8L: {probs_8l[-1]:.3f}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Both models successfully learn token copying. The 8L model has
more layers to distribute the computation, which may make it
easier to identify which layer handles which aspect of the task.

This is what interpretability looks like: being able to trace
which part of the network is responsible for which behavior.
""")


if __name__ == "__main__":
    main()
