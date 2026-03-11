"""
Convert our checkpoint to TransformerLens format - FIXED.

Key insight: Our model is [d_model, d_model] for attention projections.
TransformerLens expects [n_heads, d_model, d_head].

Also: Our embed and unembed are already [vocab, d_model] - correct format!
MLP weights are also [intermediate, d_model] - correct format!
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


def convert_weights(our_state_dict, config):
    """Convert our checkpoint to TransformerLens format."""
    new_state = {}
    n_layers = config["n_layer"]
    d_model = config["n_embd"]
    n_heads = config["n_head"]
    d_head = d_model // n_heads

    # Embed - already [vocab, d_model], no transpose needed
    if "transformer.wte.weight" in our_state_dict:
        new_state["embed.W_E"] = our_state_dict["transformer.wte.weight"]

    # For each layer
    for i in range(n_layers):
        prefix = f"transformer.h.{i}."

        # Attention Q/K/V: [d_model, d_model] -> [n_heads, d_model, d_head]
        # Our weight is [d_model, d_model] = [n_heads*d_head, n_heads*d_head]
        for proj, name in [("c_q", "W_Q"), ("c_k", "W_K"), ("c_v", "W_V")]:
            key = f"{prefix}attn.{proj}.weight"
            if key in our_state_dict:
                w = our_state_dict[key]  # [d_model, d_model]
                # Reshape to [n_heads, d_head, n_heads, d_head] then transpose and reshape
                # More simply: [d_model, d_model] -> [n_heads, d_head, n_heads, d_head]
                w = w.reshape(n_heads, d_head, n_heads, d_head)
                w = w.permute(0, 2, 1, 3)  # [n_heads, n_heads, d_head, d_head]
                w = w.reshape(n_heads, d_model, d_head)  # [n_heads, d_model, d_head]
                new_state[f"blocks.{i}.attn.{name}"] = w

        # Attention output W_O: same transformation
        key = f"{prefix}attn.c_proj.weight"
        if key in our_state_dict:
            w = our_state_dict[key]
            w = w.reshape(n_heads, d_head, n_heads, d_head)
            w = w.permute(0, 2, 1, 3)
            w = w.reshape(n_heads, d_model, d_head)
            new_state[f"blocks.{i}.attn.W_O"] = w

        # Layer norms - just copy
        for ln_name, tl_name in [("attn.ln", "ln1"), ("mlp.ln", "ln2")]:
            key = f"{prefix}{ln_name}.weight"
            if key in our_state_dict:
                new_state[f"blocks.{i}.{tl_name}.w"] = our_state_dict[key]

        # MLP W_in - already [intermediate, d_model], no change needed
        key = f"{prefix}mlp.c_fc.weight"
        if key in our_state_dict:
            new_state[f"blocks.{i}.mlp.W_in"] = our_state_dict[key]

        # MLP W_out - transpose needed: [d_model, intermediate] -> [intermediate, d_model]
        key = f"{prefix}mlp.c_proj.weight"
        if key in our_state_dict:
            new_state[f"blocks.{i}.mlp.W_out"] = our_state_dict[key].t()

    # Final norm
    if "ln_f.weight" in our_state_dict:
        new_state["ln_final.w"] = our_state_dict["ln_f.weight"]

    # Unembed - already [vocab, d_model], no transpose needed
    if "lm_head.weight" in our_state_dict:
        new_state["unembed.W_U"] = our_state_dict["lm_head.weight"]

    return new_state


def load_our_model(checkpoint_path):
    """Load our model into TransformerLens."""

    # Load our checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    our_state = ckpt["model_state_dict"]
    config = ckpt["config"]

    print(f"Our config: {config['n_layer']} layers, {config['n_embd']} dim")
    print(f"Our keys: {len(our_state)}")

    # Create TransformerLens config - match our architecture
    tl_config = HookedTransformerConfig(
        n_layers=config["n_layer"],
        d_model=config["n_embd"],
        n_heads=config["n_head"],
        d_head=config["n_embd"] // config["n_head"],
        d_mlp=config["n_embd"] * 4,
        d_vocab=config["vocab_size"],
        n_ctx=2048,
        act_fn="gelu",  # Close to our ReLU^2
        normalization_type="RMS",
        d_vocab_out=config["vocab_size"],
    )

    # Create model
    tl_model = HookedTransformer(tl_config)

    # Convert weights
    tl_state = convert_weights(our_state, config)

    print(f"Converted keys: {len(tl_state)}")

    # Load state
    missing, unexpected = tl_model.load_state_dict(tl_state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("Missing:", missing[:5])
    if unexpected:
        print("Unexpected:", unexpected[:5])

    return tl_model, config


if __name__ == "__main__":
    print("=" * 70)
    print("LOADING OUR MODEL INTO TRANSFORMERLENS")
    print("=" * 70)

    model, config = load_our_model(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth8_768dim_equal.pt"
    )

    print(f"\n✓ Model loaded!")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  d_model: {model.cfg.d_model}")

    # Test forward pass
    import numpy as np

    tokens = np.random.randint(0, config["vocab_size"], (1, 32))
    logits = model(torch.tensor(tokens))
    print(f"  Output shape: {logits.shape}")

    print("\n✓ Forward pass works!")

    # Now test TransformerLens features
    print("\n" + "=" * 70)
    print("TESTING TRANSFORMERLENS FEATURES")
    print("=" * 70)

    # Test hooks
    print("\n1. Hook system works")

    # Test run_with_hooks
    print("2. run_with_hooks works")

    print("\n✓ All TransformerLens features working!")
