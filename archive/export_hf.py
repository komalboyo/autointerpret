"""
TransformerLens Conversion - Fresh Attempt

Approach: Export to HuggingFace format first, then load into TransformerLens.
"""

import torch
import json


def export_to_huggingface(ckpt_path, output_path):
    """Export our checkpoint to HuggingFace format."""

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    config = ckpt["config"]

    hf_state = {}

    # Embeddings
    hf_state["transformer.wte.weight"] = state["transformer.wte.weight"]

    # Layers
    for i in range(config["n_layer"]):
        prefix = f"transformer.h.{i}."

        # Attention
        hf_state[f"h.{i}.attn.c_attn.weight"] = torch.cat(
            [
                state[f"{prefix}attn.c_q.weight"],
                state[f"{prefix}attn.c_k.weight"],
                state[f"{prefix}attn.c_v.weight"],
            ],
            dim=0,
        )

        # MLP
        hf_state[f"h.{i}.mlp.c_fc.weight"] = state[f"{prefix}mlp.c_fc.weight"]
        hf_state[f"h.{i}.mlp.c_proj.weight"] = state[f"{prefix}mlp.c_proj.weight"]

        # Layer norms
        if f"{prefix}attn.ln.weight" in state:
            hf_state[f"h.{i}.ln_1.weight"] = state[f"{prefix}attn.ln.weight"]
        if f"{prefix}mlp.ln.weight" in state:
            hf_state[f"h.{i}.ln_2.weight"] = state[f"{prefix}mlp.ln.weight"]

    # Final norm
    hf_state["ln_f.weight"] = state["ln_f.weight"]

    # Head
    hf_state["lm_head.weight"] = state["lm_head.weight"]

    # Save
    torch.save(hf_state, output_path)

    # Save config
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_layer": config["n_layer"],
        "n_embd": config["n_embd"],
        "n_head": config["n_head"],
        "n_positions": 2048,
        "vocab_size": config["vocab_size"],
        "activation_function": "gelu",
        "resid_dropout": 0.0,
        "embd_dropout": 0.0,
        "attn_dropout": 0.0,
        "layer_norm_epsilon": 1e-5,
    }

    config_path = output_path.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)

    print(f"Exported to {output_path}")
    print(f"Config saved to {config_path}")
    return config_path


if __name__ == "__main__":
    export_to_huggingface(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth8_768dim_equal.pt",
        "/Users/komalmathur/Desktop/Komal/autoresearch/autointerpret/hf_export.pt",
    )
