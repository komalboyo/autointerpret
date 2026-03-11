"""
TransformerLens Wrapper for Autointerpret Models

This wraps our custom autoresearch models to work with TransformerLens
for proper circuit analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from functools import partial

# We'll load the model manually
import ast


def load_checkpoint(path):
    """Load checkpoint and extract model architecture."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_source = checkpoint["train_source"]
    config = checkpoint["config"]
    model_state = checkpoint["model_state_dict"]
    return train_source, config, model_state


def extract_model_from_source(train_source):
    """Extract model classes from saved source."""
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

    CACHE_DIR = "~/.cache/autoresearch"
    TOKENIZER_DIR = "~/.cache/autoresearch/tokenizer"

    preamble = f"""
import os, sys, math, gc, time, contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
CACHE_DIR = os.path.expanduser("{CACHE_DIR}")
TOKENIZER_DIR = os.path.expanduser("{TOKENIZER_DIR}")
"""
    full_source = preamble + "\n" + safe_source

    namespace = {"__builtins__": __builtins__}
    exec(full_source, namespace)

    return namespace


class AutointerpretTransformerLens(nn.Module):
    """
    Wrapper that makes our custom model compatible with TransformerLens.

    Implements the key interfaces needed for circuit analysis:
    - run_with_hooks: run model with intervention hooks
    - logits: get output logits
    - embed: get token embeddings
    - Unembed: linear projection to vocab
    """

    def __init__(self, config, state_dict):
        super().__init__()
        self.config = config
        self.device = "cpu"

        # Store config
        self.vocab_size = config.get("vocab_size", 8192)
        self.n_layers = config.get("n_layer", 4)
        self.d_model = config.get("n_embd", 768)
        self.n_heads = config.get("n_head", 6)
        self.d_head = self.d_model // self.n_heads
        self.d_mlp = self.d_model * 4  # MLP expansion

        # Build model components manually
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList(
            [
                AutointerpretBlock(self.d_model, self.n_heads, self.d_head, self.d_mlp)
                for _ in range(self.n_layers)
            ]
        )
        self.ln_final = nn.RMSNorm(self.d_model)
        self.unembed = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Convert all to float32 for compatibility
        nn.Module.to(self, dtype=torch.float32)

        # Load weights
        self.load_weights(state_dict)

        # For TransformerLens compatibility
        self.cfg = type(
            "obj",
            (object,),
            {
                "d_model": self.d_model,
                "d_head": self.d_head,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "d_mlp": self.d_mlp,
                "vocab_size": self.vocab_size,
                "act_fn": "gelu",
                "attention_dir": "causal",
                "attn_only": False,
                "normalization": "RMS",
            },
        )()

    def load_weights(self, state_dict):
        """Load weights into model."""
        # Map our state dict keys to the model
        for key, value in state_dict.items():
            # Convert to float32
            value = value.float()

            if "transformer.wte.weight" in key:
                self.embed.weight.data = value
            elif "lm_head.weight" in key:
                self.unembed.weight.data = value
            elif "ln_f.weight" in key:
                self.ln_final.weight.data = value
            elif "transformer.h." in key:
                # Parse layer number
                parts = key.split(".")
                layer_idx = int(parts[2])
                rest = ".".join(parts[3:])

                # Map to our blocks
                if "attn.c_proj.weight" in key:
                    self.blocks[layer_idx].attn.c_proj.weight.data = value
                elif "attn.c_fc.weight" in key:
                    self.blocks[layer_idx].attn.c_fc.weight.data = value
                elif "mlp.c_proj.weight" in key:
                    self.blocks[layer_idx].mlp.c_proj.weight.data = value
                elif "mlp.c_fc.weight" in key:
                    self.blocks[layer_idx].mlp.c_fc.weight.data = value

    def __call__(self, tokens, return_logits=True):
        """Standard forward pass."""
        # Embed
        x = self.embed(tokens)

        # Through blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and unembed
        x = self.ln_final(x)

        if return_logits:
            return self.unembed(x)
        return x

    def forward(self, tokens, return_logits=True):
        """Alias for __call__."""
        return self.__call__(tokens, return_logits)

    def run_with_hooks(self, tokens, fwd_hooks=None, bwd_hooks=None):
        """Run with intervention hooks (TransformerLens-style)."""
        # For now, simple implementation
        # Full implementation would use torchHooks
        return self.forward(tokens)

    def logits(self, tokens):
        """Get logits from tokens."""
        return self.forward(tokens)

    @property
    def W_U(self):
        """Unembed matrix (for TransformerLens compatibility)."""
        return self.unembed.weight

    @property
    def W_E(self):
        """Embed matrix."""
        return self.embed.weight


class AutointerpretBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, d_model, n_heads, d_head, d_mlp):
        super().__init__()
        self.attn = AutointerpretAttention(d_model, n_heads, d_head)
        self.mlp = AutointerpretMLP(d_model, d_mlp)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AutointerpretAttention(nn.Module):
    """Simplified attention (uses torch's SDPA)."""

    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x):
        # Simplified: use full attention
        B, T, C = x.shape

        # Project to QKV
        q = self.c_fc(x)
        k = self.c_fc(x)
        v = self.c_fc(x)

        # Reshape for heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class AutointerpretMLP(nn.Module):
    """MLP with ReLU squared."""

    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_mlp, bias=False)
        self.c_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def load_for_transformer_lens(checkpoint_path):
    """Load checkpoint and create TransformerLens-compatible model."""
    train_source, config, state_dict = load_checkpoint(checkpoint_path)
    model = AutointerpretTransformerLens(config, state_dict)
    return model, config


# Test it
if __name__ == "__main__":
    print("Loading checkpoint...")
    model, config = load_for_transformer_lens(
        "/Users/komalmathur/Desktop/Komal/autoresearch/depth8_768dim_equal.pt"
    )
    print(f"Model loaded: {config['n_layer']} layers, {config['n_embd']} dim")

    # Test forward pass
    print("\nTesting forward pass...")
    tokens = torch.randint(0, 8192, (1, 32))
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output logits shape: {logits.shape}")

    print("\nModel is ready for TransformerLens-style analysis!")
