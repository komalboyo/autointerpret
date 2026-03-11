"""
GPT-2 Baseline Analysis with TransformerLens

This analyzes GPT-2 to establish baseline circuits for comparison.
Key behaviors:
1. Token copying (previous token prediction)
2. Induction heads (pattern completion)
3. Bigram prediction
"""

from transformer_lens import HookedTransformer, utils
import torch
import numpy as np


def find_circuit_copies(model, num_samples=50, vocab_size=None):
    """Find token copying behavior in GPT-2."""
    print("\n" + "=" * 60)
    print("CIRCUIT 1: TOKEN COPYING")
    print("=" * 60)

    # Get vocab size
    if vocab_size is None:
        if hasattr(model, "cfg"):
            vocab_size = model.cfg.d_vocab
        elif hasattr(model, "config"):
            vocab_size = model.config.get("vocab_size", 8192)
        else:
            vocab_size = 8192  # default

    copy_scores = []

    for _ in range(num_samples):
        # Create repeated token sequence
        tokens = torch.randint(0, vocab_size, (1, 32))

        # Get logits
        logits = model(tokens)
        probs = torch.softmax(logits[0, :-1], dim=-1)

        # Check copy probability
        prev_tokens = tokens[0, 1:]
        for pos in range(len(prev_tokens)):
            copy_prob = probs[pos, prev_tokens[pos]].item()
            copy_scores.append(copy_prob)

    # Find high-copy positions
    high_copy = [s for s in copy_scores if s > 0.3]

    print(
        f"Found {len(high_copy)} high-copy positions ({len(high_copy) / len(copy_scores) * 100:.1f}%)"
    )
    print(f"Average copy prob: {np.mean(copy_scores):.3f}")

    return copy_scores


def find_circuit_induction(model):
    """Find induction heads (A -> B -> A pattern)."""
    print("\n" + "=" * 60)
    print("CIRCUIT 2: INDUCTION HEADS")
    print("=" * 60)

    # Create pattern: A B A B A B
    vocab_size = model.cfg.d_vocab

    # Use a simple pattern
    tokens = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]])

    # Run with cache to get attention patterns
    logits, cache = model.run_with_cache(tokens)

    # Look for induction pattern in attention
    print("Looking for induction heads...")

    induction_scores = []

    for layer in range(model.cfg.n_layers):
        # Get attention pattern
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [heads, seq, seq]

        # Induction: position i attends to position i-2 when tokens repeat
        # Look at positions where token at t should predict token at t+1 being same as t-1
        for head in range(model.cfg.n_heads):
            pattern = attn[head]

            # Simple metric: does this head attend to previous occurrence?
            score = pattern[4, 2].item()  # position 4 attends to position 2
            induction_scores.append((layer, head, score))

    # Top induction heads
    induction_scores.sort(key=lambda x: x[2], reverse=True)

    print(f"Top induction head candidates:")
    for layer, head, score in induction_scores[:5]:
        print(f"  Layer {layer}, Head {head}: score={score:.3f}")

    return induction_scores[:5]


def find_circuit_bigram(model):
    """Find bigram prediction circuits."""
    print("\n" + "=" * 60)
    print("CIRCUIT 3: BIGRAM PREDICTION")
    print("=" * 60)

    # Simple bigram: predict next token based on current
    # Use common token pairs

    # Get some logits
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 100))
    logits = model(tokens)

    # How often is top prediction correct?
    predictions = logits.argmax(dim=-1)
    actual = tokens[0, 1:]

    # Move to same device
    predictions = predictions.to("cpu")
    actual = actual.to("cpu")

    accuracy = (predictions[0, :-1] == actual).float().mean().item()

    print(f"Sequence prediction accuracy: {accuracy:.3f}")

    return accuracy


def analyze_gpt2():
    """Comprehensive GPT-2 analysis."""

    print("=" * 60)
    print("GPT-2 BASELINE ANALYSIS WITH TRANSFORMERLENS")
    print("=" * 60)

    # Load GPT-2 small
    print("\n[1] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2")
    print(f"  Model: {model.cfg.n_layers} layers, {model.cfg.d_model} dim")
    print(f"  Vocab: {model.cfg.d_vocab}")

    # Analyze each circuit
    copy_scores = find_circuit_copies(model)
    induction_heads = find_circuit_induction(model)
    bigram_acc = find_circuit_bigram(model)

    # Summary
    print("\n" + "=" * 60)
    print("GPT-2 SUMMARY")
    print("=" * 60)
    print(f"""
Token Copying: {np.mean(copy_scores):.3f} avg probability
Induction Heads: Found {len(induction_heads)} candidates
Bigram Prediction: {bigram_acc:.3f} accuracy

These are established circuits in GPT-2 that we can compare against.
""")

    return {
        "copy_scores": copy_scores,
        "induction_heads": induction_heads,
        "bigram_acc": bigram_acc,
    }


if __name__ == "__main__":
    results = analyze_gpt2()
