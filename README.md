# autointerpret — Learning Project

*A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for exploring neural network interpretability.*

## What This Is

This is a **learning project** exploring whether neural architecture search can find more interpretable models. It's a proof-of-concept, not finished research.

**What works:**
- Automated experiments with custom proxy metrics
- Basic circuit analysis on our custom model
- Causal ablation (found layer 7 matters for token copying)

**What doesn't work:**
- Integration with standard MI tools (TransformerLens, SAELens)
- Comparison to established baselines (GPT-2, Pythia)
- Full circuit analysis

## What We Found

1. **Depth correlates with proxy metrics** - Deeper models score higher on our interpretability metrics
2. **Token copying circuit exists** - Found with custom analysis tools
3. **Layer 7 is necessary** - Ablation decreases copy behavior by 0.04

These are preliminary findings, not validated results.

## Why It Matters (For Learning)

If you're interested in:
- Building custom ML analysis tools
- Understanding how to analyze neural networks
- Exploring interpretability research

...this shows one approach. We built things from scratch when standard tools didn't work.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python prepare.py

# Run experiment
python train.py

# Analyze results
python interpret.py
```

## Project Structure

```
autointerpret/
├── mi_tools_backup/           # Working custom analysis tools
├── gpt2_baseline_analysis.py  # TransformerLens demo
├── notebooks/                 # Scale validation (Kaggle)
├── figures/                  # Visualizations
└── train.py, interpret.py   # Core code
```

## Honest Limitations

- Our proxy metrics may not measure true interpretability
- No comparison to established models
- Custom architecture incompatible with standard MI tools
- Small scale experiments
- Findings not validated externally

## For More Serious Work

If you want to do real interpretability research:
- Use TransformerLens or SAELens
- Start with GPT-2 or Pythia
- Compare to established circuits
- Get feedback from MI community

## License

MIT
