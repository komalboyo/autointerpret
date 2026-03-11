# autointerpret

A learning project where I explored whether architecture search can help find more interpretable neural network designs.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), this repo focuses on adding interpretability-oriented evaluation into the experiment loop.

## What I built

- Automated experiment loop for architecture exploration
- Interpretability scoring pipeline (proxy metrics)
- Scale-validation runs and plots
- Custom circuit-analysis scripts (including causal ablation experiments)
- Supporting notebooks and experiment outputs

## Repo structure

- `prepare.py`, `train.py` – training/data pipeline
- `interpret.py` – interpretability metric computation
- `plot_results.py`, `analyze_results.py` – analysis + figures
- `notebooks/` – Kaggle notebooks
- `results/` – experiment logs
- `figures/` – generated plots
- `archive/` – extra experiments, prototypes, and notes

## Why this repo exists

I wanted a hands-on way to learn:
1. How to run iterative architecture experiments
2. How to evaluate interpretability signals during search
3. How to build analysis tooling when off-the-shelf tools don’t directly fit

## Future scope

- Improve compatibility with standard MI tooling (e.g. TransformerLens workflows)
- Add stronger baseline comparisons against standard architectures
- Expand beyond initial behaviors into broader circuit-level analyses
- Turn this into a cleaner, reproducible benchmark setup

## Acknowledgment

This project is built as an exploratory fork-inspired extension of [autoresearch](https://github.com/karpathy/autoresearch).

## License

MIT
