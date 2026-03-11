# X/Twitter Post Draft

## Tweet 1

I spent the last few months exploring whether we can use neural architecture search to find more interpretable models. Here's what I learned 🧵

## Tweet 2

The idea: instead of training a model and then trying to understand it (post-hoc), can we design architectures that are inherently more transparent?

## Tweet 3

What I built:
- 56 automated experiments searching for architectures
- 4 proxy metrics for interpretability (sparsity, convergence, effective rank, attention)
- Custom circuit analysis tools (since standard tools didn't work with my model)
- Scale validation on 768-dim models

## Tweet 4

Key finding: depth correlates with interpretability in my models.

4L → 12L = +43% on my proxy metrics (even controlling for training quality).

## Tweet 5

But honestly - I'm not sure what this means. My metrics are proxies, not true interpretability. And my custom model architecture doesn't work with standard MI tools like TransformerLens.

## Tweet 6

The real learning: building custom analysis tools when nothing fits, finding one circuit (token copying), doing causal ablation.

The limitation: can't compare to GPT-2/Pythia baselines because of architecture mismatch.

## Tweet 7

This is a learning project, not a paper. I open-sourced the code: https://github.com/[username]/autointerpret

If you're interested in ML interpretability research, this shows one approach - and where I got stuck.

## Tweet 8

What I'd do differently:
- Use standard architectures from the start
- Start with TransformerLens instead of building custom
- Compare to established circuits in GPT-2

Thanks to @karpathy for autoresearch - it was a great starting point.
