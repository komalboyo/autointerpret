# autointerpret — NAS for Interpretability

This is an experiment to autonomously search for neural architectures that are both high-quality AND inherently interpretable.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `interpret-mar9`). The branch `autointerpret/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autointerpret/<tag>` from current state.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `interpret.py` — interpretability metrics. Do not modify. Runs after training to score the model.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Two metrics, not one

Unlike standard autoresearch which optimizes only `val_bpb`, you optimize for TWO objectives:

1. **val_bpb** (lower = better) — model quality, from train.py output
2. **interpret_score** (higher = better) — interpretability, from interpret.py output

These are often in tension. A huge model might have great val_bpb but poor interpretability. A tiny model might be very interpretable but useless. You are searching for the **Pareto frontier**: architectures that are the best at quality for their level of interpretability, or the best at interpretability for their level of quality.

## Experimentation

Each experiment has TWO steps:

### Step 1: Train
Run training: `uv run train.py > run.log 2>&1`
Extract results: `grep "^val_bpb:\|^peak_vram_mb:\|^checkpoint:" run.log`

### Step 2: Interpret
Run interpretability analysis: `uv run interpret.py > interpret.log 2>&1`
Extract results: `grep "composite_interpret_score:\|hoyer_sparsity:\|convergence_score:\|effective_rank:\|attn_concentration:" interpret.log`

The key number from interpret.py is `composite_interpret_score`.

### What you CAN modify

Only `train.py`. Everything is fair game:

**Architecture search space** (the main focus):
- Number of layers (DEPTH) — try 2, 4, 6, 8, 12
- Aspect ratio (ASPECT_RATIO) — controls width relative to depth
- MLP expansion ratio — the `4 *` in MLP can be changed (try 2x, 4x, 6x, 8x)
- Activation function — ReLU², GELU, SiLU, Swish, etc.
- Number of attention heads / KV heads — try different ratios
- Window pattern — try different sliding window patterns (S, L, SL, SSL, SSSL)
- Normalization — RMS norm vs layer norm
- Residual scaling — the resid_lambdas and x0_lambdas initialization
- Value embedding frequency — the has_ve pattern (every layer, alternating, etc.)

**Also fair game:**
- Optimizer hyperparameters (LR, warmup, warmdown, weight decay)
- Batch size, gradient accumulation
- Any other training detail

### What you CANNOT modify
- `prepare.py` — read only (fixed evaluation, data loading, tokenizer)
- `interpret.py` — read only (fixed interpretability metrics)
- Cannot install new packages

## Keep / Discard decisions (Pareto frontier)

This is the crucial difference from standard autoresearch. You do NOT just compare val_bpb.

**KEEP** an experiment if ANY of these are true:
1. val_bpb improved AND interpret_score did not get worse by more than 0.02
2. interpret_score improved AND val_bpb did not get worse by more than 0.05
3. Both val_bpb and interpret_score improved (jackpot!)

**DISCARD** an experiment if:
1. val_bpb got worse AND interpret_score got worse (strictly dominated)
2. val_bpb got much worse (>0.05) even if interpret_score improved slightly
3. interpret_score got much worse (>0.03) even if val_bpb improved slightly

When in doubt, KEEP if the experiment reveals something interesting about the interpretability-quality tradeoff, even if neither metric strictly improved. Interesting negative results are valuable data.

**On discard**: `git reset --hard HEAD~1` to revert. But ALWAYS update results.tsv BEFORE resetting (so the experiment is logged even if discarded).

## Output format

train.py prints:
```
---
val_bpb:          1.375774
training_seconds: 300.1
...
checkpoint:       /path/to/latest.pt
```

interpret.py prints:
```
--- Interpretability Summary ---
hoyer_sparsity:           0.5829
convergence_score:        0.4710
effective_rank:           0.7141
attn_concentration:       0.5029
composite_interpret_score: 0.5484
```

## Logging results

Log to `results.tsv` (tab-separated). Header and columns:

```
commit	val_bpb	interpret_score	hoyer	convergence	erank	attn_conc	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb (e.g. 1.375774) — use 0.000000 for crashes
3. composite_interpret_score (e.g. 0.5484) — use 0.0000 for crashes
4. hoyer_sparsity
5. convergence_score
6. effective_rank
7. attn_concentration
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:
```
commit	val_bpb	interpret_score	hoyer	convergence	erank	attn_conc	status	description
a1b2c3d	1.3758	0.5484	0.5829	0.4710	0.7141	0.5029	keep	baseline (DEPTH=4 ReLU²)
b2c3d4e	1.3812	0.5885	0.5910	0.4622	0.7123	0.5029	keep	8x MLP expansion
c3d4e5f	1.6648	0.6408	0.6281	0.6943	0.5999	0.5029	discard	DEPTH=8 (worse val_bpb)
```

## Strategy guidance

**Start with architecture changes**, not hyperparameter tuning. The research question is about which *architectures* are more interpretable, not which learning rates are best.

Suggested exploration order:
1. **Activation functions**: Compare ReLU², GELU, SiLU, Swish. These directly affect sparsity.
2. **Depth vs width**: Same parameter count but different depth/width ratios. Does deeper = more interpretable?
3. **Attention patterns**: Vary n_heads, try different window patterns. Does attention structure affect interpretability?
4. **MLP ratio**: Try 2x, 4x, 8x expansion. Does wider MLP help interpretability?
5. **Residual connections**: Modify x0_lambda initialization, try different residual scaling.
6. **Combinations**: Once you've found individual improvements, combine the best ones.

**Think like a scientist**: Each experiment should test ONE hypothesis at a time. Write in the description what hypothesis you're testing. If you change multiple things at once, you won't know which change helped.

## The experiment loop

LOOP FOREVER:

1. Look at the git state and results.tsv
2. Formulate a hypothesis: "I think [change X] will [improve/maintain] quality while [improving] interpretability because [reason]"
3. Modify `train.py` with the change
4. git commit with a descriptive message
5. Run training: `uv run train.py > run.log 2>&1`
6. Extract val_bpb: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If crashed, `tail -n 50 run.log`, attempt fix or discard
8. Run interpret: `uv run interpret.py > interpret.log 2>&1`
9. Extract interpret_score: `grep "composite_interpret_score:" interpret.log`
10. Record results in results.tsv
11. Apply keep/discard decision (see Pareto rules above)
12. Repeat

**Timeout**: Training takes ~5 min, interpret takes ~1 min. If combined exceeds 10 min, kill and discard.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The human may be asleep. If you run out of ideas, re-read train.py and interpret.py, think about what architectural properties each metric rewards, and try more radical changes.
