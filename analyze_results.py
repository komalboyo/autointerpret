"""Analyze autointerpret experiment results."""
import csv

# Load results
rows = []
with open("results.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['interpret_score'] != '0.0000' and row['interpret_score'] != '0':
            row['val_bpb'] = float(row['val_bpb'])
            row['interpret_score'] = float(row['interpret_score'])
            row['hoyer'] = float(row['hoyer'])
            row['convergence'] = float(row['convergence'])
            row['erank'] = float(row['erank'])
            row['attn_conc'] = float(row['attn_conc'])
            rows.append(row)

print(f"Total experiments: 56")
print(f"With interpret scores: {len(rows)}")
print(f"Kept: {sum(1 for r in rows if r['status'] == 'keep')}")
print()

# Pareto frontier: not dominated on (val_bpb lower, interpret_score higher)
pareto = []
for r in rows:
    dominated = False
    for r2 in rows:
        if r2['val_bpb'] <= r['val_bpb'] and r2['interpret_score'] >= r['interpret_score']:
            if r2['val_bpb'] < r['val_bpb'] or r2['interpret_score'] > r['interpret_score']:
                dominated = True
                break
    if not dominated:
        pareto.append(r)

pareto.sort(key=lambda r: r['val_bpb'])
print("=" * 80)
print("PARETO FRONTIER (not dominated on val_bpb vs interpret_score)")
print("=" * 80)
print(f"{'description':<55} {'val_bpb':>8} {'interp':>8} {'hoyer':>7} {'conv':>7} {'erank':>7} {'attn':>7}")
print("-" * 80)
for r in pareto:
    desc = r['description'][:54]
    print(f"{desc:<55} {r['val_bpb']:>8.4f} {r['interpret_score']:>8.4f} {r['hoyer']:>7.4f} {r['convergence']:>7.4f} {r['erank']:>7.4f} {r['attn_conc']:>7.4f}")

print()
print("=" * 80)
print("ANALYSIS BY CATEGORY")
print("=" * 80)

# Categorize experiments
categories = {
    "Activation function": ["GELU"],
    "Depth": ["DEPTH=", "depth", "D6", "D8", "D10", "D12"],
    "Width/MLP": ["MLP", "AR=", "ASPECT", "wider", "Gated"],
    "Attention": ["HEAD_DIM", "heads", "window", "Window", "All-S"],
    "Residual/Skip": ["x0_lambda", "resid_lambda", "VE", "No VE", "No QK"],
    "Learning rate": ["LR", "_LR", "MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR"],
    "Schedule": ["WARMUP", "WARMDOWN", "FINAL_LR", "betas"],
    "Other": ["BATCH", "WEIGHT_DECAY", "ns_steps", "SPARSITY", "Parallel", "Logit"],
}

for cat_name, keywords in categories.items():
    cat_rows = [r for r in rows if any(kw.lower() in r['description'].lower() for kw in keywords)]
    if not cat_rows:
        continue
    print(f"\n--- {cat_name} ({len(cat_rows)} experiments) ---")

    # Best and worst in category
    best = max(cat_rows, key=lambda r: r['interpret_score'])
    print(f"  Best interpret: {best['description'][:60]}")
    print(f"    val_bpb={best['val_bpb']:.4f}  interpret={best['interpret_score']:.4f}")

    # Summary of what worked
    kept = [r for r in cat_rows if r['status'] == 'keep']
    if kept:
        print(f"  Kept experiments:")
        for r in kept:
            delta_bpb = r['val_bpb'] - rows[0]['val_bpb']
            delta_int = r['interpret_score'] - rows[0]['interpret_score']
            print(f"    {r['description'][:55]}  bpb {delta_bpb:+.4f}  int {delta_int:+.4f}")

print()
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)

baseline = rows[0]
best_interp = max(rows, key=lambda r: r['interpret_score'])
best_quality = min(rows, key=lambda r: r['val_bpb'])

print(f"\nBaseline:        val_bpb={baseline['val_bpb']:.4f}  interpret={baseline['interpret_score']:.4f}")
print(f"Best quality:    val_bpb={best_quality['val_bpb']:.4f}  interpret={best_quality['interpret_score']:.4f}  ({best_quality['description'][:50]})")
print(f"Best interpret:  val_bpb={best_interp['val_bpb']:.4f}  interpret={best_interp['interpret_score']:.4f}  ({best_interp['description'][:50]})")

print(f"\nInterpretability improvement: {(best_interp['interpret_score'] - baseline['interpret_score']) / baseline['interpret_score'] * 100:.1f}%")
print(f"Quality cost: {(best_interp['val_bpb'] - baseline['val_bpb']) / baseline['val_bpb'] * 100:.1f}%")

# Metric correlations
print()
print("--- Sub-metric ranges across all experiments ---")
for metric in ['hoyer', 'convergence', 'erank', 'attn_conc']:
    vals = [r[metric] for r in rows]
    print(f"  {metric:>12}: min={min(vals):.4f}  max={max(vals):.4f}  range={max(vals)-min(vals):.4f}")
