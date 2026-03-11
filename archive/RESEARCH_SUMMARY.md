# Autointerpret: Research Summary

## What We Built

### 1. Architecture Search (Phase 1-3)
- 56 automated experiments searching for interpretable architectures
- Found: depth is the strongest lever for interpretability
- Validated at 768-dim scale

### 2. Custom MI Tools (Our Analysis)
Located in: `mi_tools_backup/`

- **full_circuit_analysis.py** - Traces which layers contribute to behaviors
- **circuit_v2.py** - Logit lens analysis

Key Finding: **Layer 7 is necessary for token copying** (causal ablation)

### 3. TransformerLens Integration (In Progress)
Located in: `autointerpret/`

- **gpt2_baseline_analysis.py** - GPT-2 baseline analysis
- **transformer_lens_gpt2.py** - TransformerLens demo
- **convert_to_transformer_lens.py** - Weight conversion (not working)

### 4. Scale Validation
Located in: `notebooks/`

- 3 Kaggle notebooks with 768-dim experiments

---

## Key Results

| Finding | Evidence |
|---------|----------|
| Depth → Interpretability | 4L→12L: 0.54→0.64 composite score |
| Layer 7 necessary for copy | Ablation decreases copy prob by 0.04 |
| Proxy metrics correlate with probes | +68.6% probe selectivity |

---

## What Works

1. ✅ Custom circuit analysis (works on our model)
2. ✅ Causal ablation (found real result)
3. ✅ Architecture search methodology
4. ✅ Scale validation

---

## What's Incomplete

1. ❌ Our model in TransformerLens (weight conversion incomplete)
2. ❌ Full GPT-2 comparison (just demonstrated tooling)
3. ❌ Multiple circuits analyzed (only token copying)

---

## Files

```
autointerpret/
├── mi_tools_backup/           # ✅ Working custom MI tools
│   ├── full_circuit_analysis.py
│   └── circuit_v2.py
├── gpt2_baseline_analysis.py  # TransformerLens + GPT-2
├── transformer_lens_gpt2.py   # TL demo
├── convert_to_transformer_lens.py  # In progress
├── compare_models.py           # Comparison script
├── notebooks/                 # Scale validation
├── figures/                   # Publication figures
└── README.md                  # Main documentation
```

---

## How to Use

### Run Circuit Analysis on Our Model
```bash
python mi_tools_backup/full_circuit_analysis.py
```

### Run GPT-2 Analysis
```bash
python gpt2_baseline_analysis.py
```

---

## Honest Assessment

**Strengths:**
- Novel methodology
- Real causal results
- Working custom tools

**Limitations:**
- No full GPT-2 comparison
- Single circuit analyzed
- Architecture incompatibility with standard MI tools
