#!/usr/bin/env python3
"""Generate publication-quality figures from autointerpret experiment results.

Usage:
    python plot_results.py

Reads results.tsv and produces four figures in figures/ directory:
    1. pareto_frontier.png
    2. depth_vs_convergence.png
    3. submetric_heatmap.png
    4. search_trajectory.png
"""

import os
import re
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(SCRIPT_DIR, "results.tsv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
FIGSIZE = (10, 7)

# Professional color palette
C_KEEP = "#2171b5"  # strong blue for kept experiments
C_DISCARD = "#bdbdbd"  # light gray for discarded
C_FRONTIER = "#d94801"  # burnt orange for Pareto frontier line
C_BEST = "#238b45"  # green accent for best-tradeoff label
C_TRAJECTORY_BEST = "#d94801"

# Try a clean style; fall back gracefully
for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
    if style in plt.style.available:
        plt.style.use(style)
        break

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 100,
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            row["index"] = i + 1  # 1-based experiment number
            for col in [
                "val_bpb",
                "interpret_score",
                "hoyer",
                "convergence",
                "erank",
                "attn_conc",
            ]:
                row[col] = float(row[col])
            rows.append(row)
    return rows


rows = load_results(TSV_PATH)

# Split into valid (interpret ran) and failed
valid = [r for r in rows if r["interpret_score"] > 0.0]
kept = [r for r in valid if r["status"] == "keep"]
discarded = [r for r in valid if r["status"] != "keep"]

# ---------------------------------------------------------------------------
# 1. Pareto Frontier
# ---------------------------------------------------------------------------


def compute_pareto_frontier(points):
    """Given list of (bpb, interpret) tuples, return indices on the Pareto frontier.

    Pareto-optimal: no other point has both lower bpb AND higher interpret.
    Sort by bpb ascending, then sweep for increasing interpret.
    """
    indexed = sorted(enumerate(points), key=lambda t: (t[1][0], -t[1][1]))
    frontier_idx = []
    best_interpret = -np.inf
    for orig_i, (bpb, interp) in indexed:
        if interp > best_interpret:
            frontier_idx.append(orig_i)
            best_interpret = interp
    return frontier_idx


def plot_pareto():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Discarded as gray
    ax.scatter(
        [r["val_bpb"] for r in discarded],
        [r["interpret_score"] for r in discarded],
        c=C_DISCARD,
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        label="Discarded",
        zorder=2,
    )

    # Kept as blue
    ax.scatter(
        [r["val_bpb"] for r in kept],
        [r["interpret_score"] for r in kept],
        c=C_KEEP,
        s=70,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
        label="Kept (Pareto)",
        zorder=3,
    )

    # Compute and draw Pareto frontier through kept points
    kept_points = [(r["val_bpb"], r["interpret_score"]) for r in kept]
    frontier_idx = compute_pareto_frontier(kept_points)
    frontier_pts = sorted([kept_points[i] for i in frontier_idx], key=lambda p: p[0])
    fx, fy = zip(*frontier_pts)
    ax.plot(
        fx,
        fy,
        color=C_FRONTIER,
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
        label="Pareto frontier",
        zorder=4,
    )

    # Label key points
    annotations = {}

    # Baseline
    baseline = [r for r in kept if r["commit"] == "baseline"][0]
    annotations["Baseline"] = (
        baseline["val_bpb"],
        baseline["interpret_score"],
        (-40, -18),
    )

    # Best quality (lowest val_bpb among kept)
    best_qual = min(kept, key=lambda r: r["val_bpb"])
    annotations["Best quality"] = (
        best_qual["val_bpb"],
        best_qual["interpret_score"],
        (-30, 12),
    )

    # Best interpret
    best_interp = max(kept, key=lambda r: r["interpret_score"])
    annotations["Best interpret"] = (
        best_interp["val_bpb"],
        best_interp["interpret_score"],
        (10, 10),
    )

    # Best tradeoff (highest interpret_score among kept with val_bpb < 1.42)
    tradeoff_candidates = [r for r in kept if r["val_bpb"] < 1.42]
    if tradeoff_candidates:
        best_trade = max(tradeoff_candidates, key=lambda r: r["interpret_score"])
        if best_trade["commit"] != best_interp["commit"]:
            annotations["Best tradeoff"] = (
                best_trade["val_bpb"],
                best_trade["interpret_score"],
                (10, -15),
            )

    for label, (x, y, offset) in annotations.items():
        color = C_BEST if "tradeoff" in label.lower() else "#333333"
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            zorder=5,
        )

    ax.set_xlabel("Validation BPB (lower = better quality)")
    ax.set_ylabel("Composite Interpretability Score (higher = better)")
    ax.set_title("Quality vs Interpretability Pareto Frontier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "pareto_frontier.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved pareto_frontier.png")


# ---------------------------------------------------------------------------
# 2. Depth vs Convergence
# ---------------------------------------------------------------------------


def extract_depth(row):
    """Extract depth from description. Baseline is DEPTH=4."""
    m = re.search(r"DEPTH=(\d+)", row["description"])
    if m:
        return int(m.group(1))
    if row["commit"] == "baseline":
        return 4
    return None


def plot_depth_convergence():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Collect all experiments where depth is identifiable
    depth_data = []
    for r in valid:
        d = extract_depth(r)
        if d is not None:
            depth_data.append((d, r["convergence"], r["status"]))

    # Also include baseline explicitly
    depths = sorted(set(d for d, _, _ in depth_data))

    # Plot individual points
    for d, conv, status in depth_data:
        color = C_KEEP if status == "keep" else C_DISCARD
        edge = "#333333" if status == "keep" else "#999999"
        ax.scatter(d, conv, c=color, s=80, edgecolors=edge, linewidths=0.8, zorder=3)

    # Compute means per depth and connect
    from collections import defaultdict

    depth_groups = defaultdict(list)
    for d, conv, _ in depth_data:
        depth_groups[d].append(conv)

    mean_depths = sorted(depth_groups.keys())
    mean_convs = [np.mean(depth_groups[d]) for d in mean_depths]

    ax.plot(
        mean_depths,
        mean_convs,
        color=C_FRONTIER,
        linewidth=2.5,
        marker="D",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor=C_FRONTIER,
        markeredgewidth=2,
        zorder=4,
        label="Mean convergence",
    )

    # Annotate the trend
    ax.annotate(
        f"{mean_convs[0]:.3f}",
        (mean_depths[0], mean_convs[0]),
        textcoords="offset points",
        xytext=(-35, -15),
        fontsize=10,
        color="#555555",
    )
    ax.annotate(
        f"{mean_convs[-1]:.3f}",
        (mean_depths[-1], mean_convs[-1]),
        textcoords="offset points",
        xytext=(8, -15),
        fontsize=10,
        color="#555555",
    )

    ax.set_xlabel("Number of Layers (Depth)")
    ax.set_ylabel("Logit Lens Convergence Score")
    ax.set_title("Logit Lens Convergence Scales with Depth")
    ax.set_xticks(depths)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "depth_vs_convergence.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved depth_vs_convergence.png")


# ---------------------------------------------------------------------------
# 3. Sub-metric Heatmap
# ---------------------------------------------------------------------------


def plot_heatmap():
    metrics = ["hoyer", "convergence", "erank", "attn_conc"]
    metric_labels = [
        "Hoyer\nSparsity",
        "Logit Lens\nConvergence",
        "Effective\nRank",
        "Attention\nConcentration",
    ]

    # Sort kept experiments by interpret_score descending
    sorted_kept = sorted(kept, key=lambda r: r["interpret_score"], reverse=True)

    data = np.array([[r[m] for m in metrics] for r in sorted_kept])

    # Shorten descriptions for row labels
    def short_desc(r):
        desc = r["description"]
        # Take the part before the first parenthesis, or first 35 chars
        paren = desc.find("(")
        if paren > 0:
            desc = desc[:paren].strip()
        if len(desc) > 38:
            desc = desc[:35] + "..."
        return desc

    row_labels = [short_desc(r) for r in sorted_kept]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data, cmap="YlOrBr", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Add value annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Choose text color based on background brightness
            text_color = "white" if val > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="medium",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Score (higher = more interpretable)", fontsize=11)

    ax.set_title("Sub-metric Decomposition Across Pareto Experiments")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "submetric_heatmap.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved submetric_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Search Trajectory
# ---------------------------------------------------------------------------


def plot_trajectory():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    indices = [r["index"] for r in rows]
    scores = [r["interpret_score"] for r in rows]
    statuses = [r["status"] for r in rows]

    # Plot all points
    for idx, score, status in zip(indices, scores, statuses):
        if score == 0.0:
            # Failed runs: small X marker
            ax.scatter(idx, score, marker="x", c="#e0e0e0", s=25, zorder=2)
        elif status == "keep":
            ax.scatter(
                idx, score, c=C_KEEP, s=55, edgecolors="white", linewidths=0.6, zorder=3
            )
        else:
            ax.scatter(
                idx,
                score,
                c=C_DISCARD,
                s=35,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )

    # Running best line (ignoring zeros)
    running_best = []
    current_best = 0.0
    for score in scores:
        if score > 0:
            current_best = max(current_best, score)
        running_best.append(current_best)

    ax.plot(
        indices,
        running_best,
        color=C_TRAJECTORY_BEST,
        linewidth=2.0,
        label="Running best",
        zorder=4,
    )

    # Add legend handles manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_KEEP,
            markersize=8,
            label="Kept (Pareto)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_DISCARD,
            markersize=8,
            label="Discarded",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#cccccc",
            markersize=8,
            linestyle="None",
            label="Failed interpret run",
        ),
        Line2D([0], [0], color=C_TRAJECTORY_BEST, linewidth=2, label="Running best"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Composite Interpretability Score")
    ax.set_title("Architecture Search Trajectory")
    ax.set_xlim(0, len(rows) + 1)
    ax.set_ylim(-0.02, max(scores) + 0.03)
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "search_trajectory.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved search_trajectory.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Scale Validation (768-dim, equal-quality protocol)
# ---------------------------------------------------------------------------

SCALE_VALIDATION_DATA = {
    "4L": {
        "val_bpb": 1.643,
        "hoyer": 0.610,
        "conv": 0.616,
        "erank": 0.543,
        "attn": 0.377,
        "v2": 0.536,
        "time": 622,
    },
    "8L": {
        "val_bpb": 1.683,
        "hoyer": 0.672,
        "conv": 0.790,
        "erank": 0.590,
        "attn": 0.429,
        "v2": 0.620,
        "time": 948,
    },
    "12L": {
        "val_bpb": 1.672,
        "hoyer": 0.680,
        "conv": 0.846,
        "erank": 0.545,
        "attn": 0.495,
        "v2": 0.641,
        "time": 1687,
    },
}


def plot_scale_validation():
    """Plot scale validation results from Kaggle (768-dim, equal-quality protocol)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    depths = [4, 8, 12]
    labels = ["4L", "8L", "12L"]
    colors = ["#2171b5", "#6baed6", "#9ecae1"]

    # Left: val_bpb vs interpretability
    ax = axes[0]
    val_bpbs = [SCALE_VALIDATION_DATA[d]["val_bpb"] for d in labels]
    v2_scores = [SCALE_VALIDATION_DATA[d]["v2"] for d in labels]

    bars = ax.bar(labels, v2_scores, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Composite Interpretability Score (v2)", fontsize=12)
    ax.set_xlabel("Depth", fontsize=12)
    ax.set_title(
        "Scale Validation: Interpretability vs Depth\n(768-dim, Equal-Quality Protocol)",
        fontsize=13,
    )

    for bar, v2, vbp in zip(bars, v2_scores, val_bpbs):
        ax.annotate(
            f"{v2:.3f}\n(bpb={vbp:.3f})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333",
        )

    ax.set_ylim(0, 0.8)
    ax.axhline(y=v2_scores[0], color="#2171b5", linestyle="--", alpha=0.5, linewidth=1)

    # Right: sub-metrics breakdown
    ax = axes[1]
    x = np.arange(len(labels))
    width = 0.2

    metrics = ["hoyer", "conv", "erank", "attn"]
    metric_labels = ["Hoyer", "Convergence", "E-Rank", "Attn"]
    metric_colors = ["#d94801", "#2171b5", "#238b45", "#6c51a4"]

    for i, (m, ml, mc) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [SCALE_VALIDATION_DATA[d][m] for d in labels]
        ax.bar(x + i * width, vals, width, label=ml, color=mc, edgecolor="white")

    ax.set_ylabel("Sub-metric Score", fontsize=12)
    ax.set_xlabel("Depth", fontsize=12)
    ax.set_title("Sub-metric Breakdown by Depth\n(768-dim, Equal-Quality)", fontsize=13)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "scale_validation.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved scale_validation.png")


# ---------------------------------------------------------------------------
# 6. Causal Validation (Linear Probes + SAE)
# ---------------------------------------------------------------------------

CAUSAL_VALIDATION_DATA = {
    "4L": {
        "ntp_selectivity": 0.091,
        "bigram_selectivity": 0.001,
        "combined_selectivity": 0.046,
        "sae_mse": 1114.4,
        "sae_l0": 1718.1,
        "sae_r2": 0.989,
    },
    "8L": {
        "ntp_selectivity": 0.093,
        "bigram_selectivity": 0.063,
        "combined_selectivity": 0.078,
        "sae_mse": 3730.8,
        "sae_l0": 1819.2,
        "sae_r2": 0.993,
    },
}


def plot_causal_validation():
    """Plot causal validation results (linear probes + SAE comparison)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ["4L", "8L"]
    colors = ["#2171b5", "#d94801"]

    # Left: Linear probe selectivity
    ax = axes[0]
    x = np.arange(3)
    width = 0.35

    ntp = [CAUSAL_VALIDATION_DATA[d]["ntp_selectivity"] for d in labels]
    bigram = [CAUSAL_VALIDATION_DATA[d]["bigram_selectivity"] for d in labels]
    combined = [CAUSAL_VALIDATION_DATA[d]["combined_selectivity"] for d in labels]

    bars1 = ax.bar(
        x - width / 2,
        [ntp[0], bigram[0], combined[0]],
        width,
        label="4L",
        color=colors[0],
    )
    bars2 = ax.bar(
        x + width / 2,
        [ntp[1], bigram[1], combined[1]],
        width,
        label="8L",
        color=colors[1],
    )

    ax.set_ylabel("Selectivity (higher = more interpretable)", fontsize=11)
    ax.set_title("Linear Probe Selectivity\n(Real Accuracy - Control)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["NTP Probe", "Bigram Probe", "Combined"])
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Middle: SAE reconstruction quality
    ax = axes[1]
    mse = [CAUSAL_VALIDATION_DATA[d]["sae_mse"] for d in labels]
    bars = ax.bar(labels, mse, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Reconstruction MSE (lower = easier to decompose)", fontsize=11)
    ax.set_title("SAE Reconstruction Quality\n(Avg across layers)", fontsize=12)

    for bar, m in zip(bars, mse):
        ax.annotate(
            f"{m:.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Right: SAE R² variance explained
    ax = axes[2]
    r2 = [CAUSAL_VALIDATION_DATA[d]["sae_r2"] for d in labels]
    bars = ax.bar(labels, r2, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("R² Explained Variance (higher = better)", fontsize=11)
    ax.set_title("SAE Variance Explained\n(Avg across layers)", fontsize=12)
    ax.set_ylim(0.98, 1.0)

    for bar, r in zip(bars, r2):
        ax.annotate(
            f"{r:.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "causal_validation.png"), dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved causal_validation.png")


if __name__ == "__main__":
    print(f"Loaded {len(rows)} experiments ({len(valid)} valid, {len(kept)} kept)")
    print("Generating figures...")
    plot_pareto()
    plot_depth_convergence()
    plot_heatmap()
    plot_trajectory()
    plot_scale_validation()
    plot_causal_validation()
    print(f"All figures saved to {FIG_DIR}/")
