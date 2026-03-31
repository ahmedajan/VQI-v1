"""
Step 7.16: Visualize VQI-S x VQI-V Cross-Analysis Results

Generates 10 plots and analysis.md from experiment outputs.

Usage:
    python scripts/visualize_cross_analysis.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "step7", "cross_validation", "cross_analysis")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "step7", "cross_validation")

# Consistent style
S_COLOR = "#1f77b4"  # blue
V_COLOR = "#d62728"  # red
C_COLOR = "#2ca02c"  # green


def _savefig(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ====================================================================
# Plot 1: Cross Accuracy Comparison
# ====================================================================
def plot_1_accuracy_comparison():
    with open(os.path.join(DATA_DIR, "combined_training_metrics.yaml"), "r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f)

    cv = metrics["cv_results"]
    configs = [r["config"] for r in cv]
    means = [r["cv_mean"] for r in cv]
    stds = [r["cv_std"] for r in cv]
    colors = [S_COLOR, V_COLOR, C_COLOR]

    # Also get OOB from combined
    combined_oob = metrics["combined_oob_accuracy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=11)
    ax.set_title("VQI-S vs VQI-V vs Combined: Cross-Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(min(means) - 0.05, max(means) + 0.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add t-test results
    ttests = metrics.get("paired_ttests", [])
    txt_lines = []
    for tt in ttests:
        sig = "SIG" if tt["significant_bonferroni"] else "n.s."
        txt_lines.append(f"{tt['comparison']}: p={tt['p_value']:.4f} ({sig})")
    if txt_lines:
        ax.text(0.02, 0.02, "\n".join(txt_lines), transform=ax.transAxes,
                fontsize=8, verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    _savefig(fig, "cross_accuracy_comparison.png")


# ====================================================================
# Plot 2: Cross-Correlation Heatmap (Top-30 x Top-30)
# ====================================================================
def plot_2_cross_corr_heatmap_top30():
    corr = np.load(os.path.join(DATA_DIR, "cross_corr_spearman.npy"))
    with open(os.path.join(PROJECT_ROOT, "data", "step6", "full_feature", "training", "feature_names.txt"), "r", encoding="utf-8") as f:
        s_names = [l.strip() for l in f if l.strip()]
    with open(os.path.join(PROJECT_ROOT, "data", "step6", "full_feature", "training_v", "feature_names.txt"), "r", encoding="utf-8") as f:
        v_names = [l.strip() for l in f if l.strip()]

    # Top-30 by max absolute correlation
    s_max_corr = np.max(np.abs(corr), axis=1)
    v_max_corr = np.max(np.abs(corr), axis=0)
    top_s = np.argsort(s_max_corr)[::-1][:30]
    top_v = np.argsort(v_max_corr)[::-1][:30]

    sub = corr[np.ix_(top_s, top_v)]
    s_labels = [s_names[i][:25] for i in top_s]
    v_labels = [v_names[i][:25] for i in top_v]

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(v_labels)))
    ax.set_yticks(range(len(s_labels)))
    ax.set_xticklabels(v_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(s_labels, fontsize=7)
    ax.set_xlabel("VQI-V Features (top-30)", fontsize=10)
    ax.set_ylabel("VQI-S Features (top-30)", fontsize=10)
    ax.set_title("Cross-Correlation: Top-30 S x Top-30 V Features (Spearman)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")

    _savefig(fig, "cross_corr_heatmap_top30.png")


# ====================================================================
# Plot 3: Full Cross-Correlation Heatmap
# ====================================================================
def plot_3_cross_corr_heatmap_full():
    corr = np.load(os.path.join(DATA_DIR, "cross_corr_spearman.npy"))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"VQI-V Features ({corr.shape[1]})", fontsize=11)
    ax.set_ylabel(f"VQI-S Features ({corr.shape[0]})", fontsize=11)
    ax.set_title("Full Cross-Correlation Matrix: 430 S x 133 V Features", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")

    # Add summary stats
    abs_c = np.abs(corr)
    txt = (
        f"Mean |rho| = {abs_c.mean():.4f}\n"
        f"Max |rho| = {abs_c.max():.4f}\n"
        f"|rho| > 0.3: {(abs_c > 0.3).sum() / abs_c.size * 100:.1f}%\n"
        f"|rho| > 0.5: {(abs_c > 0.5).sum() / abs_c.size * 100:.1f}%\n"
        f"|rho| > 0.7: {(abs_c > 0.7).sum() / abs_c.size * 100:.1f}%"
    )
    ax.text(1.15, 0.5, txt, transform=ax.transAxes, fontsize=9, verticalalignment="center",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    _savefig(fig, "cross_corr_heatmap_full.png")


# ====================================================================
# Plot 4: Combined Model Top-30 Feature Importances
# ====================================================================
def plot_4_combined_importance_top30():
    df = pd.read_csv(os.path.join(DATA_DIR, "combined_feature_importances.csv"))
    top30 = df.head(30)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [S_COLOR if o == "S" else V_COLOR for o in top30["origin"]]
    y_pos = np.arange(len(top30))
    ax.barh(y_pos, top30["combined_importance"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Combined Model)", fontsize=10)
    ax.set_title("Top-30 Features in Combined S+V Model", fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=S_COLOR, label=f"VQI-S ({(top30['origin']=='S').sum()})"),
                       Patch(facecolor=V_COLOR, label=f"VQI-V ({(top30['origin']=='V').sum()})")]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    _savefig(fig, "combined_importance_top30.png")


# ====================================================================
# Plot 5: Importance Rank Shift Scatter
# ====================================================================
def plot_5_rank_shift():
    df = pd.read_csv(os.path.join(DATA_DIR, "combined_feature_importances.csv"))
    # Only plot features with valid solo_rank
    valid = df.dropna(subset=["solo_rank"])

    fig, ax = plt.subplots(figsize=(8, 8))
    for origin, color in [("S", S_COLOR), ("V", V_COLOR)]:
        subset = valid[valid["origin"] == origin]
        ax.scatter(subset["solo_rank"], subset["combined_rank"],
                   c=color, alpha=0.4, s=15, label=f"VQI-{origin}")

    # Identity line
    max_rank = max(valid["solo_rank"].max(), valid["combined_rank"].max())
    ax.plot([1, max_rank], [1, max_rank], "k--", alpha=0.3, label="No change")
    ax.set_xlabel("Rank in Solo Model", fontsize=11)
    ax.set_ylabel("Rank in Combined Model", fontsize=11)
    ax.set_title("Feature Rank Shift: Solo vs Combined", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _savefig(fig, "importance_rank_shift.png")


# ====================================================================
# Plot 6: Ablation Block Importance
# ====================================================================
def plot_6_ablation():
    with open(os.path.join(DATA_DIR, "ablation_results.yaml"), "r", encoding="utf-8") as f:
        abl = yaml.safe_load(f)

    bp = abl["block_permutation"]
    baseline = bp["baseline_accuracy"]
    drops = [bp["mean_drop_permute_s"], bp["mean_drop_permute_v"]]
    stds = [bp["std_drop_permute_s"], bp["std_drop_permute_v"]]
    labels = ["Permute S Block\n(keep V)", "Permute V Block\n(keep S)"]
    colors = [S_COLOR, V_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy drop bars
    ax = axes[0]
    x = [0, 1]
    bars = ax.bar(x, drops, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")
    for bar, d, s in zip(bars, drops, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.001,
                f"{d:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy Drop", fontsize=10)
    ax.set_title("Block Permutation Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Right: unique contribution pie-like stacked bar
    ax = axes[1]
    s_unique = bp["s_unique_pct"]
    v_unique = bp["v_unique_pct"]
    shared = 100 - s_unique - v_unique
    if shared < 0:
        shared = 0
    ax.barh([0], [s_unique], color=S_COLOR, alpha=0.8, label=f"S unique: {s_unique:.1f}%")
    ax.barh([0], [shared], left=[s_unique], color="gray", alpha=0.5, label=f"Shared: {shared:.1f}%")
    ax.barh([0], [v_unique], left=[s_unique + shared], color=V_COLOR, alpha=0.8, label=f"V unique: {v_unique:.1f}%")
    ax.set_xlabel("% of Baseline Accuracy", fontsize=10)
    ax.set_title("Unique vs Shared Contribution", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, max(s_unique + shared + v_unique + 5, 50))

    fig.tight_layout()
    _savefig(fig, "ablation_block_importance.png")


# ====================================================================
# Plot 7: Incremental Feature Curve
# ====================================================================
def plot_7_incremental_features():
    df = pd.read_csv(os.path.join(DATA_DIR, "incremental_features.csv"))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy curve
    ax1.plot(df["k"], df["oob_accuracy"], "ko-", linewidth=2, markersize=6, label="OOB Accuracy")
    ax1.set_xlabel("Number of Features (sorted by combined importance)", fontsize=11)
    ax1.set_ylabel("OOB Accuracy", fontsize=11)
    ax1.set_title("Incremental Feature Accuracy: S vs V Composition", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Stacked area for S/V fraction
    ax2 = ax1.twinx()
    ax2.fill_between(df["k"], 0, df["s_fraction"], alpha=0.2, color=S_COLOR, label="S fraction")
    ax2.fill_between(df["k"], df["s_fraction"], 1, alpha=0.2, color=V_COLOR, label="V fraction")
    ax2.set_ylabel("Feature Origin Fraction", fontsize=11)
    ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    _savefig(fig, "incremental_feature_curve.png")


# ====================================================================
# Plot 8: Cross-Prediction Scatter
# ====================================================================
def plot_8_cross_prediction_scatter():
    data = np.load(os.path.join(DATA_DIR, "cross_prediction_probas.npz"))
    proba_s = data["proba_s"]
    proba_v = data["proba_v"]
    y_true = data["y_true"]

    fig, ax = plt.subplots(figsize=(8, 8))
    # Class 0 first, Class 1 on top
    for cls, color, label in [(0, "red", "Class 0 (Low quality)"), (1, "blue", "Class 1 (High quality)")]:
        mask = y_true == cls
        ax.scatter(proba_s[mask], proba_v[mask], c=color, alpha=0.15, s=3, label=label, rasterized=True)

    ax.set_xlabel("P(Class1 | S features)", fontsize=11)
    ax.set_ylabel("P(Class1 | V features)", fontsize=11)
    ax.set_title("Cross-Prediction: S-Model vs V-Model Probabilities", fontsize=12, fontweight="bold")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect agreement")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, markerscale=5)
    ax.set_aspect("equal")

    # Add agreement stats
    with open(os.path.join(DATA_DIR, "cross_prediction.yaml"), "r", encoding="utf-8") as f:
        cp = yaml.safe_load(f)
    txt = (
        f"Agreement: {cp['agreement_rate']:.3f}\n"
        f"Kappa: {cp['cohen_kappa']:.3f}\n"
        f"Proba rho: {cp['spearman_proba_rho']:.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    _savefig(fig, "cross_prediction_scatter.png")


# ====================================================================
# Plot 9: Disagreement Profiles
# ====================================================================
def plot_9_disagreement_profiles():
    csv_path = os.path.join(DATA_DIR, "disagreement_profiles.csv")
    if not os.path.exists(csv_path):
        logger.warning("No disagreement_profiles.csv found, skipping plot 9")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        logger.warning("Empty disagreement_profiles.csv, skipping plot 9")
        return

    groups = df["group"].unique()
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 8))
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        gdf = df[df["group"] == group].head(20)
        colors = [S_COLOR if o == "S" else V_COLOR for o in gdf["origin"]]
        y_pos = np.arange(len(gdf))
        ax.barh(y_pos, gdf["mean_z_score"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gdf["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Z-Score (disagreement samples)", fontsize=10)
        title = group.replace("_", " ")
        n_samp = gdf["n_samples"].iloc[0] if len(gdf) > 0 else 0
        ax.set_title(f"{title}\n(n={n_samp})", fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Profiles of S/V Disagreement Regions", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "disagreement_profiles.png")


# ====================================================================
# Plot 10: Verdict Summary
# ====================================================================
def plot_10_verdict_summary():
    with open(os.path.join(DATA_DIR, "verdict.yaml"), "r", encoding="utf-8") as f:
        verdict = yaml.safe_load(f)

    ev = verdict["evidence"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Title
    verdict_str = verdict["verdict"]
    color_map = {
        "BOTH_NEEDED": "green",
        "TRULY_INDEPENDENT": "darkgreen",
        "ONE_SUFFICES": "orange",
        "MARGINAL_BENEFIT": "goldenrod",
        "BOTH_RECOMMENDED": "green",
        "EITHER_SUFFICIENT": "orange",
    }
    vcolor = color_map.get(verdict_str, "black")

    ax.text(0.5, 0.95, "VQI-S x VQI-V Cross-Analysis Verdict",
            ha="center", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.85, verdict_str,
            ha="center", va="top", fontsize=24, fontweight="bold", color=vcolor, transform=ax.transAxes)

    ax.text(0.5, 0.75, verdict["explanation"],
            ha="center", va="top", fontsize=10, wrap=True, transform=ax.transAxes,
            style="italic", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    # Evidence table
    evidence_lines = [
        f"CV Accuracy:  S={ev['cv_s']:.4f}  |  V={ev['cv_v']:.4f}  |  Combined={ev['cv_combined']:.4f}",
        f"Combined gain over best solo: {ev['combined_gain_over_best_solo']:.4f}",
        f"S unique contribution: {ev['s_unique_contribution_pct']:.1f}%",
        f"V unique contribution: {ev['v_unique_contribution_pct']:.1f}%",
        f"S->V cross-prediction OOB: {ev['s_to_v_cross_prediction_oob']:.4f}",
        f"V->S cross-prediction OOB: {ev['v_to_s_cross_prediction_oob']:.4f}",
    ]
    txt = "\n".join(evidence_lines)
    ax.text(0.5, 0.55, txt,
            ha="center", va="top", fontsize=10, fontfamily="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

    # Venn-like circles
    from matplotlib.patches import Circle
    c1 = Circle((0.3, 0.15), 0.12, color=S_COLOR, alpha=0.3, transform=ax.transAxes)
    c2 = Circle((0.55, 0.15), 0.12, color=V_COLOR, alpha=0.3, transform=ax.transAxes)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(0.22, 0.15, f"VQI-S\n430 feat", ha="center", va="center", fontsize=9, fontweight="bold",
            transform=ax.transAxes, color=S_COLOR)
    ax.text(0.63, 0.15, f"VQI-V\n133 feat", ha="center", va="center", fontsize=9, fontweight="bold",
            transform=ax.transAxes, color=V_COLOR)
    # Overlap region
    ax.text(0.425, 0.15, "?", ha="center", va="center", fontsize=14, fontweight="bold",
            transform=ax.transAxes, color="purple")

    _savefig(fig, "verdict_summary.png")


# ====================================================================
# Generate analysis.md
# ====================================================================
def generate_analysis_md():
    # Load all results
    with open(os.path.join(DATA_DIR, "combined_training_metrics.yaml"), "r", encoding="utf-8") as f:
        exp_a = yaml.safe_load(f)
    with open(os.path.join(DATA_DIR, "cross_corr_summary.yaml"), "r", encoding="utf-8") as f:
        exp_b = yaml.safe_load(f)
    with open(os.path.join(DATA_DIR, "importance_redistribution.yaml"), "r", encoding="utf-8") as f:
        exp_c = yaml.safe_load(f)
    with open(os.path.join(DATA_DIR, "ablation_results.yaml"), "r", encoding="utf-8") as f:
        exp_d = yaml.safe_load(f)
    with open(os.path.join(DATA_DIR, "cross_prediction.yaml"), "r", encoding="utf-8") as f:
        exp_e = yaml.safe_load(f)
    with open(os.path.join(DATA_DIR, "verdict.yaml"), "r", encoding="utf-8") as f:
        verdict = yaml.safe_load(f)

    ev = verdict["evidence"]
    bp = exp_d["block_permutation"]

    # Build top correlated pairs string
    top_pairs_csv = os.path.join(DATA_DIR, "top_correlated_pairs.csv")
    if os.path.exists(top_pairs_csv):
        top_pairs = pd.read_csv(top_pairs_csv).head(10)
        pairs_str = "\n".join(
            f"| {r['s_feature']} | {r['v_feature']} | {r['spearman_rho']:.4f} |"
            for _, r in top_pairs.iterrows()
        )
    else:
        pairs_str = "| (data not available) | | |"

    # Build incremental features string
    incr_csv = os.path.join(DATA_DIR, "incremental_features.csv")
    if os.path.exists(incr_csv):
        incr_df = pd.read_csv(incr_csv)
        incr_str = "\n".join(
            f"| {int(r['k'])} | {r['oob_accuracy']:.4f} | {r['n_s_features']} ({r['s_fraction']*100:.0f}%) | {r['n_v_features']} ({r['v_fraction']*100:.0f}%) |"
            for _, r in incr_df.iterrows()
        )
    else:
        incr_str = "| (data not available) | | | |"

    # T-test results string
    ttests = exp_a.get("paired_ttests", [])
    ttest_str = "\n".join(
        f"| {t['comparison']} | {t['t_statistic']:.4f} | {t['p_value']:.6f} | {'Yes' if t['significant_bonferroni'] else 'No'} |"
        for t in ttests
    )

    # Top combined features
    top10 = exp_c.get("top10_features", [])
    top_feat_str = "\n".join(
        f"| {i+1} | {r['feature']} | {r['origin']} | {r['combined_importance']:.4f} | {r.get('solo_rank', 'N/A')} | {r['combined_rank']} |"
        for i, r in enumerate(top10)
    )

    md = f"""# Step 7.16: VQI-S x VQI-V Feature-Level Cross-Analysis

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Verdict

**{verdict['verdict']}**

> {verdict['explanation']}

---

## Experiment A: Combined Model Performance

Trained RF on 563 combined features (430 S + 133 V) and compared via 5-fold stratified CV.

| Config | N Features | CV Mean | CV Std |
|--------|-----------|---------|--------|
| S-only | 430 | {ev['cv_s']:.4f} | {exp_a['cv_results'][0]['cv_std']:.4f} |
| V-only | 133 | {ev['cv_v']:.4f} | {exp_a['cv_results'][1]['cv_std']:.4f} |
| Combined | 563 | {ev['cv_combined']:.4f} | {exp_a['cv_results'][2]['cv_std']:.4f} |

**Combined gain over best solo: {ev['combined_gain_over_best_solo']:.4f}**

### Paired t-tests (Bonferroni alpha = {exp_a.get('bonferroni_alpha', 0.0167)})

| Comparison | t-stat | p-value | Significant? |
|-----------|--------|---------|-------------|
{ttest_str}

Combined model best params: n_estimators={exp_a['combined_best_params']['n_estimators']}, max_features={exp_a['combined_best_params']['max_features']}
Combined OOB accuracy: {exp_a['combined_oob_accuracy']:.4f}

---

## Experiment B: Cross-Correlation Matrix

Full Spearman correlation between all 430 S x 133 V features ({exp_b['total_pairs']:,} pairs).

| Metric | Value |
|--------|-------|
| Mean |rho| | {exp_b['mean_abs_rho']:.4f} |
| Max |rho| | {exp_b['max_abs_rho']:.4f} |
| Median |rho| | {exp_b['median_abs_rho']:.4f} |
| Fraction |rho| > 0.3 | {exp_b['frac_above_0.3']*100:.2f}% |
| Fraction |rho| > 0.5 | {exp_b['frac_above_0.5']*100:.2f}% |
| Fraction |rho| > 0.7 | {exp_b['frac_above_0.7']*100:.2f}% |

### Top-10 Most Correlated Cross-Pairs

| S Feature | V Feature | Spearman rho |
|-----------|-----------|-------------|
{pairs_str}

---

## Experiment C: Feature Importance Redistribution

When S and V features compete in one combined model:

| Metric | Value |
|--------|-------|
| S total importance share | {exp_c['s_share_pct']:.1f}% |
| V total importance share | {exp_c['v_share_pct']:.1f}% |
| S features in top-30 | {exp_c['s_features_in_top30']} |
| V features in top-30 | {exp_c['v_features_in_top30']} |
| Mean rank shift (S) | {exp_c['mean_rank_shift_s']:.1f} |
| Mean rank shift (V) | {exp_c['mean_rank_shift_v']:.1f} |

### Top-10 Combined Model Features

| Rank | Feature | Origin | Importance | Solo Rank | Combined Rank |
|------|---------|--------|-----------|-----------|---------------|
{top_feat_str}

---

## Experiment D: Ablation / Unique Contribution

### Block Permutation

| Metric | Value |
|--------|-------|
| Baseline accuracy | {bp['baseline_accuracy']:.4f} |
| Mean drop permuting S | {bp['mean_drop_permute_s']:.4f} +/- {bp['std_drop_permute_s']:.4f} |
| Mean drop permuting V | {bp['mean_drop_permute_v']:.4f} +/- {bp['std_drop_permute_v']:.4f} |
| S unique contribution | {bp['s_unique_pct']:.1f}% |
| V unique contribution | {bp['v_unique_pct']:.1f}% |

### Incremental Features (by combined importance)

| k | OOB Accuracy | S Features | V Features |
|---|-------------|------------|------------|
{incr_str}

---

## Experiment E: Cross-Prediction

| Metric | Value |
|--------|-------|
| Agreement rate (S vs V predictions) | {exp_e['agreement_rate']:.4f} |
| Cohen's kappa | {exp_e['cohen_kappa']:.4f} |
| Spearman rho (P(Class1) probabilities) | {exp_e['spearman_proba_rho']:.4f} |
| S -> V cross-prediction OOB | {exp_e['s_to_v_cross_prediction_oob']:.4f} |
| V -> S cross-prediction OOB | {exp_e['v_to_s_cross_prediction_oob']:.4f} |

---

## Experiment F: Validation Set Comparison

See `validation_comparison.csv` and `mcnemar_results.yaml` for detailed results.

---

## Plots

1. `cross_accuracy_comparison.png` - Grouped bar: S vs V vs Combined accuracy
2. `cross_corr_heatmap_top30.png` - Top-30 S x Top-30 V Spearman heatmap
3. `cross_corr_heatmap_full.png` - Full 430x133 correlation matrix
4. `combined_importance_top30.png` - Top-30 combined features colored by origin
5. `importance_rank_shift.png` - Solo vs combined rank scatter
6. `ablation_block_importance.png` - Block permutation bars + unique/shared
7. `incremental_feature_curve.png` - Accuracy vs features with S/V fraction
8. `cross_prediction_scatter.png` - P(Class1|S) vs P(Class1|V) scatter
9. `disagreement_profiles.png` - Feature heatmaps for disagreement regions
10. `verdict_summary.png` - Venn diagram + verdict text

---

## Decision Framework Applied

| Finding | Conclusion |
|---------|------------|
| Combined acc <= max(S,V) + 0.01 AND cross-prediction > 0.75 | One set suffices |
| Combined acc > max(S,V) + 0.01 AND both blocks contribute > 2% unique | Both needed |
| Combined improves BUT one block < 1% unique | Marginal benefit from smaller set |
| Cross-prediction near chance AND low cross-correlation | Truly independent, combining most valuable |

**Applied to our data -> {verdict['verdict']}**
"""

    md_path = os.path.join(REPORT_DIR, "analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Analysis written to %s", md_path)


# ====================================================================
# Main
# ====================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.makedirs(REPORT_DIR, exist_ok=True)

    logger.info("Generating 10 plots for Step 7.16 cross-analysis...")

    plot_1_accuracy_comparison()
    plot_2_cross_corr_heatmap_top30()
    plot_3_cross_corr_heatmap_full()
    plot_4_combined_importance_top30()
    plot_5_rank_shift()
    plot_6_ablation()
    plot_7_incremental_features()
    plot_8_cross_prediction_scatter()
    plot_9_disagreement_profiles()
    plot_10_verdict_summary()

    logger.info("Generating analysis.md...")
    generate_analysis_md()

    logger.info("All 10 plots + analysis.md generated in %s", REPORT_DIR)


if __name__ == "__main__":
    main()
