"""
Step 5 Visualization: Generate 20 plots + analysis.md for feature evaluation.

Outputs to implementation/reports/step5/
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_S = os.path.join(PROJECT_ROOT, "data", "evaluation")
EVAL_V = os.path.join(PROJECT_ROOT, "data", "evaluation_v")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
REPORTS = os.path.join(PROJECT_ROOT, "reports", "step5")


def load_data():
    """Load all Step 5 outputs."""
    data = {}
    # VQI-S
    data["spearman_s"] = pd.read_csv(os.path.join(EVAL_S, "spearman_correlations.csv"))
    data["removed_s"] = pd.read_csv(os.path.join(EVAL_S, "removed_redundant_features.csv"))
    data["importance_s"] = pd.read_csv(os.path.join(EVAL_S, "rf_importance_rankings.csv"))
    data["erc_s"] = pd.read_csv(os.path.join(EVAL_S, "erc_per_feature.csv"))
    data["corr_s"] = np.load(os.path.join(EVAL_S, "feature_correlation_matrix.npy"))
    with open(os.path.join(EVAL_S, "feature_selection_summary.yaml"), encoding="utf-8") as f:
        data["summary_s"] = yaml.safe_load(f)
    with open(os.path.join(EVAL_S, "selected_features.txt"), encoding="utf-8") as f:
        data["selected_s"] = [l.strip() for l in f if l.strip()]

    # VQI-V
    data["spearman_v"] = pd.read_csv(os.path.join(EVAL_V, "spearman_correlations.csv"))
    data["removed_v"] = pd.read_csv(os.path.join(EVAL_V, "removed_redundant_features.csv"))
    data["importance_v"] = pd.read_csv(os.path.join(EVAL_V, "rf_importance_rankings.csv"))
    data["erc_v"] = pd.read_csv(os.path.join(EVAL_V, "erc_per_feature.csv"))
    data["corr_v"] = np.load(os.path.join(EVAL_V, "feature_correlation_matrix.npy"))
    with open(os.path.join(EVAL_V, "feature_selection_summary.yaml"), encoding="utf-8") as f:
        data["summary_v"] = yaml.safe_load(f)
    with open(os.path.join(EVAL_V, "selected_features.txt"), encoding="utf-8") as f:
        data["selected_v"] = [l.strip() for l in f if l.strip()]

    # Feature names
    with open(os.path.join(FEATURES_DIR, "feature_names_s.json"), encoding="utf-8") as f:
        data["names_s"] = json.load(f)
    with open(os.path.join(FEATURES_DIR, "feature_names_v.json"), encoding="utf-8") as f:
        data["names_v"] = json.load(f)

    return data


# ---- VQI-S Plots (1-13) ----

def plot_spearman_barplot(data, out_dir):
    """Plot 1: Top 50 features by |rho|, horizontal bars."""
    df = data["spearman_s"].sort_values("abs_rho_mean", ascending=False).head(50)
    fig, ax = plt.subplots(figsize=(10, 14))
    colors = ["#2196F3" if r > 0 else "#F44336" for r in df["rho_mean"]]
    ax.barh(range(len(df)), df["abs_rho_mean"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_name"].values, fontsize=7)
    ax.set_xlabel("|Spearman rho| (mean across P1-P3)")
    ax.set_title("VQI-S: Top 50 Features by Spearman Correlation with Fisher d'")
    ax.invert_yaxis()
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.5, label="|rho|=0.3")
    ax.axvline(0.2, color="orange", linestyle="--", alpha=0.5, label="|rho|=0.2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "spearman_correlation_barplot.png"), dpi=150)
    plt.close(fig)
    print("  [1/20] spearman_correlation_barplot.png")


def plot_redundancy_dendrogram(data, out_dir):
    """Plot 2: Hierarchical clustering dendrogram with threshold line."""
    corr = data["corr_s"]
    # Use absolute correlation as similarity, convert to distance
    np.fill_diagonal(corr, 0)
    abs_corr = np.abs(corr)
    # Cap at 1.0 for numerical stability
    abs_corr = np.clip(abs_corr, 0, 1)
    dist = 1 - abs_corr
    np.fill_diagonal(dist, 0)
    # Make symmetric
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.05,
               above_threshold_color="#888888")
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5,
               label="r=0.95 threshold")
    ax.set_xlabel("Features (513 valid)")
    ax.set_ylabel("Distance (1 - |r|)")
    ax.set_title("VQI-S: Feature Correlation Dendrogram")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "redundancy_removal_dendrogram.png"), dpi=150)
    plt.close(fig)
    print("  [2/20] redundancy_removal_dendrogram.png")


def plot_redundancy_table(data, out_dir):
    """Plot 3: Removed features table."""
    df = data["removed_s"].sort_values("pearson_r", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.3)))
    ax.axis("off")
    if len(df) == 0:
        ax.text(0.5, 0.5, "No redundant features removed", ha="center", va="center")
    else:
        table = ax.table(
            cellText=df[["removed_feature", "kept_feature", "pearson_r",
                         "removed_abs_rho", "kept_abs_rho"]].round(4).values,
            colLabels=["Removed", "Kept", "Pearson r", "Removed |rho|", "Kept |rho|"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(range(5))
    ax.set_title("VQI-S: Redundant Feature Pairs (top 30 by |r|)", fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "redundancy_removal_table.png"), dpi=150)
    plt.close(fig)
    print("  [3/20] redundancy_removal_table.png")


def plot_rf_importance_barplot(data, out_dir):
    """Plot 4: RF Gini importance for all post-redundancy features."""
    df = data["importance_s"].sort_values("importance", ascending=False)
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.05)))
    ax.barh(range(n), df["importance"].values, color="#4CAF50", height=0.8)
    # Label top 20
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        ax.text(row["importance"] + 0.0005, i, row["feature_name"], fontsize=5, va="center")
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"VQI-S: RF Feature Importance ({n} selected features)")
    ax.invert_yaxis()
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_gini_importance_barplot.png"), dpi=150)
    plt.close(fig)
    print("  [4/20] rf_gini_importance_barplot.png")


def plot_erc_per_feature(data, out_dir):
    """Plot 5: ERC curves for top 20 features at FNMR=10%."""
    erc = data["erc_s"].sort_values("auc_mean").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(erc)))
    for i, (_, row) in enumerate(erc.iterrows()):
        aucs = [row[f"auc_fnmr10_{p}"] for p in ["P1", "P2", "P3"]]
        ax.barh(i, np.mean(aucs), color=colors[i], height=0.7)
    ax.set_yticks(range(len(erc)))
    ax.set_yticklabels(erc["feature_name"].values, fontsize=7)
    ax.set_xlabel("ERC AUC (lower = better quality predictor)")
    ax.set_title("VQI-S: Top 20 Features by ERC AUC (FNMR=10%)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "erc_per_feature.png"), dpi=150)
    plt.close(fig)
    print("  [5/20] erc_per_feature.png")


def plot_selection_summary_table(data, out_dir):
    """Plot 6: Summary table of final selected features."""
    imp = data["importance_s"].sort_values("importance", ascending=False)
    sp = data["spearman_s"].set_index("feature_name")
    top30 = imp.head(30).copy()
    top30["abs_rho"] = top30["feature_name"].map(sp["abs_rho_mean"])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    table = ax.table(
        cellText=top30[["feature_name", "importance", "abs_rho"]].round(4).values,
        colLabels=["Feature", "RF Importance", "|Spearman rho|"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(3))
    ax.set_title(f"VQI-S: Top 30 Selected Features (N_selected={len(imp)})", fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_selection_summary_table.png"), dpi=150)
    plt.close(fig)
    print("  [6/20] feature_selection_summary_table.png")


def plot_selection_funnel(data, out_dir):
    """Plot 7: Selection stage funnel."""
    s = data["summary_s"]
    stages = ["Candidates", "Valid\n(non-constant)", "Post-redundancy\n(|r|<0.95)",
              f"Selected\n(RF pruning)"]
    counts = [544, 513, 449, s["n_selected"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(stages)), counts, color=["#9E9E9E", "#64B5F6", "#4CAF50", "#FF9800"],
                  width=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(c), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Feature Count")
    ax.set_title("VQI-S: Feature Selection Funnel")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selection_stage_funnel.png"), dpi=150)
    plt.close(fig)
    print("  [7/20] selection_stage_funnel.png")


def plot_gini_lollipop(data, out_dir):
    """Plot 8: Lollipop chart for top 30 selected features."""
    df = data["importance_s"].sort_values("importance", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = range(len(df))
    ax.hlines(y, 0, df["importance"].values, color="#2196F3", linewidth=1.5)
    ax.plot(df["importance"].values, y, "o", color="#1565C0", markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature_name"].values, fontsize=7)
    ax.set_xlabel("Gini Importance")
    ax.set_title("VQI-S: Top 30 Features (Lollipop)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "gini_importance_lollipop.png"), dpi=150)
    plt.close(fig)
    print("  [8/20] gini_importance_lollipop.png")


def plot_parallel_coordinates(data, out_dir):
    """Plot 9: Parallel coordinates for top 10 features, 200 samples."""
    X_s = np.load(os.path.join(FEATURES_DIR, "features_s_train.npy"))
    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "training_set_final.csv"))
    labels = train_df["label"].values

    top10 = data["importance_s"].sort_values("importance", ascending=False).head(10)
    names_s = data["names_s"]
    feat_idx = [names_s.index(n) for n in top10["feature_name"]]

    # Sample 100 per class
    rng = np.random.RandomState(42)
    idx0 = rng.choice(np.where(labels == 0)[0], 100, replace=False)
    idx1 = rng.choice(np.where(labels == 1)[0], 100, replace=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx_group, color, label in [(idx0, "#F44336", "Class 0"), (idx1, "#2196F3", "Class 1")]:
        for i in idx_group:
            vals = X_s[i, feat_idx]
            # Normalize per feature for visualization
            ax.plot(range(10), vals, color=color, alpha=0.1, linewidth=0.5)
        # Legend proxy
        ax.plot([], [], color=color, label=label, linewidth=2)

    ax.set_xticks(range(10))
    ax.set_xticklabels(top10["feature_name"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Feature Value (raw)")
    ax.set_title("VQI-S: Parallel Coordinates (top 10 features, 200 samples)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selected_features_parallel_coordinates.png"), dpi=150)
    plt.close(fig)
    print("  [9/20] selected_features_parallel_coordinates.png")


def plot_pruning_waterfall(data, out_dir):
    """Plot 10: Feature pruning waterfall chart."""
    stages = ["Total", "Zero-var\nremoved", "Redundancy\nremoved", "RF\npruned"]
    removed = [0, 31, 64, 449 - data["summary_s"]["n_selected"]]
    remaining = [544, 513, 449, data["summary_s"]["n_selected"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(stages)), remaining, color="#4CAF50", label="Remaining")
    bottom = np.array(remaining)
    ax.bar(range(len(stages)), removed, bottom=bottom, color="#F44336", alpha=0.5, label="Removed")
    for i, (rem, rem_ct) in enumerate(zip(remaining, removed)):
        ax.text(i, rem + rem_ct + 5, f"-{rem_ct}", ha="center", fontsize=9,
                color="#F44336" if rem_ct > 0 else "black")
        ax.text(i, rem / 2, str(rem), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Feature Count")
    ax.set_title("VQI-S: Feature Pruning Waterfall")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_pruning_waterfall.png"), dpi=150)
    plt.close(fig)
    print("  [10/20] feature_pruning_waterfall.png")


def plot_correlation_network(data, out_dir):
    """Plot 11: Correlation network before/after redundancy removal (top features only)."""
    sp = data["spearman_s"].sort_values("abs_rho_mean", ascending=False)
    top30_names = sp.head(30)["feature_name"].tolist()

    # Get correlation among top 30 from full matrix
    names_s = data["names_s"]
    valid_mask = np.std(np.load(os.path.join(FEATURES_DIR, "features_s_train.npy")), axis=0) > 1e-12
    valid_idx = np.where(valid_mask)[0]
    valid_names = [names_s[i] for i in valid_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for panel, (title, feature_set) in enumerate([
        ("Before removal", top30_names),
        ("After removal", [n for n in top30_names if n in data["selected_s"]]),
    ]):
        ax = axes[panel]
        idx_in_valid = [valid_names.index(n) for n in feature_set if n in valid_names]
        if len(idx_in_valid) < 2:
            ax.text(0.5, 0.5, "Too few features", ha="center", va="center")
            ax.set_title(title)
            continue

        sub_corr = data["corr_s"][np.ix_(idx_in_valid, idx_in_valid)]
        n = len(idx_in_valid)

        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # Draw edges for |r| > 0.5
        for i in range(n):
            for j in range(i + 1, n):
                r = abs(sub_corr[i, j])
                if r > 0.5:
                    alpha = min(1.0, r)
                    color = "red" if r > 0.95 else ("orange" if r > 0.8 else "lightgray")
                    ax.plot([x[i], x[j]], [y[i], y[j]], color=color, alpha=alpha * 0.6, linewidth=0.8)

        ax.scatter(x, y, s=30, c="#1565C0", zorder=5)
        for i, name in enumerate([feature_set[k] for k in range(min(n, len(feature_set)))]):
            short = name[:15]
            ax.annotate(short, (x[i], y[i]), fontsize=5, ha="center", va="bottom")
        ax.set_title(title)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("VQI-S: Feature Correlation Network (top 30 by |rho|)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selection_correlation_network_before_after.png"), dpi=150)
    plt.close(fig)
    print("  [11/20] selection_correlation_network_before_after.png")


def plot_erc_cumulative(data, out_dir):
    """Plot 12: Cumulative ERC AUC as features are added by importance."""
    imp = data["importance_s"].sort_values("importance", ascending=False)
    erc = data["erc_s"].set_index("feature_name")

    cumulative_auc = []
    for i in range(1, len(imp) + 1):
        top_names = imp["feature_name"].values[:i]
        mean_auc = erc.loc[erc.index.isin(top_names), "auc_mean"].mean()
        cumulative_auc.append(mean_auc)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumulative_auc) + 1), cumulative_auc, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Number of Features (by importance rank)")
    ax.set_ylabel("Mean ERC AUC (lower = better)")
    ax.set_title("VQI-S: Cumulative ERC AUC vs Feature Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "erc_cumulative_features.png"), dpi=150)
    plt.close(fig)
    print("  [12/20] erc_cumulative_features.png")


# ---- VQI-V Plots (14-19) ----

def plot_spearman_v_barplot(data, out_dir):
    """Plot 14: VQI-V top features by |rho|."""
    df = data["spearman_v"].sort_values("abs_rho_mean", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3" if r > 0 else "#F44336" for r in df["rho_mean"]]
    ax.barh(range(len(df)), df["abs_rho_mean"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_name"].values, fontsize=7)
    ax.set_xlabel("|Spearman rho| (mean across P1-P3)")
    ax.set_title("VQI-V: Top 30 Features by Spearman Correlation with Fisher d'")
    ax.invert_yaxis()
    ax.axvline(0.2, color="orange", linestyle="--", alpha=0.5, label="|rho|=0.2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "spearman_v_correlation_barplot.png"), dpi=150)
    plt.close(fig)
    print("  [14/20] spearman_v_correlation_barplot.png")


def plot_redundancy_v_dendrogram(data, out_dir):
    """Plot 15: VQI-V correlation dendrogram."""
    corr = data["corr_v"].copy()
    np.fill_diagonal(corr, 0)
    abs_corr = np.clip(np.abs(corr), 0, 1)
    dist = 1 - abs_corr
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.05,
               above_threshold_color="#888888")
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5,
               label="r=0.95 threshold")
    ax.set_xlabel("Features (161)")
    ax.set_ylabel("Distance (1 - |r|)")
    ax.set_title("VQI-V: Feature Correlation Dendrogram")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "redundancy_v_removal_dendrogram.png"), dpi=150)
    plt.close(fig)
    print("  [15/20] redundancy_v_removal_dendrogram.png")


def plot_rf_v_importance_barplot(data, out_dir):
    """Plot 16: VQI-V RF importance."""
    df = data["importance_v"].sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.12)))
    ax.barh(range(len(df)), df["importance"].values, color="#9C27B0", height=0.7)
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        ax.text(row["importance"] + 0.001, i, row["feature_name"], fontsize=6, va="center")
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"VQI-V: RF Feature Importance ({len(df)} selected features)")
    ax.invert_yaxis()
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "rf_v_gini_importance_barplot.png"), dpi=150)
    plt.close(fig)
    print("  [16/20] rf_v_gini_importance_barplot.png")


def plot_selection_v_funnel(data, out_dir):
    """Plot 17: VQI-V selection funnel."""
    s = data["summary_v"]
    stages = ["Candidates\n(161)", "Valid\n(non-constant)", "Post-redundancy\n(|r|<0.95)",
              f"Selected\n(RF pruning)"]
    counts = [161, 161, 133, s["n_selected"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(stages)), counts, color=["#9E9E9E", "#CE93D8", "#AB47BC", "#7B1FA2"],
                  width=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(c), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Feature Count")
    ax.set_title("VQI-V: Feature Selection Funnel")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selection_v_stage_funnel.png"), dpi=150)
    plt.close(fig)
    print("  [17/20] selection_v_stage_funnel.png")


def plot_erc_v_per_feature(data, out_dir):
    """Plot 18: VQI-V ERC for top 20 features."""
    erc = data["erc_v"].sort_values("auc_mean").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(erc)))
    for i, (_, row) in enumerate(erc.iterrows()):
        aucs = [row[f"auc_fnmr10_{p}"] for p in ["P1", "P2", "P3"]]
        ax.barh(i, np.mean(aucs), color=colors[i], height=0.7)
    ax.set_yticks(range(len(erc)))
    ax.set_yticklabels(erc["feature_name"].values, fontsize=7)
    ax.set_xlabel("ERC AUC (lower = better quality predictor)")
    ax.set_title("VQI-V: Top 20 Features by ERC AUC (FNMR=10%)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "erc_v_per_feature.png"), dpi=150)
    plt.close(fig)
    print("  [18/20] erc_v_per_feature.png")


def plot_v_selection_summary_table(data, out_dir):
    """Plot 19: VQI-V summary table."""
    imp = data["importance_v"].sort_values("importance", ascending=False)
    sp = data["spearman_v"].set_index("feature_name")
    top30 = imp.head(30).copy()
    top30["abs_rho"] = top30["feature_name"].map(sp["abs_rho_mean"])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    table = ax.table(
        cellText=top30[["feature_name", "importance", "abs_rho"]].round(4).values,
        colLabels=["Feature", "RF Importance", "|Spearman rho|"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(3))
    ax.set_title(f"VQI-V: Top 30 Selected Features (N_selected_V={len(imp)})", fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_v_selection_summary_table.png"), dpi=150)
    plt.close(fig)
    print("  [19/20] feature_v_selection_summary_table.png")


# ---- Cross-score Plot (20) ----

def plot_s_vs_v_overlap(data, out_dir):
    """Plot 20: VQI-S vs VQI-V feature overlap/comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart comparing counts
    ax = axes[0]
    categories = ["Candidates", "Zero-var\nremoved", "Redundancy\nremoved", "Selected"]
    s_vals = [544, 31, 64, data["summary_s"]["n_selected"]]
    v_vals = [161, 0, 28, data["summary_v"]["n_selected"]]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, s_vals, w, label="VQI-S", color="#2196F3")
    ax.bar(x + w / 2, v_vals, w, label="VQI-V", color="#9C27B0")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Pipeline Comparison")
    ax.legend()

    # Right: Top features comparison
    ax = axes[1]
    top_s = data["importance_s"].sort_values("importance", ascending=False).head(15)
    top_v = data["importance_v"].sort_values("importance", ascending=False).head(15)

    y_s = range(15)
    y_v = range(15, 30)
    ax.barh(y_s, top_s["importance"].values, color="#2196F3", height=0.7, label="VQI-S")
    ax.barh(y_v, top_v["importance"].values, color="#9C27B0", height=0.7, label="VQI-V")
    ax.set_yticks(list(y_s) + list(y_v))
    ax.set_yticklabels(
        list(top_s["feature_name"]) + list(top_v["feature_name"]),
        fontsize=6,
    )
    ax.set_xlabel("Gini Importance")
    ax.set_title("Top 15 Features Each")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    fig.suptitle("VQI-S vs VQI-V: Feature Selection Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_s_vs_v_overlap.png"), dpi=150)
    plt.close(fig)
    print("  [20/20] feature_s_vs_v_overlap.png")


# ---- Analysis.md (Plot 13) ----

def write_analysis(data, out_dir):
    """Write analysis.md with key findings."""
    ss = data["summary_s"]
    sv = data["summary_v"]
    imp_s = data["importance_s"].sort_values("importance", ascending=False)
    imp_v = data["importance_v"].sort_values("importance", ascending=False)
    sp_s = data["spearman_s"]
    sp_v = data["spearman_v"]
    erc_s = data["erc_s"].sort_values("auc_mean")
    erc_v = data["erc_v"].sort_values("auc_mean")

    # Concentration metrics for VQI-S
    s_imps = imp_s["importance"].values
    top10_share_s = s_imps[:10].sum() / s_imps.sum() * 100
    top30_share_s = s_imps[:30].sum() / s_imps.sum() * 100

    v_imps = imp_v["importance"].values
    top10_share_v = v_imps[:10].sum() / v_imps.sum() * 100
    top30_share_v = v_imps[:30].sum() / v_imps.sum() * 100

    lines = [
        "# Step 5: Feature Evaluation and Selection - Analysis",
        "",
        f"**Date:** 2026-02-16",
        f"**Pipeline runtime:** ~2.1 minutes",
        "",
        "## VQI-S (Signal Quality)",
        "",
        f"- **Candidates:** 544 -> 513 valid (31 zero-variance: 8 DNSMOS/NISQA + 1 ClickRate + 22 histogram bins)",
        f"- **Post-redundancy:** 449 (64 pairs removed at |r| > 0.95)",
        f"- **Selected:** {ss['n_selected']} features (RF pruning: {ss['n_iterations']} iterations)",
        f"- **OOB accuracy:** {ss['final_oob_accuracy']:.4f}",
        "",
        "### Spearman Correlations",
        f"- Features with |rho| > 0.3: {(sp_s['abs_rho_mean'] > 0.3).sum()}",
        f"- Features with |rho| > 0.2: {(sp_s['abs_rho_mean'] > 0.2).sum()}",
        f"- Max |rho|: {sp_s['abs_rho_mean'].max():.4f} ({sp_s.loc[sp_s['abs_rho_mean'].idxmax(), 'feature_name']})",
        "",
        "### Top 10 Features by RF Importance",
        "| Rank | Feature | Importance | |rho| |",
        "|------|---------|------------|-------|",
    ]
    sp_lookup = dict(zip(sp_s["feature_name"], sp_s["abs_rho_mean"]))
    for i, (_, row) in enumerate(imp_s.head(10).iterrows()):
        rho = sp_lookup.get(row["feature_name"], 0)
        lines.append(f"| {i+1} | {row['feature_name']} | {row['importance']:.6f} | {rho:.4f} |")

    lines += [
        "",
        "### Importance Concentration",
        f"- Top 10 features: {top10_share_s:.1f}% of total importance",
        f"- Top 30 features: {top30_share_s:.1f}% of total importance",
        "",
        "### Top 5 ERC Features (best quality predictors)",
        "| Rank | Feature | ERC AUC |",
        "|------|---------|---------|",
    ]
    for i, (_, row) in enumerate(erc_s.head(5).iterrows()):
        lines.append(f"| {i+1} | {row['feature_name']} | {row['auc_mean']:.4f} |")

    lines += [
        "",
        "## VQI-V (Voice Distinctiveness)",
        "",
        f"- **Candidates:** 161 -> 161 valid (0 zero-variance)",
        f"- **Post-redundancy:** 133 (28 pairs removed at |r| > 0.95)",
        f"- **Selected:** {sv['n_selected']} features (RF pruning: {sv['n_iterations']} iterations)",
        f"- **OOB accuracy:** {sv['final_oob_accuracy']:.4f}",
        "",
        "### Spearman Correlations",
        f"- Features with |rho| > 0.2: {(sp_v['abs_rho_mean'] > 0.2).sum()}",
        f"- Max |rho|: {sp_v['abs_rho_mean'].max():.4f} ({sp_v.loc[sp_v['abs_rho_mean'].idxmax(), 'feature_name']})",
        "",
        "### Top 10 Features by RF Importance",
        "| Rank | Feature | Importance | |rho| |",
        "|------|---------|------------|-------|",
    ]
    sp_v_lookup = dict(zip(sp_v["feature_name"], sp_v["abs_rho_mean"]))
    for i, (_, row) in enumerate(imp_v.head(10).iterrows()):
        rho = sp_v_lookup.get(row["feature_name"], 0)
        lines.append(f"| {i+1} | {row['feature_name']} | {row['importance']:.6f} | {rho:.4f} |")

    lines += [
        "",
        "### Importance Concentration",
        f"- Top 10 features: {top10_share_v:.1f}% of total importance",
        f"- Top 30 features: {top30_share_v:.1f}% of total importance",
        "",
        "### Top 5 ERC Features (best quality predictors)",
        "| Rank | Feature | ERC AUC |",
        "|------|---------|---------|",
    ]
    for i, (_, row) in enumerate(erc_v.head(5).iterrows()):
        lines.append(f"| {i+1} | {row['feature_name']} | {row['auc_mean']:.4f} |")

    lines += [
        "",
        "## VQI-S vs VQI-V Comparison",
        "",
        f"| Metric | VQI-S | VQI-V |",
        f"|--------|-------|-------|",
        f"| Candidates | 544 | 161 |",
        f"| Zero-variance | 31 | 0 |",
        f"| Redundancy removed | 64 | 28 |",
        f"| Selected | {ss['n_selected']} | {sv['n_selected']} |",
        f"| OOB accuracy | {ss['final_oob_accuracy']:.4f} | {sv['final_oob_accuracy']:.4f} |",
        f"| Top-10 importance share | {top10_share_s:.1f}% | {top10_share_v:.1f}% |",
        "",
        "### Key Findings",
        "",
        "1. **Low pruning rate:** RF importance pruning removed very few features (19 for S, 0 for V) ",
        "   because all post-redundancy features have importance above the 0.5% threshold. This suggests ",
        "   the redundancy removal stage did most of the work in eliminating uninformative features.",
        "",
        "2. **Spearman correlations are modest:** Only 3 VQI-S features and 1 VQI-V feature exceed ",
        "   |rho| > 0.3. This is expected because Fisher d' is a per-speaker metric, while features ",
        "   are per-utterance, and quality affects recognition through complex, non-linear pathways.",
        "",
        "3. **VQI-V has higher importance concentration:** The top VQI-V feature (V_LTFD_Entropy) has ",
        "   importance 0.142 vs VQI-S top (SpeakerTurns) at 0.069, suggesting VQI-V relies more ",
        "   heavily on a few key features.",
        "",
        "4. **Both scores have strong OOB:** ~82% OOB accuracy for both VQI-S and VQI-V indicates ",
        "   good class separation from the feature sets, supporting viable model training in Step 6.",
        "",
        "5. **Feature types:** VQI-S top features span signal (SNR, spectral flux), environment ",
        "   (RT60, C50), temporal (SpeakerTurns, SpeechContinuity) and clinical (DSI). VQI-V top ",
        "   features are dominated by dynamic cepstral features (DeltaMFCC) and long-term ",
        "   distributional features.",
        "",
        "## Output Files",
        "",
        "### VQI-S (`data/evaluation/`)",
        "- `spearman_correlations.csv` (544 rows)",
        "- `feature_correlation_matrix.npy` (513 x 513)",
        "- `removed_redundant_features.csv` (64 pairs)",
        "- `rf_importance_rankings.csv` (430 features)",
        "- `rf_pruning_history.csv` (iteration details)",
        "- `selected_features.txt` (430 feature names)",
        "- `feature_selection_summary.yaml`",
        "- `erc_per_feature.csv` (430 features)",
        "",
        "### VQI-V (`data/evaluation_v/`)",
        "- `spearman_correlations.csv` (161 rows)",
        "- `feature_correlation_matrix.npy` (161 x 161)",
        "- `removed_redundant_features.csv` (28 pairs)",
        "- `rf_importance_rankings.csv` (133 features)",
        "- `rf_pruning_history.csv`",
        "- `selected_features.txt` (133 feature names)",
        "- `feature_selection_summary.yaml`",
        "- `erc_per_feature.csv` (133 features)",
    ]

    path = os.path.join(out_dir, "analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [13/20] analysis.md")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    print("Loading Step 5 data...")
    data = load_data()
    print(f"Generating 20 outputs to {REPORTS}")

    # VQI-S plots (1-12)
    plot_spearman_barplot(data, REPORTS)
    plot_redundancy_dendrogram(data, REPORTS)
    plot_redundancy_table(data, REPORTS)
    plot_rf_importance_barplot(data, REPORTS)
    plot_erc_per_feature(data, REPORTS)
    plot_selection_summary_table(data, REPORTS)
    plot_selection_funnel(data, REPORTS)
    plot_gini_lollipop(data, REPORTS)
    plot_parallel_coordinates(data, REPORTS)
    plot_pruning_waterfall(data, REPORTS)
    plot_correlation_network(data, REPORTS)
    plot_erc_cumulative(data, REPORTS)

    # Analysis (13)
    write_analysis(data, REPORTS)

    # VQI-V plots (14-19)
    plot_spearman_v_barplot(data, REPORTS)
    plot_redundancy_v_dendrogram(data, REPORTS)
    plot_rf_v_importance_barplot(data, REPORTS)
    plot_selection_v_funnel(data, REPORTS)
    plot_erc_v_per_feature(data, REPORTS)
    plot_v_selection_summary_table(data, REPORTS)

    # Cross-score (20)
    plot_s_vs_v_overlap(data, REPORTS)

    print(f"\nDone! 20 outputs in {REPORTS}")


if __name__ == "__main__":
    main()
