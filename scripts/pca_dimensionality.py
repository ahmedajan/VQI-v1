"""
PCA Effective Dimensionality Analysis

Answers: how redundant are the VQI-S (430) and VQI-V (133) feature spaces?
Produces variance-explained curves and a summary report.

Outputs to implementation/reports/pca/
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_S = os.path.join(PROJECT_ROOT, "data", "training")
TRAIN_V = os.path.join(PROJECT_ROOT, "data", "training_v")
REPORTS = os.path.join(PROJECT_ROOT, "reports", "pca")

COLOR_S = "#2196F3"
COLOR_V = "#9C27B0"
COLOR_G = "#4CAF50"
THRESHOLDS = [0.90, 0.95, 0.99]
THRESHOLD_COLORS = ["#4CAF50", "#FF9800", "#F44336"]
THRESHOLD_LABELS = ["90%", "95%", "99%"]


def load_data():
    """Load X_train for both VQI-S and VQI-V."""
    data = {}
    data["X_s"] = np.load(os.path.join(TRAIN_S, "X_train.npy"))
    data["X_v"] = np.load(os.path.join(TRAIN_V, "X_train.npy"))
    print(f"  VQI-S: {data['X_s'].shape}")
    print(f"  VQI-V: {data['X_v'].shape}")
    return data


def run_pca(X):
    """StandardScaler -> PCA(n_components=None) -> return fitted PCA."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    return pca


def n_components_for_threshold(cumvar, threshold):
    """Return number of components needed to reach a variance threshold."""
    return int(np.searchsorted(cumvar, threshold)) + 1


def plot_variance_curve(pca, label, color, out_path):
    """Cumulative explained variance curve with threshold annotations."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_total = len(cumvar)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_total + 1), cumvar, color=color, linewidth=2, label=label)

    for thresh, tc, tl in zip(THRESHOLDS, THRESHOLD_COLORS, THRESHOLD_LABELS):
        n_pc = n_components_for_threshold(cumvar, thresh)
        ax.axhline(thresh, color=tc, linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(n_pc, color=tc, linestyle=":", alpha=0.4, linewidth=1)
        ax.annotate(
            f"{tl}: {n_pc} PCs",
            xy=(n_pc, thresh),
            xytext=(n_pc + n_total * 0.05, thresh - 0.03),
            fontsize=9,
            color=tc,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=tc, lw=1.2),
        )

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"{label}: Cumulative Variance Explained ({n_total} features)")
    ax.set_xlim(1, n_total)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_curve(pca_s, pca_v, out_path):
    """Both VQI-S and VQI-V cumulative variance on one figure."""
    cumvar_s = np.cumsum(pca_s.explained_variance_ratio_)
    cumvar_v = np.cumsum(pca_v.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumvar_s) + 1), cumvar_s, color=COLOR_S, linewidth=2, label="VQI-S (430 features)")
    ax.plot(range(1, len(cumvar_v) + 1), cumvar_v, color=COLOR_V, linewidth=2, label="VQI-V (133 features)")

    for thresh, tc, tl in zip(THRESHOLDS, THRESHOLD_COLORS, THRESHOLD_LABELS):
        ax.axhline(thresh, color=tc, linestyle="--", alpha=0.4, linewidth=1)
        # Annotate for S
        n_s = n_components_for_threshold(cumvar_s, thresh)
        n_v = n_components_for_threshold(cumvar_v, thresh)
        ax.plot(n_s, thresh, "o", color=COLOR_S, markersize=6, zorder=5)
        ax.plot(n_v, thresh, "s", color=COLOR_V, markersize=6, zorder=5)
        ax.annotate(f"S:{n_s}", xy=(n_s, thresh), xytext=(n_s + 8, thresh + 0.015),
                    fontsize=8, color=COLOR_S, fontweight="bold")
        ax.annotate(f"V:{n_v}", xy=(n_v, thresh), xytext=(n_v + 8, thresh - 0.025),
                    fontsize=8, color=COLOR_V, fontweight="bold")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Effective Dimensionality: VQI-S vs VQI-V")
    ax.set_xlim(1, max(len(cumvar_s), len(cumvar_v)))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_component(pca, label, color, out_path, top_n=50):
    """Scree plot: individual component variance for top N components."""
    var_ratio = pca.explained_variance_ratio_[:top_n]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(1, len(var_ratio) + 1), var_ratio, color=color, alpha=0.7, width=0.8)
    ax.plot(range(1, len(var_ratio) + 1), var_ratio, "o-", color=color, markersize=3, linewidth=1)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"{label}: Scree Plot (top {top_n} components)")
    ax.set_xlim(0.5, top_n + 0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate PC1
    ax.annotate(
        f"PC1: {var_ratio[0]:.3f}",
        xy=(1, var_ratio[0]),
        xytext=(5, var_ratio[0] * 0.95),
        fontsize=9, fontweight="bold", color=color,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_analysis(pca_s, pca_v, out_path):
    """Write analysis.md with summary table and interpretation."""
    cumvar_s = np.cumsum(pca_s.explained_variance_ratio_)
    cumvar_v = np.cumsum(pca_v.explained_variance_ratio_)

    n_s = len(pca_s.explained_variance_ratio_)
    n_v = len(pca_v.explained_variance_ratio_)

    # Component counts for thresholds
    n90_s = n_components_for_threshold(cumvar_s, 0.90)
    n95_s = n_components_for_threshold(cumvar_s, 0.95)
    n99_s = n_components_for_threshold(cumvar_s, 0.99)
    n90_v = n_components_for_threshold(cumvar_v, 0.90)
    n95_v = n_components_for_threshold(cumvar_v, 0.95)
    n99_v = n_components_for_threshold(cumvar_v, 0.99)

    # Top component contributions
    pc1_s = pca_s.explained_variance_ratio_[0]
    pc1_v = pca_v.explained_variance_ratio_[0]
    top10_s = cumvar_s[9]
    top10_v = cumvar_v[9]

    lines = [
        "# PCA Effective Dimensionality Analysis",
        "",
        f"**Date:** 2026-02-21",
        f"**Purpose:** Quantify redundancy in VQI-S ({n_s} features) and VQI-V ({n_v} features) spaces",
        "",
        "## Summary Table",
        "",
        "| Metric | VQI-S | VQI-V |",
        "|--------|-------|-------|",
        f"| Total features | {n_s} | {n_v} |",
        f"| PCs for 90% variance | {n90_s} | {n90_v} |",
        f"| PCs for 95% variance | {n95_s} | {n95_v} |",
        f"| PCs for 99% variance | {n99_s} | {n99_v} |",
        f"| Effective dim ratio (95%) | {n95_s / n_s:.4f} | {n95_v / n_v:.4f} |",
        f"| PC1 variance share | {pc1_s:.4f} | {pc1_v:.4f} |",
        f"| Top-10 PCs variance | {top10_s:.4f} | {top10_v:.4f} |",
        "",
        "## Key Finding",
        "",
        f"**VQI-S:** {n95_s} principal components capture 95% of the variance in {n_s} features — "
        f"an effective dimensionality ratio of {n95_s / n_s:.1%}. The 90% threshold is reached at just "
        f"{n90_s} PCs ({n90_s / n_s:.1%} of features).",
        "",
        f"**VQI-V:** {n95_v} principal components capture 95% of the variance in {n_v} features — "
        f"an effective dimensionality ratio of {n95_v / n_v:.1%}. The 90% threshold is reached at "
        f"{n90_v} PCs ({n90_v / n_v:.1%} of features).",
        "",
        "## Interpretation",
        "",
        f"The PCA analysis reveals substantial redundancy in both feature spaces, "
        f"even after the 0.95 correlation threshold removal in Step 5. "
        f"For VQI-S, approximately {n_s - n95_s} of {n_s} features ({(n_s - n95_s) / n_s:.0%}) "
        f"contribute less than 5% of total variance — these features largely describe the same "
        f"underlying dimensions as the top {n95_s} PCs. For VQI-V, the picture is similar: "
        f"{n_v - n95_v} of {n_v} features ({(n_v - n95_v) / n_v:.0%}) are redundant at the 95% level.",
        "",
        f"This is consistent with the feature selection analysis showing accuracy plateaus: "
        f"OOB 0.815 at k=200 vs 0.822 at k=430. The marginal features add noise rather than "
        f"new information. However, Random Forests are robust to redundant features "
        f"(random feature subsampling at each split naturally de-correlates trees), "
        f"so the redundancy does not necessarily harm model performance — it just means "
        f"a smaller feature set could achieve similar accuracy.",
        "",
        "## Variance Explained Detail",
        "",
        "### VQI-S — Top 10 Components",
        "",
        "| PC | Variance Ratio | Cumulative |",
        "|-----|---------------|------------|",
    ]

    for i in range(10):
        lines.append(
            f"| PC{i+1} | {pca_s.explained_variance_ratio_[i]:.4f} | {cumvar_s[i]:.4f} |"
        )

    lines += [
        "",
        "### VQI-V — Top 10 Components",
        "",
        "| PC | Variance Ratio | Cumulative |",
        "|-----|---------------|------------|",
    ]

    for i in range(10):
        lines.append(
            f"| PC{i+1} | {pca_v.explained_variance_ratio_[i]:.4f} | {cumvar_v[i]:.4f} |"
        )

    lines += [
        "",
        "## Output Files",
        "",
        "- `variance_curve_s.png` — cumulative variance for VQI-S",
        "- `variance_curve_v.png` — cumulative variance for VQI-V",
        "- `variance_curve_combined.png` — both on one chart",
        "- `scree_plot_s.png` — per-component variance for VQI-S (top 50)",
        "- `scree_plot_v.png` — per-component variance for VQI-V (top 50)",
        "- `analysis.md` — this file",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(REPORTS, exist_ok=True)

    print("Loading training data...")
    data = load_data()

    print("Running PCA on VQI-S...")
    pca_s = run_pca(data["X_s"])
    print("Running PCA on VQI-V...")
    pca_v = run_pca(data["X_v"])

    # Quick summary
    cumvar_s = np.cumsum(pca_s.explained_variance_ratio_)
    cumvar_v = np.cumsum(pca_v.explained_variance_ratio_)
    for label, cumvar, n_total in [("VQI-S", cumvar_s, 430), ("VQI-V", cumvar_v, 133)]:
        n90 = n_components_for_threshold(cumvar, 0.90)
        n95 = n_components_for_threshold(cumvar, 0.95)
        n99 = n_components_for_threshold(cumvar, 0.99)
        print(f"  {label}: 90%={n90}, 95%={n95}, 99%={n99} PCs (of {n_total})")

    print("Generating plots...")
    plot_variance_curve(pca_s, "VQI-S", COLOR_S,
                        os.path.join(REPORTS, "variance_curve_s.png"))
    print("  [1/5] variance_curve_s.png")

    plot_variance_curve(pca_v, "VQI-V", COLOR_V,
                        os.path.join(REPORTS, "variance_curve_v.png"))
    print("  [2/5] variance_curve_v.png")

    plot_combined_curve(pca_s, pca_v,
                        os.path.join(REPORTS, "variance_curve_combined.png"))
    print("  [3/5] variance_curve_combined.png")

    plot_per_component(pca_s, "VQI-S", COLOR_S,
                       os.path.join(REPORTS, "scree_plot_s.png"))
    print("  [4/5] scree_plot_s.png")

    plot_per_component(pca_v, "VQI-V", COLOR_V,
                       os.path.join(REPORTS, "scree_plot_v.png"))
    print("  [5/5] scree_plot_v.png")

    print("Writing analysis.md...")
    write_analysis(pca_s, pca_v, os.path.join(REPORTS, "analysis.md"))

    print(f"\nDone! 5 plots + analysis.md in {REPORTS}")


if __name__ == "__main__":
    main()
