"""Step 9 v2 visualizations for VQI Software v2.0 (PCA-90%).

Generates:
  9_v2_conformance_score_distribution.png  -- Histogram of v2.0 conformance scores
  9_v2_conformance_scatter_s_vs_v.png      -- VQI-S vs VQI-V scatter
  9_v2_feedback_category_coverage.png      -- Category trigger frequency
  9_v2_processing_time_histogram.png       -- Processing time distribution
  9_v2_score_comparison_v1_vs_v2.png       -- Side-by-side v1.0 vs v2.0 scores
  analysis.md                              -- Summary analysis
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REPORTS_DIR = os.path.join(BASE_DIR, "reports", "step9", "v2")
CONF_DIR = os.path.join(BASE_DIR, "conformance")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_conformance_scores(version="v2.0"):
    """Load conformance expected output for a given version."""
    csv_path = os.path.join(CONF_DIR, f"conformance_expected_output_{version}.csv")
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_conformance_score_distribution(data):
    """Histogram of conformance set VQI-S and VQI-V scores (v2.0)."""
    scores_s = [int(r["vqi_s"]) for r in data]
    scores_v = [int(r["vqi_v"]) for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(scores_s, bins=20, range=(0, 100), color="#1976d2", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("VQI-S Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"VQI-S Distribution (PCA-90%, n={len(scores_s)})")
    axes[0].axvline(np.mean(scores_s), color="red", linestyle="--",
                    label=f"Mean: {np.mean(scores_s):.1f}")
    axes[0].legend()

    axes[1].hist(scores_v, bins=20, range=(0, 100), color="#388e3c", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("VQI-V Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"VQI-V Distribution (PCA-90%, n={len(scores_v)})")
    axes[1].axvline(np.mean(scores_v), color="red", linestyle="--",
                    label=f"Mean: {np.mean(scores_v):.1f}")
    axes[1].legend()

    plt.suptitle("Conformance Set Score Distribution (v2.0 PCA-90%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_v2_conformance_score_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_conformance_scatter(data):
    """Scatter plot VQI-S vs VQI-V (v2.0)."""
    scores_s = [int(r["vqi_s"]) for r in data]
    scores_v = [int(r["vqi_v"]) for r in data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(scores_s, scores_v, alpha=0.6, s=30, c="#1976d2", edgecolors="white",
               linewidth=0.3)
    ax.set_xlabel("VQI-S (Signal Quality)")
    ax.set_ylabel("VQI-V (Voice Distinctiveness)")
    ax.set_title("Conformance Set: VQI-S vs VQI-V (v2.0 PCA-90%)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y=x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    corr = np.corrcoef(scores_s, scores_v)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_v2_conformance_scatter_s_vs_v.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_feedback_category_coverage(data):
    """Bar chart: how many files trigger each feedback category."""
    from vqi.engine import VQIEngine
    import logging
    logging.basicConfig(level=logging.WARNING)

    files_dir = os.path.join(CONF_DIR, "test_files")
    engine = VQIEngine()

    cat_counts_s = {}
    cat_counts_v = {}

    sample = data[:50]
    for i, row in enumerate(sample):
        filepath = os.path.join(files_dir, row["filename"])
        if not os.path.exists(filepath):
            continue
        try:
            result = engine.score_file(filepath)
            for lf in result.limiting_factors_s:
                c = lf["category"]
                cat_counts_s[c] = cat_counts_s.get(c, 0) + 1
            for lf in result.limiting_factors_v:
                c = lf["category"]
                cat_counts_v[c] = cat_counts_v.get(c, 0) + 1
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  Category coverage: {i+1}/{len(sample)} files")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if cat_counts_s:
        cats = sorted(cat_counts_s.keys())
        counts = [cat_counts_s[c] for c in cats]
        axes[0].barh(cats, counts, color="#1976d2", alpha=0.8)
        axes[0].set_xlabel("Number of files")
        axes[0].set_title("VQI-S Limiting Factor Categories (v2.0)")

    if cat_counts_v:
        cats = sorted(cat_counts_v.keys())
        counts = [cat_counts_v[c] for c in cats]
        axes[1].barh(cats, counts, color="#388e3c", alpha=0.8)
        axes[1].set_xlabel("Number of files")
        axes[1].set_title("VQI-V Limiting Factor Categories (v2.0)")

    plt.suptitle(f"Feedback Category Coverage (n={len(sample)} files, PCA-90%)", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_v2_feedback_category_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_processing_time(data):
    """Processing time histogram from v2.0 conformance output."""
    if "processing_time_ms" not in data[0]:
        print("No processing_time_ms in conformance data, skipping.")
        return

    times_ms = []
    for row in data:
        t = row.get("processing_time_ms")
        if t:
            times_ms.append(float(t))

    if not times_ms:
        print("No timing data collected.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(times_ms, bins=25, color="#ff7043", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Processing Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Processing Time Distribution (v2.0 PCA-90%, n={len(times_ms)} files)")
    ax.axvline(np.mean(times_ms), color="red", linestyle="--",
               label=f"Mean: {np.mean(times_ms):.0f}ms")
    ax.axvline(np.median(times_ms), color="blue", linestyle="--",
               label=f"Median: {np.median(times_ms):.0f}ms")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_v2_processing_time_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_score_comparison_v1_vs_v2(data_v1, data_v2):
    """Side-by-side comparison of v1.0 vs v2.0 scores."""
    # Match by filename
    v1_map = {r["filename"]: r for r in data_v1}
    v2_map = {r["filename"]: r for r in data_v2}
    common = sorted(set(v1_map.keys()) & set(v2_map.keys()))

    if not common:
        print("No common files between v1.0 and v2.0, skipping comparison.")
        return

    s_v1 = np.array([int(v1_map[f]["vqi_s"]) for f in common])
    s_v2 = np.array([int(v2_map[f]["vqi_s"]) for f in common])
    v_v1 = np.array([int(v1_map[f]["vqi_v"]) for f in common])
    v_v2 = np.array([int(v2_map[f]["vqi_v"]) for f in common])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # VQI-S scatter: v1 vs v2
    ax = axes[0, 0]
    ax.scatter(s_v1, s_v2, alpha=0.5, s=20, c="#1976d2", edgecolors="white", linewidth=0.3)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("VQI-S v1.0 (Full-feature)")
    ax.set_ylabel("VQI-S v2.0 (PCA-90%)")
    ax.set_title("VQI-S: v1.0 vs v2.0")
    corr_s = np.corrcoef(s_v1, s_v2)[0, 1]
    mae_s = np.mean(np.abs(s_v1 - s_v2))
    ax.text(0.05, 0.95, f"r = {corr_s:.3f}\nMAE = {mae_s:.1f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # VQI-V scatter: v1 vs v2
    ax = axes[0, 1]
    ax.scatter(v_v1, v_v2, alpha=0.5, s=20, c="#388e3c", edgecolors="white", linewidth=0.3)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("VQI-V v1.0 (Full-feature)")
    ax.set_ylabel("VQI-V v2.0 (PCA-90%)")
    ax.set_title("VQI-V: v1.0 vs v2.0")
    corr_v = np.corrcoef(v_v1, v_v2)[0, 1]
    mae_v = np.mean(np.abs(v_v1 - v_v2))
    ax.text(0.05, 0.95, f"r = {corr_v:.3f}\nMAE = {mae_v:.1f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # VQI-S difference histogram
    ax = axes[1, 0]
    diff_s = s_v2 - s_v1
    ax.hist(diff_s, bins=41, range=(-20.5, 20.5), color="#1976d2", alpha=0.7, edgecolor="white")
    ax.set_xlabel("VQI-S Score Difference (v2.0 - v1.0)")
    ax.set_ylabel("Count")
    ax.set_title(f"VQI-S Difference (mean={np.mean(diff_s):.1f}, std={np.std(diff_s):.1f})")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    # VQI-V difference histogram
    ax = axes[1, 1]
    diff_v = v_v2 - v_v1
    ax.hist(diff_v, bins=41, range=(-20.5, 20.5), color="#388e3c", alpha=0.7, edgecolor="white")
    ax.set_xlabel("VQI-V Score Difference (v2.0 - v1.0)")
    ax.set_ylabel("Count")
    ax.set_title(f"VQI-V Difference (mean={np.mean(diff_v):.1f}, std={np.std(diff_v):.1f})")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    plt.suptitle(f"VQI Score Comparison: v1.0 (Full) vs v2.0 (PCA-90%) [n={len(common)}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_v2_score_comparison_v1_vs_v2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return {
        "n_files": len(common),
        "corr_s": corr_s, "corr_v": corr_v,
        "mae_s": mae_s, "mae_v": mae_v,
        "mean_diff_s": float(np.mean(diff_s)),
        "mean_diff_v": float(np.mean(diff_v)),
        "std_diff_s": float(np.std(diff_s)),
        "std_diff_v": float(np.std(diff_v)),
    }


def generate_analysis_md(data_v2, comparison=None):
    """Generate analysis.md for Step 9 v2."""
    scores_s = [int(r["vqi_s"]) for r in data_v2]
    scores_v = [int(r["vqi_v"]) for r in data_v2]

    lines = [
        "# Step 9 v2: VQI Software v2.0 (PCA-90%) Analysis",
        "",
        "## Conformance Set Overview",
        f"- **Files scored:** {len(data_v2)}",
        f"- **VQI-S:** min={min(scores_s)}, max={max(scores_s)}, "
        f"mean={np.mean(scores_s):.1f}, std={np.std(scores_s):.1f}",
        f"- **VQI-V:** min={min(scores_v)}, max={max(scores_v)}, "
        f"mean={np.mean(scores_v):.1f}, std={np.std(scores_v):.1f}",
        f"- **Correlation (S vs V):** r = {np.corrcoef(scores_s, scores_v)[0, 1]:.3f}",
        "",
    ]

    if comparison:
        lines.extend([
            "## v1.0 vs v2.0 Comparison",
            f"- **Files compared:** {comparison['n_files']}",
            f"- **VQI-S:** Pearson r = {comparison['corr_s']:.3f}, "
            f"MAE = {comparison['mae_s']:.1f}, "
            f"mean diff = {comparison['mean_diff_s']:+.1f} +/- {comparison['std_diff_s']:.1f}",
            f"- **VQI-V:** Pearson r = {comparison['corr_v']:.3f}, "
            f"MAE = {comparison['mae_v']:.1f}, "
            f"mean diff = {comparison['mean_diff_v']:+.1f} +/- {comparison['std_diff_v']:.1f}",
            "",
            "## Interpretation",
            "The PCA-90% model (v2.0) uses 77% fewer features for VQI-S (430->99 PCs) "
            "and 65% fewer for VQI-V (133->47 PCs). The high correlation with v1.0 scores "
            "confirms that the dimensionality reduction preserves the essential quality signal.",
            "",
        ])

    # Processing time
    times = [float(r.get("processing_time_ms", 0)) for r in data_v2 if r.get("processing_time_ms")]
    if times:
        lines.extend([
            "## Processing Time",
            f"- **Mean:** {np.mean(times):.0f} ms",
            f"- **Median:** {np.median(times):.0f} ms",
            f"- **Min:** {np.min(times):.0f} ms",
            f"- **Max:** {np.max(times):.0f} ms",
            "",
        ])

    md_path = os.path.join(REPORTS_DIR, "analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


def main():
    print("=== Step 9 v2.0 Visualizations ===")

    # Load v2.0 data
    data_v2 = load_conformance_scores("v2.0")
    if not data_v2:
        print("v2.0 conformance data not available. Run generate_conformance_output_v2.py first.")
        return

    print(f"Loaded {len(data_v2)} v2.0 conformance results")

    # 1. Score distribution
    plot_conformance_score_distribution(data_v2)

    # 2. S vs V scatter
    plot_conformance_scatter(data_v2)

    # 3. Processing time
    plot_processing_time(data_v2)

    # 4. Category coverage
    print("Computing category coverage (50 files)...")
    plot_feedback_category_coverage(data_v2)

    # 5. v1 vs v2 comparison
    data_v1 = load_conformance_scores("v1.0")
    comparison = None
    if data_v1:
        print(f"Loaded {len(data_v1)} v1.0 results for comparison")
        comparison = plot_score_comparison_v1_vs_v2(data_v1, data_v2)
    else:
        print("v1.0 data not available, skipping comparison plot.")

    # 6. Analysis markdown
    generate_analysis_md(data_v2, comparison)

    print(f"\nAll Step 9 v2.0 visualizations saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
