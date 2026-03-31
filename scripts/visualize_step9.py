"""Step 9 visualizations for VQI GUI application.

Generates:
  9_app_layout_diagram.png         -- Application layout mockup
  9_conformance_score_distribution.png  -- Histogram of conformance set scores
  9_conformance_scatter_s_vs_v.png      -- VQI-S vs VQI-V scatter
  9_feedback_category_coverage.png      -- Bar chart: category trigger frequency
  9_processing_time_histogram.png       -- Processing time distribution
"""

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REPORTS_DIR = os.path.join(BASE_DIR, "reports", "step9", "v1")
CONF_DIR = os.path.join(BASE_DIR, "conformance")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_conformance_scores():
    """Load conformance expected output."""
    csv_path = os.path.join(CONF_DIR, "conformance_expected_output_v1.0.csv")
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_app_layout_diagram():
    """Create a mockup diagram of the application layout."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("VQI Desktop Application - Layout", fontsize=14, fontweight="bold")

    # Window border
    ax.add_patch(plt.Rectangle((2, 2), 96, 96, fill=False, edgecolor="black", linewidth=2))
    ax.text(50, 97, "VQI - Voice Quality Index v1.0.0", ha="center", va="top",
            fontsize=12, fontweight="bold")

    # Input section
    ax.add_patch(plt.Rectangle((4, 70), 92, 24, fill=True, facecolor="#e3f2fd",
                                edgecolor="#1976d2", linewidth=1.5))
    ax.text(50, 92, "Input Tabs: [Upload File] [Record Audio]",
            ha="center", va="top", fontsize=10)
    ax.text(50, 84, "Drag-and-drop zone / Browse button",
            ha="center", va="center", fontsize=9, color="#666")
    ax.text(50, 76, "OR Microphone selector + Record/Stop",
            ha="center", va="center", fontsize=9, color="#666")

    # Progress bar
    ax.add_patch(plt.Rectangle((4, 66), 92, 3, fill=True, facecolor="#c8e6c9",
                                edgecolor="#388e3c"))
    ax.text(50, 67.5, "Progress: Step 3/4 - Extracting features...",
            ha="center", va="center", fontsize=8)

    # Score gauges
    ax.add_patch(plt.Rectangle((4, 44), 44, 20, fill=True, facecolor="#fff3e0",
                                edgecolor="#f57c00"))
    ax.text(26, 62, "Signal Quality (VQI-S)", ha="center", va="top", fontsize=9,
            fontweight="bold")
    ax.text(26, 53, "72", ha="center", va="center", fontsize=20, fontweight="bold",
            color="#388e3c")
    ax.text(26, 48, "/100 - Good", ha="center", va="center", fontsize=9, color="#666")

    ax.add_patch(plt.Rectangle((52, 44), 44, 20, fill=True, facecolor="#fff3e0",
                                edgecolor="#f57c00"))
    ax.text(74, 62, "Voice Distinctiveness (VQI-V)", ha="center", va="top", fontsize=9,
            fontweight="bold")
    ax.text(74, 53, "58", ha="center", va="center", fontsize=20, fontweight="bold",
            color="#ffc107")
    ax.text(74, 48, "/100 - Fair", ha="center", va="center", fontsize=9, color="#666")

    # Feedback tabs
    ax.add_patch(plt.Rectangle((4, 10), 92, 32, fill=True, facecolor="#f5f5f5",
                                edgecolor="#757575"))
    ax.text(50, 40, "Feedback: [Summary] [Expert Details] [Waveform/Spectrogram]",
            ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(50, 33, "What's Good: Clean recording, good signal quality.",
            ha="center", va="center", fontsize=8, style="italic")
    ax.text(50, 28, "What to Improve: Low prosodic uniqueness...",
            ha="center", va="center", fontsize=8, style="italic")
    ax.text(50, 22, "Fix: Use natural, expressive speech...",
            ha="center", va="center", fontsize=8, color="#1976d2")

    # Buttons
    ax.text(35, 12, "[Export Report]", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e0e0"))
    ax.text(65, 12, "[Score Another File]", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e0e0"))

    # Status bar
    ax.add_patch(plt.Rectangle((2, 2), 96, 5, fill=True, facecolor="#eeeeee",
                                edgecolor="#bdbdbd"))
    ax.text(50, 4.5, "VQI-S: 72 | VQI-V: 58 | 5230ms | Audio: 8.4s",
            ha="center", va="center", fontsize=8, color="#666")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_app_layout_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_conformance_score_distribution(data):
    """Histogram of conformance set VQI-S and VQI-V scores."""
    scores_s = [int(r["vqi_s"]) for r in data]
    scores_v = [int(r["vqi_v"]) for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(scores_s, bins=20, range=(0, 100), color="#1976d2", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("VQI-S Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"VQI-S Distribution (n={len(scores_s)})")
    axes[0].axvline(np.mean(scores_s), color="red", linestyle="--",
                    label=f"Mean: {np.mean(scores_s):.1f}")
    axes[0].legend()

    axes[1].hist(scores_v, bins=20, range=(0, 100), color="#388e3c", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("VQI-V Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"VQI-V Distribution (n={len(scores_v)})")
    axes[1].axvline(np.mean(scores_v), color="red", linestyle="--",
                    label=f"Mean: {np.mean(scores_v):.1f}")
    axes[1].legend()

    plt.suptitle("Conformance Set Score Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_conformance_score_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_conformance_scatter(data):
    """Scatter plot VQI-S vs VQI-V for conformance set."""
    scores_s = [int(r["vqi_s"]) for r in data]
    scores_v = [int(r["vqi_v"]) for r in data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(scores_s, scores_v, alpha=0.6, s=30, c="#1976d2", edgecolors="white",
               linewidth=0.3)
    ax.set_xlabel("VQI-S (Signal Quality)")
    ax.set_ylabel("VQI-V (Voice Distinctiveness)")
    ax.set_title("Conformance Set: VQI-S vs VQI-V")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y=x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation
    corr = np.corrcoef(scores_s, scores_v)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_conformance_scatter_s_vs_v.png")
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

    # Sample 50 files for speed
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
        axes[0].set_title("VQI-S Limiting Factor Categories")

    if cat_counts_v:
        cats = sorted(cat_counts_v.keys())
        counts = [cat_counts_v[c] for c in cats]
        axes[1].barh(cats, counts, color="#388e3c", alpha=0.8)
        axes[1].set_xlabel("Number of files")
        axes[1].set_title("VQI-V Limiting Factor Categories")

    plt.suptitle(f"Feedback Category Coverage (n={len(sample)} files)", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_feedback_category_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_processing_time(data):
    """Processing time histogram from conformance scoring."""
    from vqi.engine import VQIEngine
    import logging
    logging.basicConfig(level=logging.WARNING)

    files_dir = os.path.join(CONF_DIR, "test_files")
    engine = VQIEngine()

    times_ms = []
    sample = data[:50]
    for i, row in enumerate(sample):
        filepath = os.path.join(files_dir, row["filename"])
        if not os.path.exists(filepath):
            continue
        try:
            result = engine.score_file(filepath)
            times_ms.append(result.processing_time_ms)
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  Timing: {i+1}/{len(sample)} files")

    if not times_ms:
        print("No timing data collected.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(times_ms, bins=25, color="#ff7043", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Processing Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Processing Time Distribution (n={len(times_ms)} files)")
    ax.axvline(np.mean(times_ms), color="red", linestyle="--",
               label=f"Mean: {np.mean(times_ms):.0f}ms")
    ax.axvline(np.median(times_ms), color="blue", linestyle="--",
               label=f"Median: {np.median(times_ms):.0f}ms")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "9_processing_time_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    print("=== Step 9 Visualizations ===")

    # 1. App layout diagram (always possible)
    plot_app_layout_diagram()

    # Load conformance data
    data = load_conformance_scores()
    if not data:
        print("Conformance data not yet available. Run generate_conformance_output.py first.")
        print("Skipping conformance-dependent plots.")
        return

    print(f"Loaded {len(data)} conformance results")

    # 2. Score distribution
    plot_conformance_score_distribution(data)

    # 3. S vs V scatter
    plot_conformance_scatter(data)

    # 4. Category coverage (requires re-scoring ~50 files)
    print("Computing category coverage (50 files)...")
    plot_feedback_category_coverage(data)

    # 5. Processing time (reuse engine from category coverage)
    print("Computing processing times (50 files)...")
    plot_processing_time(data)

    print(f"\nAll Step 9 visualizations saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
