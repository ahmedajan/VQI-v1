#!/usr/bin/env python3
"""
Step 2.1 Visualization — Speech Duration after VAD

Generates sub-step completion visualizations:
  1. Speech duration histogram (all samples)
  2. Speech duration by dataset source (violin/box plot)
  3. Speech ratio distribution (VAD speech / total)
  4. Class eligibility preview (>= 3.0s, < 1.5s, ambiguous 1.5-3.0s)
  5. Duration vs dataset strip plot
  6. Summary statistics table per dataset
  7. Total vs speech duration scatter
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- paths ----
INPUT_CSV = os.path.join("D:", os.sep, "VQI", "implementation", "data", "step2", "labels", "train_pool_durations.csv")
REPORT_DIR = os.path.join("D:", os.sep, "VQI", "implementation", "reports", "step2")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")
    # Filter out failed files (duration == -1)
    n_failed = (df["total_duration_sec"] < 0).sum()
    if n_failed > 0:
        print(f"Warning: {n_failed} failed files (duration=-1), excluding from plots")
        df = df[df["total_duration_sec"] >= 0].copy()
    return df


def plot_speech_duration_histogram(df):
    """Plot 1: Overall speech duration histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full range
    ax = axes[0]
    ax.hist(df["speech_duration_sec"], bins=200, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(3.0, color="green", linestyle="--", linewidth=1.5, label=">= 3.0s (Class 1 eligible)")
    ax.axvline(1.5, color="red", linestyle="--", linewidth=1.5, label="< 1.5s (Class 0 forced)")
    ax.set_xlabel("Speech Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Speech Duration Distribution (Full Range)")
    ax.legend(fontsize=8)

    # Zoomed 0-10s
    ax = axes[1]
    mask = df["speech_duration_sec"] <= 10
    ax.hist(df.loc[mask, "speech_duration_sec"], bins=200, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(3.0, color="green", linestyle="--", linewidth=1.5, label=">= 3.0s")
    ax.axvline(1.5, color="red", linestyle="--", linewidth=1.5, label="< 1.5s")
    ax.set_xlabel("Speech Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Speech Duration Distribution (0-10s Zoom)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_speech_duration_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_duration_by_dataset(df):
    """Plot 2: Speech duration by dataset source (box plot)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = sorted(df["dataset_source"].unique())
    data_list = [df.loc[df["dataset_source"] == ds, "speech_duration_sec"].values for ds in datasets]

    bp = ax.boxplot(data_list, labels=datasets, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.axhline(3.0, color="green", linestyle="--", linewidth=1, alpha=0.7, label=">= 3.0s threshold")
    ax.axhline(1.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="< 1.5s threshold")
    ax.set_ylabel("Speech Duration (seconds)")
    ax.set_title("Speech Duration by Dataset Source")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_duration_by_dataset.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_speech_ratio_distribution(df):
    """Plot 3: Speech ratio (speech / total) distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["speech_ratio"], bins=100, color="darkorange", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Speech Ratio (speech duration / total duration)")
    ax.set_ylabel("Count")
    ax.set_title("VAD Speech Ratio Distribution")

    mean_ratio = df["speech_ratio"].mean()
    median_ratio = df["speech_ratio"].median()
    ax.axvline(mean_ratio, color="red", linestyle="-", linewidth=1.5, label=f"Mean: {mean_ratio:.3f}")
    ax.axvline(median_ratio, color="blue", linestyle="--", linewidth=1.5, label=f"Median: {median_ratio:.3f}")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_speech_ratio_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_class_eligibility(df):
    """Plot 4: Class eligibility pie chart + bar chart."""
    dur = df["speech_duration_sec"]
    n_class1 = (dur >= 3.0).sum()
    n_class0 = (dur < 1.5).sum()
    n_ambig = ((dur >= 1.5) & (dur < 3.0)).sum()
    total = len(dur)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    ax = axes[0]
    sizes = [n_class1, n_ambig, n_class0]
    labels = [
        f">= 3.0s\n(Class 1 eligible)\n{n_class1:,} ({100*n_class1/total:.1f}%)",
        f"1.5-3.0s\n(ambiguous)\n{n_ambig:,} ({100*n_ambig/total:.1f}%)",
        f"< 1.5s\n(Class 0 forced)\n{n_class0:,} ({100*n_class0/total:.1f}%)",
    ]
    colors_pie = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.pie(sizes, labels=labels, colors=colors_pie, startangle=90, textprops={"fontsize": 9})
    ax.set_title("Duration-Based Class Eligibility")

    # Bar chart
    ax = axes[1]
    categories = [">= 3.0s\n(Class 1)", "1.5-3.0s\n(Ambiguous)", "< 1.5s\n(Class 0)"]
    counts = [n_class1, n_ambig, n_class0]
    bars = ax.bar(categories, counts, color=colors_pie, edgecolor="gray")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                f"{count:,}\n({100*count/total:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Duration-Based Class Eligibility Counts")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_class_eligibility.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_total_vs_speech_duration(df):
    """Plot 5: Total duration vs speech duration scatter (subsampled)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample for plotting efficiency
    n_plot = min(50000, len(df))
    idx = np.random.RandomState(42).choice(len(df), n_plot, replace=False)
    sub = df.iloc[idx]

    ax.scatter(sub["total_duration_sec"], sub["speech_duration_sec"],
               alpha=0.1, s=1, c="steelblue")
    max_val = max(sub["total_duration_sec"].max(), sub["speech_duration_sec"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Total Duration (seconds)")
    ax.set_ylabel("Speech Duration after VAD (seconds)")
    ax.set_title(f"Total vs Speech Duration ({n_plot:,} samples)")
    ax.legend()
    ax.set_xlim(0, min(max_val, 30))
    ax.set_ylim(0, min(max_val, 30))

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_total_vs_speech_duration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_duration_cdf(df):
    """Plot 6: Cumulative distribution of speech duration."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sorted_dur = np.sort(df["speech_duration_sec"].values)
    cdf = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)

    ax.plot(sorted_dur, cdf, color="steelblue", linewidth=1.5)
    ax.axvline(1.5, color="red", linestyle="--", alpha=0.7, label="1.5s threshold")
    ax.axvline(3.0, color="green", linestyle="--", alpha=0.7, label="3.0s threshold")

    # Mark key percentiles
    for pct in [1, 5, 10, 50, 90, 95, 99]:
        val = np.percentile(sorted_dur, pct)
        ax.axhline(pct/100, color="gray", linestyle=":", alpha=0.3)
        ax.text(val + 0.1, pct/100, f"P{pct}={val:.1f}s", fontsize=7, va="center")

    ax.set_xlabel("Speech Duration (seconds)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of Speech Duration after VAD")
    ax.set_xlim(0, 20)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_1_duration_cdf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def generate_summary_table(df):
    """Generate and save per-dataset summary statistics."""
    datasets = sorted(df["dataset_source"].unique())
    rows = []

    for ds in datasets:
        sub = df[df["dataset_source"] == ds]
        dur = sub["speech_duration_sec"]
        ratio = sub["speech_ratio"]
        rows.append({
            "dataset": ds,
            "n_samples": len(sub),
            "dur_mean": round(dur.mean(), 3),
            "dur_median": round(dur.median(), 3),
            "dur_std": round(dur.std(), 3),
            "dur_p5": round(dur.quantile(0.05), 3),
            "dur_p95": round(dur.quantile(0.95), 3),
            "dur_min": round(dur.min(), 3),
            "dur_max": round(dur.max(), 3),
            "ratio_mean": round(ratio.mean(), 4),
            "ratio_median": round(ratio.median(), 4),
            "n_ge_3s": int((dur >= 3.0).sum()),
            "pct_ge_3s": round(100 * (dur >= 3.0).mean(), 1),
            "n_lt_1_5s": int((dur < 1.5).sum()),
            "pct_lt_1_5s": round(100 * (dur < 1.5).mean(), 1),
        })

    # Add total row
    dur = df["speech_duration_sec"]
    ratio = df["speech_ratio"]
    rows.append({
        "dataset": "TOTAL",
        "n_samples": len(df),
        "dur_mean": round(dur.mean(), 3),
        "dur_median": round(dur.median(), 3),
        "dur_std": round(dur.std(), 3),
        "dur_p5": round(dur.quantile(0.05), 3),
        "dur_p95": round(dur.quantile(0.95), 3),
        "dur_min": round(dur.min(), 3),
        "dur_max": round(dur.max(), 3),
        "ratio_mean": round(ratio.mean(), 4),
        "ratio_median": round(ratio.median(), 4),
        "n_ge_3s": int((dur >= 3.0).sum()),
        "pct_ge_3s": round(100 * (dur >= 3.0).mean(), 1),
        "n_lt_1_5s": int((dur < 1.5).sum()),
        "pct_lt_1_5s": round(100 * (dur < 1.5).mean(), 1),
    })

    summary_df = pd.DataFrame(rows)
    path = os.path.join(REPORT_DIR, "2_1_summary_statistics.csv")
    summary_df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved: {path}")

    # Print summary
    print("\n--- Summary Statistics ---")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    print("=" * 60)
    print("Step 2.1 Visualization: Speech Duration after VAD")
    print("=" * 60)

    df = load_data()

    # Basic stats
    print(f"\nTotal samples: {len(df):,}")
    print(f"Datasets: {sorted(df['dataset_source'].unique())}")

    # Generate all plots
    plot_speech_duration_histogram(df)
    plot_duration_by_dataset(df)
    plot_speech_ratio_distribution(df)
    plot_class_eligibility(df)
    plot_total_vs_speech_duration(df)
    plot_duration_cdf(df)
    summary_df = generate_summary_table(df)

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
