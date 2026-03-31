"""
Step 2.2-2.6 Visualizations.

Generates all plots for the label definition phase:
  2.2  Threshold distributions (genuine + impostor with threshold lines)
  2.3  Binary label statistics (counts, score distributions, duration, dataset)
  2.4  Balanced set (before/after, pie chart)
  2.5  Fisher Ratio distributions and correlations
  2.6  Validation set composition and overlap check

Output: implementation/reports/step2/{2.2_thresholds, 2.3_labels, 2.4_balanced,
        2.5_dprime, 2.6_validation}/
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCORE_DIR = PROJECT_ROOT / "data" / "step1" / "provider_scores"
DATA_DIR = PROJECT_ROOT / "data" / "step2"
REPORT_DIR = PROJECT_ROOT / "reports" / "step2"

THRESHOLDS_YAML = DATA_DIR / "label_thresholds.yaml"
TRAINING_LABELS_CSV = DATA_DIR / "training_labels.csv"
TRAINING_SET_FINAL_CSV = DATA_DIR / "training_set_final.csv"
FISHER_VALUES_CSV = DATA_DIR / "fisher_values.csv"
VALIDATION_SET_CSV = DATA_DIR / "validation_set.csv"
STATS_JSON = SCORE_DIR / "score_statistics.json"

PROVIDER_INFO = [
    ("P1", "P1_ECAPA", "scores_P1_ECAPA_ecapa.csv", "impostor_norm_P1_ECAPA.npy"),
    ("P2", "P2_RESNET", "scores_P2_RESNET_resnet.csv", "impostor_norm_P2_RESNET.npy"),
    ("P3", "P3_ECAPA2", "scores_P3_ECAPA2_ecapa2.csv", "impostor_norm_P3_ECAPA2.npy"),
]


def viz_2_2_thresholds():
    """2.2: Genuine and impostor distributions with threshold lines."""
    out_dir = REPORT_DIR / "2.2_thresholds"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(THRESHOLDS_YAML, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)

    # Plot genuine distributions with 90th percentile lines
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (short, full, csv_name, _) in enumerate(PROVIDER_INFO):
        ax = axes[idx]
        df = pd.read_csv(SCORE_DIR / csv_name, usecols=["genuine_norm"])
        vals = df["genuine_norm"].values

        ax.hist(vals, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        p90 = thresholds[short]["percentile_90"]
        fmr = thresholds[short]["fmr_001"]
        ax.axvline(p90, color="red", linewidth=2, linestyle="--", label=f"P90={p90:.2f}")
        ax.axvline(fmr, color="orange", linewidth=2, linestyle=":", label=f"FMR={fmr:.2f}")
        ax.set_title(f"{short} Genuine Scores (S-norm)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "genuine_distributions_with_thresholds.png", dpi=150)
    plt.close()

    # Plot impostor distributions with FMR=0.001 lines
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (short, full, _, npy_name) in enumerate(PROVIDER_INFO):
        ax = axes[idx]
        vals = np.load(SCORE_DIR / npy_name)

        # Subsample for plotting (28M is too many points)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(vals), min(500_000, len(vals)), replace=False)
        sample_vals = vals[sample_idx]

        ax.hist(sample_vals, bins=300, density=True, alpha=0.7, color="salmon", edgecolor="none")
        fmr = thresholds[short]["fmr_001"]
        ax.axvline(fmr, color="red", linewidth=2, linestyle="--", label=f"FMR=0.001={fmr:.2f}")
        ax.set_title(f"{short} Impostor Scores (S-norm)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_xlim(-5, max(fmr + 2, 5))

    plt.tight_layout()
    plt.savefig(out_dir / "impostor_distributions_with_fmr.png", dpi=150)
    plt.close()

    # Threshold summary table as a figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table_data = [["Provider", "90th Percentile", "FMR=0.001", "Gap"]]
    for short in ["P1", "P2", "P3"]:
        p90 = thresholds[short]["percentile_90"]
        fmr = thresholds[short]["fmr_001"]
        table_data.append([short, f"{p90:.4f}", f"{fmr:.4f}", f"{p90-fmr:.4f}"])

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    # Style header row
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Step 2.2: Provider Score Thresholds", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_summary_table.png", dpi=150)
    plt.close()

    logger.info(f"2.2 plots saved to {out_dir}")


def viz_2_3_labels():
    """2.3: Label statistics and distributions."""
    out_dir = REPORT_DIR / "2.3_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(TRAINING_LABELS_CSV)

    # We also need total count (including excluded)
    total_pool = 1_210_451  # known from Step 2.1

    n_class0 = (labels_df["label"] == 0).sum()
    n_class1 = (labels_df["label"] == 1).sum()
    n_excluded = total_pool - n_class0 - n_class1

    # 1. Label count bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Class 0\n(Low-performing)", "Class 1\n(High-performing)", "Excluded"]
    counts = [n_class0, n_class1, n_excluded]
    colors = ["#E74C3C", "#2ECC71", "#BDC3C7"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = 100 * count / total_pool
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_pool * 0.01,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of Samples")
    ax.set_title("Step 2.3: Binary Label Assignment", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(out_dir / "label_counts.png", dpi=150)
    plt.close()

    # 2. Score distributions by label class
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, short in enumerate(["P1", "P2", "P3"]):
        ax = axes[idx]
        col = f"score_{short}"
        c0 = labels_df[labels_df["label"] == 0][col].values
        c1 = labels_df[labels_df["label"] == 1][col].values

        ax.hist(c0, bins=100, density=True, alpha=0.6, color="#E74C3C", label=f"Class 0 (n={len(c0):,})")
        ax.hist(c1, bins=100, density=True, alpha=0.6, color="#2ECC71", label=f"Class 1 (n={len(c1):,})")
        ax.set_title(f"{short} Score Distribution by Label")
        ax.set_xlabel("Genuine Norm Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "score_distributions_by_label.png", dpi=150)
    plt.close()

    # 3. Duration distribution by label class
    fig, ax = plt.subplots(figsize=(10, 5))
    c0 = labels_df[labels_df["label"] == 0]["speech_duration"].values
    c1 = labels_df[labels_df["label"] == 1]["speech_duration"].values

    ax.hist(c0, bins=100, density=True, alpha=0.6, color="#E74C3C", label=f"Class 0 (n={len(c0):,})")
    ax.hist(c1, bins=100, density=True, alpha=0.6, color="#2ECC71", label=f"Class 1 (n={len(c1):,})")
    ax.axvline(3.0, color="blue", linewidth=1.5, linestyle="--", label="3.0s threshold")
    ax.axvline(1.5, color="red", linewidth=1.5, linestyle="--", label="1.5s threshold")
    ax.set_xlabel("Speech Duration (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Duration Distribution by Label Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "duration_by_label.png", dpi=150)
    plt.close()

    # 4. Dataset composition by label class
    comp = labels_df.groupby(["dataset_source", "label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    comp.plot(kind="barh", stacked=False, ax=ax, color=["#E74C3C", "#2ECC71"])
    ax.set_xlabel("Number of Samples")
    ax.set_title("Dataset Composition by Label Class")
    ax.legend(["Class 0", "Class 1"], title="Label")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset_composition_by_label.png", dpi=150)
    plt.close()

    logger.info(f"2.3 plots saved to {out_dir}")


def viz_2_4_balanced():
    """2.4: Before/after balancing comparison."""
    out_dir = REPORT_DIR / "2.4_balanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(TRAINING_LABELS_CSV)
    balanced_df = pd.read_csv(TRAINING_SET_FINAL_CSV)

    before_counts = labels_df["label"].value_counts().sort_index()
    after_counts = balanced_df["label"].value_counts().sort_index()

    # 1. Before/after comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Before
    bars1 = ax1.bar(["Class 0", "Class 1"], [before_counts.get(0, 0), before_counts.get(1, 0)],
                    color=["#E74C3C", "#2ECC71"], edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars1, [before_counts.get(0, 0), before_counts.get(1, 0)]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"{count:,}", ha="center", va="bottom", fontsize=11)
    ax1.set_title("Before Balancing", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count")

    # After
    bars2 = ax2.bar(["Class 0", "Class 1"], [after_counts.get(0, 0), after_counts.get(1, 0)],
                    color=["#E74C3C", "#2ECC71"], edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars2, [after_counts.get(0, 0), after_counts.get(1, 0)]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"{count:,}", ha="center", va="bottom", fontsize=11)
    ax2.set_title("After Balancing (1:1)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.set_ylim(ax1.get_ylim())  # same scale

    plt.suptitle("Step 2.4: Training Set Balancing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "before_after_balancing.png", dpi=150)
    plt.close()

    # 2. Class balance pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(after_counts.values, labels=["Class 0", "Class 1"],
           colors=["#E74C3C", "#2ECC71"], autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title(f"Balanced Training Set\n(n={len(balanced_df):,})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "balance_pie.png", dpi=150)
    plt.close()

    logger.info(f"2.4 plots saved to {out_dir}")


def viz_2_5_fisher():
    """2.5: Fisher Ratio distributions and correlations."""
    out_dir = REPORT_DIR / "2.5_fisher"
    out_dir.mkdir(parents=True, exist_ok=True)

    fisher_df = pd.read_csv(FISHER_VALUES_CSV)

    # Per-speaker Fisher Ratio (deduplicate -- same value for all samples of a speaker)
    speaker_fr = fisher_df.groupby("speaker_id")[
        ["fisher_P1", "fisher_P2", "fisher_P3", "fisher_mean"]
    ].first()

    # 1. Fisher Ratio distribution histograms per provider
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, col in enumerate(["fisher_P1", "fisher_P2", "fisher_P3"]):
        ax = axes[idx]
        vals = speaker_fr[col].values
        ax.hist(vals, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(np.mean(vals), color="red", linewidth=1.5, linestyle="--",
                   label=f"Mean={np.mean(vals):.2f}")
        ax.axvline(np.median(vals), color="green", linewidth=1.5, linestyle=":",
                   label=f"Median={np.median(vals):.2f}")
        short = col.split("_")[1]
        ax.set_title(f"Fisher Ratio Distribution ({short})")
        ax.set_xlabel("Fisher Ratio (d')")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("Step 2.5: Per-Speaker Fisher Ratio (for Feature Evaluation)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fisher_distributions.png", dpi=150)
    plt.close()

    # 2. Fisher Ratio correlation matrix across providers
    corr = speaker_fr[["fisher_P1", "fisher_P2", "fisher_P3"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=0, vmax=1)

    labels = ["P1 (ECAPA)", "P2 (ResNet)", "P3 (ECAPA2)"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title("Fisher Ratio Correlation Across Providers")
    plt.tight_layout()
    plt.savefig(out_dir / "fisher_correlation_matrix.png", dpi=150)
    plt.close()

    # 3. Fisher Ratio vs mean genuine score scatter
    # Load genuine scores to compute per-speaker mean
    score_csv = SCORE_DIR / "scores_P1_ECAPA_ecapa.csv"
    scores = pd.read_csv(score_csv, usecols=["speaker_id", "genuine_norm"])
    spk_mean_score = scores.groupby("speaker_id")["genuine_norm"].mean()

    merged = speaker_fr.join(spk_mean_score.rename("mean_genuine_P1"))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["mean_genuine_P1"], merged["fisher_P1"], alpha=0.3, s=8, color="steelblue")
    ax.set_xlabel("Mean Genuine Score (P1, S-norm)")
    ax.set_ylabel("Fisher Ratio (P1)")
    ax.set_title("Fisher Ratio vs Mean Genuine Score (Per Speaker)")

    # Correlation annotation
    valid = merged[["mean_genuine_P1", "fisher_P1"]].dropna()
    r = np.corrcoef(valid["mean_genuine_P1"], valid["fisher_P1"])[0, 1]
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fisher_vs_genuine_score.png", dpi=150)
    plt.close()

    # 4. Fisher Ratio mean distribution (across providers)
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = speaker_fr["fisher_mean"].values
    ax.hist(vals, bins=80, density=True, alpha=0.7, color="darkorange", edgecolor="none")
    ax.axvline(np.mean(vals), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean={np.mean(vals):.2f}")
    ax.set_xlabel("Mean Fisher Ratio (across P1-P3)")
    ax.set_ylabel("Density")
    ax.set_title("Mean Fisher Ratio Distribution (Per Speaker)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fisher_mean_distribution.png", dpi=150)
    plt.close()

    logger.info(f"2.5 plots saved to {out_dir}")


def viz_2_6_validation():
    """2.6: Validation set composition and overlap check."""
    out_dir = REPORT_DIR / "2.6_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(VALIDATION_SET_CSV)
    train_df = pd.read_csv(TRAINING_SET_FINAL_CSV)

    # 1. Overlap verification
    val_files = set(val_df["filename"])
    train_files = set(train_df["filename"])
    overlap = val_files & train_files

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Training Set", "Validation Set", "Overlap"],
           [len(train_files), len(val_files), len(overlap)],
           color=["#2ECC71", "#3498DB", "#E74C3C" if len(overlap) > 0 else "#BDC3C7"],
           edgecolor="black", linewidth=0.5)
    for i, (lbl, cnt) in enumerate(zip(["Training", "Validation", "Overlap"],
                                        [len(train_files), len(val_files), len(overlap)])):
        ax.text(i, cnt + max(len(train_files), len(val_files)) * 0.02,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Training/Validation Overlap Check", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(out_dir / "overlap_check.png", dpi=150)
    plt.close()

    # 2. Validation set composition by dataset
    comp = val_df["dataset_source"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(comp.index, comp.values, color="steelblue", edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, comp.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", ha="left", va="center", fontsize=10)
    ax.set_xlabel("Number of Samples")
    ax.set_title(f"Validation Set Composition (n={len(val_df):,})")
    plt.tight_layout()
    plt.savefig(out_dir / "validation_composition.png", dpi=150)
    plt.close()

    logger.info(f"2.6 plots saved to {out_dir}")


def viz_m2_selection_funnel():
    """M.2#2: Label selection funnel diagram."""
    total_pool = 1_210_451
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)
    n_c1 = (labels_df["label"] == 1).sum()
    n_c0 = (labels_df["label"] == 0).sum()
    n_labeled = n_c0 + n_c1
    n_dur_fail = 24  # only 24 fail duration < 1.5s

    stages = [
        ("Total Pool", total_pool, "#3498DB"),
        ("Pass Duration (>= 1.5s)", total_pool - n_dur_fail, "#2980B9"),
        ("Class 1 (All 3 P >= 90th pctl)", n_c1, "#27AE60"),
        ("Class 0 (All 3 P < FMR=0.001)", n_c0, "#E74C3C"),
        ("Total Labeled", n_labeled, "#F39C12"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_positions = np.arange(len(stages))[::-1]

    for i, (label, count, color) in enumerate(stages):
        y = y_positions[i]
        w = max(count / total_pool, 0.06)
        left = (1.0 - w) / 2
        ax.barh(y, w, height=0.65, left=left, color=color,
                edgecolor="black", linewidth=0.5, alpha=0.85)
        pct = 100.0 * count / total_pool
        ax.text(0.5, y, f"{label}\n{count:,} ({pct:.2f}%)",
                ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, len(stages) - 0.3)
    ax.axis("off")
    ax.set_title("Step 2: Label Selection Funnel", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_selection_funnel.png", dpi=150)
    plt.close()
    logger.info("Saved label_selection_funnel.png")


def viz_m2_quality_scatter():
    """M.2#7: Scatter plots of duration vs scores, colored by label."""
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {0: "#E74C3C", 1: "#2ECC71"}

    for idx, short in enumerate(["P1", "P2", "P3"]):
        ax = axes[0, idx]
        for lbl in [0, 1]:
            m = labels_df["label"] == lbl
            ax.scatter(labels_df.loc[m, "speech_duration"],
                       labels_df.loc[m, f"score_{short}"],
                       c=colors[lbl], alpha=0.1, s=2,
                       label=f"Class {lbl}", rasterized=True)
        ax.set_xlabel("Speech Duration (s)")
        ax.set_ylabel(f"{short} Score")
        ax.set_title(f"Duration vs {short}")
        ax.legend(fontsize=8, markerscale=5)

    pairs = [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]
    for idx, (pa, pb) in enumerate(pairs):
        ax = axes[1, idx]
        for lbl in [0, 1]:
            m = labels_df["label"] == lbl
            ax.scatter(labels_df.loc[m, f"score_{pa}"],
                       labels_df.loc[m, f"score_{pb}"],
                       c=colors[lbl], alpha=0.1, s=2,
                       label=f"Class {lbl}", rasterized=True)
        ax.set_xlabel(f"{pa} Score")
        ax.set_ylabel(f"{pb} Score")
        ax.set_title(f"{pa} vs {pb}")
        ax.legend(fontsize=8, markerscale=5)

    plt.suptitle("Label Quality Scatter Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_quality_scatter.png", dpi=150)
    plt.close()
    logger.info("Saved label_quality_scatter.png")


def viz_m2_ambiguous_zone():
    """M.2#8: Score distribution for excluded (ambiguous) speakers."""
    with open(THRESHOLDS_YAML, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)

    spk_scores = {}
    for short, _, csv_name, _ in PROVIDER_INFO:
        df = pd.read_csv(SCORE_DIR / csv_name,
                         usecols=["speaker_id", "genuine_norm"])
        spk_scores[short] = df.groupby("speaker_id")["genuine_norm"].first()
    merged = pd.DataFrame(spk_scores)

    is_c1 = np.ones(len(merged), dtype=bool)
    is_c0 = np.ones(len(merged), dtype=bool)
    for short in ["P1", "P2", "P3"]:
        is_c1 &= merged[short] >= thresholds[short]["percentile_90"]
        is_c0 &= merged[short] < thresholds[short]["fmr_001"]
    excluded = ~is_c1 & ~is_c0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, short in enumerate(["P1", "P2", "P3"]):
        ax = axes[idx]
        for mask, label, color, alpha in [
            (excluded, f"Excluded ({excluded.sum():,})", "#95A5A6", 0.6),
            (is_c0, f"Class 0 ({is_c0.sum():,})", "#E74C3C", 0.5),
            (is_c1, f"Class 1 ({is_c1.sum():,})", "#2ECC71", 0.5),
        ]:
            vals = merged.loc[mask, short].values
            if len(vals) > 0:
                ax.hist(vals, bins=60, density=True, alpha=alpha,
                        color=color, label=label)
        ax.axvline(thresholds[short]["percentile_90"], color="green",
                   ls="--", lw=1.5, label="90th pctl")
        ax.axvline(thresholds[short]["fmr_001"], color="red",
                   ls="--", lw=1.5, label="FMR=0.001")
        ax.set_title(f"{short} Scores")
        ax.set_xlabel("Genuine Norm Score")
        ax.legend(fontsize=7)

    plt.suptitle("Ambiguous Zone Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ambiguous_zone_analysis.png", dpi=150)
    plt.close()
    logger.info("Saved ambiguous_zone_analysis.png")


def viz_m2_score_kde():
    """M.2#9: KDE density overlay per provider, Class 0 vs Class 1."""
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, short in enumerate(["P1", "P2", "P3"]):
        ax = axes[idx]
        c0 = labels_df.loc[labels_df["label"] == 0, f"score_{short}"].values
        c1 = labels_df.loc[labels_df["label"] == 1, f"score_{short}"].values

        lo = min(c0.min(), c1.min()) - 1
        hi = max(c0.max(), c1.max()) + 1
        x = np.linspace(lo, hi, 500)

        k0, k1 = gaussian_kde(c0), gaussian_kde(c1)
        ax.fill_between(x, k0(x), alpha=0.4, color="#E74C3C", label="Class 0")
        ax.fill_between(x, k1(x), alpha=0.4, color="#2ECC71", label="Class 1")
        ax.plot(x, k0(x), color="#C0392B", lw=1.5)
        ax.plot(x, k1(x), color="#27AE60", lw=1.5)
        ax.set_title(f"{short} Score KDE")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle("Score KDE by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_score_kde_by_class.png", dpi=150)
    plt.close()
    logger.info("Saved label_score_kde_by_class.png")


def viz_m2_waterfall():
    """M.2#10: Waterfall chart of sample attrition."""
    total = 1_210_451
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)
    n_c1 = (labels_df["label"] == 1).sum()
    n_c0 = (labels_df["label"] == 0).sum()
    n_excl = total - n_c0 - n_c1
    n_labeled = n_c0 + n_c1

    stages = ["Total Pool", "- Excluded", "= Labeled", "- Class 1", "- Class 0"]
    heights = [total, n_excl, n_labeled, n_c1, n_c0]
    bottoms = [0, n_labeled, 0, n_c0, 0]
    colors = ["#3498DB", "#BDC3C7", "#F39C12", "#2ECC71", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(stages, heights, bottom=bottoms, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.6)

    for i in range(len(stages) - 1):
        top = bottoms[i] + heights[i]
        next_top = bottoms[i + 1] + heights[i + 1]
        ax.plot([i + 0.3, i + 0.7], [top, next_top],
                color="gray", ls="--", lw=0.8)

    for i, (b, h) in enumerate(zip(bottoms, heights)):
        pct = 100 * h / total
        ax.text(i, b + h / 2, f"{h:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_ylabel("Number of Samples")
    ax.set_title("Step 2: Label Assignment Waterfall", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_waterfall_funnel.png", dpi=150)
    plt.close()
    logger.info("Saved label_waterfall_funnel.png")


def viz_m2_strip_swarm():
    """M.2#11: Strip plots of label-driving features colored by class."""
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)
    rng = np.random.RandomState(42)

    c0 = labels_df[labels_df["label"] == 0].sample(n=500, random_state=42)
    c1 = labels_df[labels_df["label"] == 1].sample(n=500, random_state=42)
    sample = pd.concat([c0, c1])

    features = ["speech_duration", "score_P1", "score_P2", "score_P3"]
    names = ["Duration (s)", "P1 Score", "P2 Score", "P3 Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for idx, (feat, name) in enumerate(zip(features, names)):
        ax = axes[idx]
        for lbl, color, xb in [(0, "#E74C3C", 0), (1, "#2ECC71", 1)]:
            vals = sample.loc[sample["label"] == lbl, feat].values
            jitter = rng.uniform(-0.2, 0.2, len(vals))
            ax.scatter(xb + jitter, vals, c=color, alpha=0.4, s=8,
                       edgecolors="none")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Class 0", "Class 1"])
        ax.set_ylabel(name)
        ax.set_title(name)

    plt.suptitle("Label Quality Strip Plots (n=500/class)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_quality_strip_swarm.png", dpi=150)
    plt.close()
    logger.info("Saved label_quality_strip_swarm.png")


def viz_m2_parallel_coords():
    """M.2#12: Parallel coordinates colored by class."""
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)

    c0 = labels_df[labels_df["label"] == 0].sample(n=250, random_state=42)
    c1 = labels_df[labels_df["label"] == 1].sample(n=250, random_state=42)
    sample = pd.concat([c0, c1])

    features = ["speech_duration", "score_P1", "score_P2", "score_P3"]
    names = ["Duration", "P1 Score", "P2 Score", "P3 Score"]

    norm = sample[features].copy()
    for col in features:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn) if mx > mn else 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(features))

    for _, row in sample.iterrows():
        color = "#E74C3C" if row["label"] == 0 else "#2ECC71"
        ax.plot(x, [norm.loc[row.name, f] for f in features],
                color=color, alpha=0.08, lw=0.5)

    ax.plot([], [], color="#E74C3C", lw=2, label="Class 0")
    ax.plot([], [], color="#2ECC71", lw=2, label="Class 1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Normalized [0, 1]")
    ax.set_title("Parallel Coordinates (n=250/class)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_parallel_coordinates.png", dpi=150)
    plt.close()
    logger.info("Saved label_parallel_coordinates.png")


def viz_m2_sankey():
    """M.2#13: Sankey flow diagram of label assignment."""
    from matplotlib.sankey import Sankey

    labels_df = pd.read_csv(TRAINING_LABELS_CSV)
    total = 1_210_451
    n_c0 = (labels_df["label"] == 0).sum()
    n_c1 = (labels_df["label"] == 1).sum()
    n_excl = total - n_c0 - n_c1

    fig, ax = plt.subplots(figsize=(12, 7))

    sankey = Sankey(ax=ax, scale=0.8 / total, unit="",
                    gap=0.25, shoulder=0.05)
    sankey.add(
        flows=[total, -n_excl, -n_c1, -n_c0],
        labels=[f"Total Pool\n{total:,}", f"Excluded\n{n_excl:,}",
                f"Class 1\n{n_c1:,}", f"Class 0\n{n_c0:,}"],
        orientations=[0, 1, 0, -1],
        pathlengths=[0.25, 0.6, 0.6, 0.6],
        trunklength=1.2,
        facecolor="#AED6F1",
        edgecolor="#2C3E50"
    )
    sankey.finish()

    ax.set_title("Step 2: Label Assignment Flow (Sankey)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_sankey_diagram.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    logger.info("Saved label_sankey_diagram.png")


def viz_m2_forest_plot():
    """M.2#14: Cohen's d forest plot with bootstrap 95% CI."""
    labels_df = pd.read_csv(TRAINING_LABELS_CSV)

    features = ["speech_duration", "score_P1", "score_P2", "score_P3"]
    names = ["Duration", "P1 Score", "P2 Score", "P3 Score"]

    d_vals, ci_los, ci_his = [], [], []
    rng = np.random.RandomState(42)

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        s = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                     (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
        return (np.mean(b) - np.mean(a)) / s if s > 0 else 0.0

    for feat in features:
        c0 = labels_df.loc[labels_df["label"] == 0, feat].values
        c1 = labels_df.loc[labels_df["label"] == 1, feat].values
        n0, n1 = len(c0), len(c1)

        d_vals.append(cohens_d(c0, c1))

        boots = []
        for _ in range(1000):
            bc0 = rng.choice(c0, n0, replace=True)
            bc1 = rng.choice(c1, n1, replace=True)
            boots.append(cohens_d(bc0, bc1))
        ci_los.append(np.percentile(boots, 2.5))
        ci_his.append(np.percentile(boots, 97.5))

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))

    xerr_lo = np.array(d_vals) - np.array(ci_los)
    xerr_hi = np.array(ci_his) - np.array(d_vals)
    ax.errorbar(d_vals, y, xerr=[xerr_lo, xerr_hi], fmt="o",
                color="#2C3E50", markersize=8, capsize=5,
                capthick=2, elinewidth=2)
    ax.axvline(0, color="gray", ls="--", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's d (Class 1 - Class 0)")
    ax.set_title("Class Separation Forest Plot (95% Bootstrap CI)",
                 fontsize=13, fontweight="bold")

    for i in range(len(names)):
        ax.text(ci_his[i] + 0.05, i,
                f"d={d_vals[i]:.2f} [{ci_los[i]:.2f}, {ci_his[i]:.2f}]",
                va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label_class_separation_forest.png", dpi=150)
    plt.close()
    logger.info("Saved label_class_separation_forest.png")


def main():
    logger.info("Generating Step 2.2-2.6 visualizations...")

    viz_2_2_thresholds()
    viz_2_3_labels()
    viz_2_4_balanced()
    viz_2_5_fisher()
    viz_2_6_validation()

    # M.2 gap visualizations
    viz_m2_selection_funnel()
    viz_m2_quality_scatter()
    viz_m2_ambiguous_zone()
    viz_m2_score_kde()
    viz_m2_waterfall()
    viz_m2_strip_swarm()
    viz_m2_parallel_coords()
    viz_m2_sankey()
    viz_m2_forest_plot()

    # Count total plots
    total_plots = 0
    for sub_dir in REPORT_DIR.iterdir():
        if sub_dir.is_dir() and sub_dir.name.startswith("2."):
            pngs = list(sub_dir.glob("*.png"))
            total_plots += len(pngs)
            logger.info(f"  {sub_dir.name}: {len(pngs)} plots")

    logger.info(f"Total plots generated: {total_plots}")


if __name__ == "__main__":
    main()
