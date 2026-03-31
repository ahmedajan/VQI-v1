"""X1.9d: Two-gate label selection for expansion pool.

Applies the same two-gate logic from compute_labels.py using thresholds
from data/step2/label_thresholds.yaml on the expansion pool's S-norm scores
and durations.

Class 1: speech_duration >= 3.0s AND genuine_norm >= P90 for ALL 3 providers
Class 0: speech_duration < 1.5s OR genuine_norm < FMR_001 for ALL 3 providers
Excluded: samples matching neither rule

Output: implementation/data/step2/labels/x1_expansion_pool_labels.csv
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Inputs
DURATIONS_CSV = os.path.join(PROJECT_ROOT, "data", "step2", "labels",
                              "x1_expansion_pool_durations.csv")
THRESHOLDS_YAML = os.path.join(PROJECT_ROOT, "data", "step2",
                                "label_thresholds.yaml")
SCORE_DIR = os.path.join(PROJECT_ROOT, "data", "step1", "provider_scores")

# Score CSV names for expansion pool (different prefix from original)
PROVIDERS = [
    ("P1", "scores_x1_expansion_pool_P1_ECAPA_ecapa.csv"),
    ("P2", "scores_x1_expansion_pool_P2_RESNET_resnet.csv"),
    ("P3", "scores_x1_expansion_pool_P3_ECAPA2_ecapa2.csv"),
]

# Duration thresholds (same as original)
DURATION_HIGH = 3.0   # Class 1 eligibility
DURATION_LOW = 1.5    # Class 0 forced

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "step2", "labels")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "x1_expansion_pool_labels.csv")


def main():
    logger.info("=" * 70)
    logger.info("X1.9d: Two-Gate Label Selection for Expansion Pool")
    logger.info("=" * 70)

    # Load thresholds
    logger.info(f"Loading thresholds: {THRESHOLDS_YAML}")
    with open(THRESHOLDS_YAML, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)
    for p in ["P1", "P2", "P3"]:
        logger.info(f"  {p}: P90={thresholds[p]['percentile_90']:.4f}, "
                     f"FMR={thresholds[p]['fmr_001']:.4f}")

    # Load durations
    logger.info(f"Loading durations: {DURATIONS_CSV}")
    dur_df = pd.read_csv(
        DURATIONS_CSV,
        usecols=["row_idx", "filename", "speaker_id", "dataset_source",
                 "speech_duration_sec"],
    )
    dur_df = dur_df.rename(columns={"speech_duration_sec": "speech_duration"})

    # Filter out failed files (speech_duration < 0)
    n_failed = (dur_df["speech_duration"] < 0).sum()
    if n_failed > 0:
        logger.warning(f"Removing {n_failed} failed files (duration < 0)")
        dur_df = dur_df[dur_df["speech_duration"] >= 0].copy()
    logger.info(f"Duration data: {len(dur_df):,} rows")

    # Load and merge provider scores
    for short_name, score_csv in PROVIDERS:
        logger.info(f"Loading scores: {short_name} ({score_csv})")
        score_df = pd.read_csv(
            os.path.join(SCORE_DIR, score_csv),
            usecols=["row_idx", "genuine_norm"],
        )
        score_df = score_df.rename(
            columns={"genuine_norm": f"score_{short_name}"})
        dur_df = dur_df.merge(score_df, on="row_idx", how="left")

    logger.info(f"Merged dataset: {len(dur_df):,} rows")

    # Verify no NaN
    score_cols = [f"score_{p[0]}" for p in PROVIDERS]
    nan_counts = dur_df[score_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN in score columns: "
                       f"{nan_counts[nan_counts > 0].to_dict()}")

    # === Two-Gate Logic ===

    # Class 1: duration >= 3.0 AND all 3 scores >= P90
    class1_duration = dur_df["speech_duration"] >= DURATION_HIGH
    class1_scores = pd.Series(True, index=dur_df.index)
    for short_name, _ in PROVIDERS:
        class1_scores &= (dur_df[f"score_{short_name}"] >=
                          thresholds[short_name]["percentile_90"])
    class1_mask = class1_duration & class1_scores

    # Class 0: duration < 1.5 (forced) OR all 3 scores < FMR
    class0_duration = dur_df["speech_duration"] < DURATION_LOW
    class0_scores = pd.Series(True, index=dur_df.index)
    for short_name, _ in PROVIDERS:
        class0_scores &= (dur_df[f"score_{short_name}"] <
                          thresholds[short_name]["fmr_001"])
    class0_mask = class0_duration | class0_scores

    # Resolve conflicts (Class 0 priority)
    conflict_mask = class1_mask & class0_mask
    if conflict_mask.sum() > 0:
        logger.warning(f"{conflict_mask.sum()} samples match both rules. "
                       f"Assigning to Class 0 (conservative).")
        class1_mask &= ~conflict_mask

    # Assign labels
    dur_df["label"] = -1  # excluded
    dur_df.loc[class1_mask, "label"] = 1
    dur_df.loc[class0_mask, "label"] = 0

    # === Statistics ===
    n_class1 = class1_mask.sum()
    n_class0 = class0_mask.sum()
    n_excluded = len(dur_df) - n_class1 - n_class0
    logger.info(f"Label assignment results:")
    logger.info(f"  Class 1 (high-quality):   {n_class1:,} "
                f"({100*n_class1/len(dur_df):.2f}%)")
    logger.info(f"  Class 0 (low-quality):    {n_class0:,} "
                f"({100*n_class0/len(dur_df):.2f}%)")
    logger.info(f"  Excluded:                 {n_excluded:,} "
                f"({100*n_excluded/len(dur_df):.2f}%)")
    logger.info(f"  Total labeled:            {n_class1 + n_class0:,}")

    # Class 0 breakdown
    n_c0_duration_only = (class0_duration & ~class0_scores).sum()
    n_c0_scores_only = (class0_scores & ~class0_duration).sum()
    n_c0_both = (class0_duration & class0_scores).sum()
    logger.info(f"  Class 0 breakdown:")
    logger.info(f"    Duration < {DURATION_LOW}s only:    {n_c0_duration_only:,}")
    logger.info(f"    All scores < FMR only:   {n_c0_scores_only:,}")
    logger.info(f"    Both duration + scores:  {n_c0_both:,}")

    # Class 1 score distribution
    c1_scores = dur_df.loc[class1_mask]
    for short_name, _ in PROVIDERS:
        vals = c1_scores[f"score_{short_name}"]
        logger.info(f"  Class 1 {short_name} scores: "
                     f"mean={vals.mean():.3f}, min={vals.min():.3f}, "
                     f"max={vals.max():.3f}")

    # Dataset source breakdown
    logger.info(f"  By dataset source:")
    for src in dur_df["dataset_source"].unique():
        mask_src = dur_df["dataset_source"] == src
        n_src = mask_src.sum()
        n_c1 = (class1_mask & mask_src).sum()
        n_c0 = (class0_mask & mask_src).sum()
        logger.info(f"    {src}: {n_src:,} total, "
                     f"C1={n_c1:,} ({100*n_c1/n_src:.1f}%), "
                     f"C0={n_c0:,} ({100*n_c0/n_src:.1f}%)")

    # Speaker count in labeled data
    labeled_df = dur_df[dur_df["label"] >= 0].copy()
    labeled_df["label"] = labeled_df["label"].astype(int)
    n_speakers_c1 = labeled_df[labeled_df["label"] == 1]["speaker_id"].nunique()
    n_speakers_c0 = labeled_df[labeled_df["label"] == 0]["speaker_id"].nunique()
    logger.info(f"  Unique speakers: C1={n_speakers_c1:,}, C0={n_speakers_c0:,}")

    # Save labeled-only CSV
    output_cols = [
        "row_idx", "filename", "speaker_id", "dataset_source",
        "label", "speech_duration", "score_P1", "score_P2", "score_P3",
    ]
    labeled_df = labeled_df[output_cols].reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labeled_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved: {OUTPUT_CSV} ({len(labeled_df):,} rows)")

    # Compare with original training pool
    logger.info(f"\nComparison with original training pool:")
    logger.info(f"  Original: 20,288 balanced (10,144 per class)")
    logger.info(f"  Expansion: {len(labeled_df):,} labeled "
                f"(C1={n_class1:,}, C0={n_class0:,})")
    logger.info(f"  Ratio C1:C0 = {n_class1/(max(n_class0,1)):.2f}:1")

    logger.info("\nX1.9d label selection COMPLETE")


if __name__ == "__main__":
    main()
