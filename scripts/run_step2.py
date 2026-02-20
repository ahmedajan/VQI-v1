"""
Step 2.2-2.6 Orchestrator: Binary Label Definition + Fisher Ratio.

Runs sub-steps sequentially:
  2.2  Compute provider score thresholds
  2.3  Assign binary labels (Class 0 / Class 1)
  2.4  Balance training set (1:1 downsample)
  2.5  Compute Fisher Ratio (d') for feature evaluation
  2.6  Prepare validation set

Runtime: ~5-10 minutes total, no checkpointing needed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vqi.training.compute_labels import (
    compute_thresholds,
    save_thresholds,
    assign_labels,
    balance_training_set,
    compute_fisher_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
SCORE_DIR = PROJECT_ROOT / "data" / "provider_scores"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
STATS_JSON = SCORE_DIR / "score_statistics.json"

# Outputs
THRESHOLDS_YAML = DATA_DIR / "label_thresholds.yaml"
TRAINING_LABELS_CSV = DATA_DIR / "training_labels.csv"
TRAINING_SET_FINAL_CSV = DATA_DIR / "training_set_final.csv"
FISHER_VALUES_CSV = DATA_DIR / "fisher_values.csv"
VALIDATION_SET_CSV = DATA_DIR / "validation_set.csv"

DURATIONS_CSV = LABELS_DIR / "train_pool_durations.csv"
VAL_SET_CSV = SPLITS_DIR / "val_set.csv"


def step_2_2():
    """Compute provider score thresholds."""
    logger.info("=" * 60)
    logger.info("Step 2.2: Compute Provider Score Thresholds")
    logger.info("=" * 60)

    thresholds = compute_thresholds(SCORE_DIR)
    save_thresholds(thresholds, THRESHOLDS_YAML)

    # Print summary
    print("\n  Threshold Summary:")
    print(f"  {'Provider':<8} {'90th Percentile':>16} {'FMR=0.001':>12} {'Gap':>10}")
    print(f"  {'-'*8:<8} {'-'*16:>16} {'-'*12:>12} {'-'*10:>10}")
    for p_name, vals in thresholds.items():
        gap = vals["percentile_90"] - vals["fmr_001"]
        print(f"  {p_name:<8} {vals['percentile_90']:>16.4f} {vals['fmr_001']:>12.4f} {gap:>10.4f}")
    print()

    return thresholds


def step_2_3(thresholds):
    """Assign binary labels."""
    logger.info("=" * 60)
    logger.info("Step 2.3: Assign Binary Labels")
    logger.info("=" * 60)

    labels_df = assign_labels(DURATIONS_CSV, thresholds, SCORE_DIR)
    labels_df.to_csv(TRAINING_LABELS_CSV, index=False, encoding="utf-8")
    logger.info(f"Training labels saved to {TRAINING_LABELS_CSV}")
    logger.info(f"  Total labeled samples: {len(labels_df):,}")

    # Verify
    assert labels_df["label"].isin([0, 1]).all(), "Labels must be 0 or 1"
    assert labels_df["row_idx"].is_unique, "row_idx must be unique"
    assert not labels_df.isna().any().any(), "No NaN allowed in training labels"

    return labels_df


def step_2_4(labels_df):
    """Balance training set."""
    logger.info("=" * 60)
    logger.info("Step 2.4: Balance Training Set")
    logger.info("=" * 60)

    balanced_df = balance_training_set(labels_df, seed=42)
    balanced_df.to_csv(TRAINING_SET_FINAL_CSV, index=False, encoding="utf-8")
    logger.info(f"Balanced training set saved to {TRAINING_SET_FINAL_CSV}")

    # Verify 1:1 ratio
    counts = balanced_df["label"].value_counts()
    assert counts[0] == counts[1], f"Not balanced: {counts.to_dict()}"
    # Verify subset of labels_df
    assert set(balanced_df["row_idx"]).issubset(set(labels_df["row_idx"])), \
        "Balanced set must be subset of labeled set"

    # Print dataset composition
    print("\n  Balanced set composition by dataset:")
    for ds, group in balanced_df.groupby("dataset_source"):
        c0 = (group["label"] == 0).sum()
        c1 = (group["label"] == 1).sum()
        print(f"    {ds}: Class 0={c0:,}, Class 1={c1:,}, Total={len(group):,}")
    print()

    return balanced_df


def step_2_5():
    """Compute Fisher Ratio (d') for feature evaluation."""
    logger.info("=" * 60)
    logger.info("Step 2.5: Compute Fisher Ratio (Feature Evaluation)")
    logger.info("=" * 60)

    fisher_df = compute_fisher_ratio(SCORE_DIR, STATS_JSON)
    fisher_df.to_csv(FISHER_VALUES_CSV, index=False, encoding="utf-8")
    logger.info(f"Fisher Ratio values saved to {FISHER_VALUES_CSV}")

    # Verify
    for col in ["fisher_P1", "fisher_P2", "fisher_P3", "fisher_mean"]:
        assert not fisher_df[col].isna().any(), f"NaN in {col}"
    # P4/P5 should be all NaN
    assert fisher_df["fisher_P4"].isna().all(), "fisher_P4 should be all NaN"
    assert fisher_df["fisher_P5"].isna().all(), "fisher_P5 should be all NaN"

    # Print Fisher Ratio distribution summary
    print("\n  Fisher Ratio Distribution Summary:")
    print(f"  {'Column':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'P5':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*16:<16} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    for col in ["fisher_P1", "fisher_P2", "fisher_P3", "fisher_mean"]:
        vals = fisher_df[col]
        print(
            f"  {col:<16} {vals.mean():>8.3f} {vals.std():>8.3f} "
            f"{vals.min():>8.3f} {np.percentile(vals, 5):>8.3f} "
            f"{vals.median():>8.3f} {np.percentile(vals, 95):>8.3f} "
            f"{vals.max():>8.3f}"
        )
    print()

    return fisher_df


def step_2_6(balanced_df):
    """Prepare validation set."""
    logger.info("=" * 60)
    logger.info("Step 2.6: Prepare Validation Set")
    logger.info("=" * 60)

    # Load validation split
    val_df = pd.read_csv(VAL_SET_CSV)
    logger.info(f"Loaded validation set: {len(val_df):,} samples")

    # To merge with scores, we need row_idx. The val_set.csv uses filename
    # as key, so we load the score CSVs to get row_idx + scores for val samples.
    # Load one score CSV to get the filename->row_idx mapping
    score_csv = SCORE_DIR / "scores_P1_ECAPA_ecapa.csv"
    full_scores = pd.read_csv(score_csv, usecols=["row_idx", "filename"])

    # But val_set samples may not be in the training pool scores
    # (they were split out before scoring). Let's check.
    # Actually, Step 1.6 scored ALL samples in the full pool BEFORE splitting.
    # The split was done on the same pool. So val samples should have scores.

    # Build filename -> row_idx lookup from score file
    # The score CSVs contain ALL 1,210,451 training pool samples.
    # val_set.csv was drawn from the full combined pool (train_pool + val_set + test sets).
    # We need to check: are val_set filenames in the score CSVs?

    val_filenames = set(val_df["filename"].values)
    score_filenames = set(full_scores["filename"].values)
    overlap = val_filenames & score_filenames

    if len(overlap) == len(val_filenames):
        logger.info("All validation samples found in score CSVs.")
    else:
        # Val samples not in training pool won't have P1-P3 scores yet.
        # This is expected -- val samples are separate from training pool.
        logger.info(
            f"Validation samples in score CSVs: {len(overlap):,} / {len(val_filenames):,}"
        )
        if len(overlap) == 0:
            logger.info(
                "No overlap -- val_set was split BEFORE scoring. "
                "Scores for validation will be computed in Step 7."
            )

    # Merge available scores
    result = val_df.copy()

    if len(overlap) > 0:
        for short_name, full_name, score_csv_name, _ in [
            ("P1", "P1_ECAPA", "scores_P1_ECAPA_ecapa.csv", None),
            ("P2", "P2_RESNET", "scores_P2_RESNET_resnet.csv", None),
            ("P3", "P3_ECAPA2", "scores_P3_ECAPA2_ecapa2.csv", None),
        ]:
            sdf = pd.read_csv(
                SCORE_DIR / score_csv_name,
                usecols=["filename", "genuine_norm"],
            )
            sdf = sdf.rename(columns={"genuine_norm": f"score_{short_name}"})
            result = result.merge(sdf, on="filename", how="left")
    else:
        # Add empty score columns
        for short_name in ["P1", "P2", "P3"]:
            result[f"score_{short_name}"] = np.nan

    # Verify zero overlap with balanced training set
    train_filenames = set(balanced_df["filename"].values)
    val_train_overlap = val_filenames & train_filenames
    if len(val_train_overlap) > 0:
        logger.error(
            f"CRITICAL: {len(val_train_overlap)} samples overlap between "
            f"validation and training sets!"
        )
        raise ValueError("Training/validation overlap detected")
    else:
        logger.info("Zero overlap between validation and training sets. OK.")

    result.to_csv(VALIDATION_SET_CSV, index=False, encoding="utf-8")
    logger.info(f"Validation set saved to {VALIDATION_SET_CSV}")
    logger.info(f"  Rows: {len(result):,}")

    # Print composition
    print("\n  Validation set composition:")
    for ds, group in result.groupby("dataset_source"):
        print(f"    {ds}: {len(group):,}")
    n_with_scores = result["score_P1"].notna().sum()
    print(f"  Samples with P1-P3 scores: {n_with_scores:,} / {len(result):,}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(description="Step 2.2-2.6: Label definition pipeline")
    parser.add_argument("--skip-vis", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    t_start = time.time()

    # 2.2 Thresholds
    t0 = time.time()
    thresholds = step_2_2()
    logger.info(f"Step 2.2 completed in {time.time()-t0:.1f}s\n")

    # 2.3 Labels
    t0 = time.time()
    labels_df = step_2_3(thresholds)
    logger.info(f"Step 2.3 completed in {time.time()-t0:.1f}s\n")

    # 2.4 Balance
    t0 = time.time()
    balanced_df = step_2_4(labels_df)
    logger.info(f"Step 2.4 completed in {time.time()-t0:.1f}s\n")

    # 2.5 Fisher Ratio
    t0 = time.time()
    fisher_df = step_2_5()
    logger.info(f"Step 2.5 completed in {time.time()-t0:.1f}s\n")

    # 2.6 Validation
    t0 = time.time()
    val_df = step_2_6(balanced_df)
    logger.info(f"Step 2.6 completed in {time.time()-t0:.1f}s\n")

    total = time.time() - t_start
    logger.info(f"All sub-steps (2.2-2.6) completed in {total:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("Step 2 Output Summary")
    print("=" * 60)
    print(f"  label_thresholds.yaml:   {THRESHOLDS_YAML}")
    print(f"  training_labels.csv:     {len(labels_df):,} rows")
    print(f"  training_set_final.csv:  {len(balanced_df):,} rows (1:1 balanced)")
    print(f"  fisher_values.csv:       {len(fisher_df):,} rows")
    print(f"  validation_set.csv:      {len(val_df):,} rows")
    print(f"  Total time:              {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
