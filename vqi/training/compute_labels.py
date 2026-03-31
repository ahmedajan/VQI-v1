"""
Step 2.2-2.5: Binary label definition and Fisher Ratio computation.

Two distinct concepts:
  - Binary labels (Class 0 / Class 1): Training targets for the RF classifier.
    Driven by provider scores + duration thresholds. Used in Steps 6-7.
  - Fisher Ratio (per-speaker): Fisher Discriminant Ratio (d') per speaker per
    provider. Used for feature evaluation in Step 5 (Spearman correlation).

These are independent -- binary labels drive model training; Fisher Ratio drives
feature selection.

Score type: All thresholds and computations use S-normalized scores
(genuine_norm, impostor_norm) which center impostors near 0 with unit variance,
making thresholds comparable across providers.

P4/P5 gap: Training pool only has P1-P3 embeddings/scores (Step 1.6 extracted
P1-P3 only). Fisher Ratio computed for P1-P3 only. P4/P5 columns set to NaN.
P4/P5 extraction deferred until before Step 7.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Provider names matching file naming convention from Step 1.6b
PROVIDERS = [
    ("P1", "P1_ECAPA", "scores_P1_ECAPA_ecapa.csv", "impostor_norm_P1_ECAPA.npy"),
    ("P2", "P2_RESNET", "scores_P2_RESNET_resnet.csv", "impostor_norm_P2_RESNET.npy"),
    ("P3", "P3_ECAPA2", "scores_P3_ECAPA2_ecapa2.csv", "impostor_norm_P3_ECAPA2.npy"),
]

# Duration thresholds (from blueprint Phase_Step2)
DURATION_HIGH = 3.0   # seconds, Class 1 eligibility
DURATION_LOW = 1.5    # seconds, Class 0 forced


def compute_thresholds(score_dir: str | Path) -> dict:
    """Step 2.2: Compute provider score thresholds for binary label assignment.

    For each provider (P1-P3):
      - percentile_90: 90th percentile of genuine_norm scores (Class 1 gate)
      - fmr_001: score at FMR=0.001 on impostor_norm distribution (Class 0 gate)

    Args:
        score_dir: Path to implementation/data/step1/provider_scores/

    Returns:
        Dict with thresholds per provider, e.g.:
        {"P1": {"percentile_90": 11.2, "fmr_001": 3.05}, ...}
    """
    score_dir = Path(score_dir)
    thresholds = {}

    for short_name, full_name, score_csv, impostor_npy in PROVIDERS:
        logger.info(f"Computing thresholds for {short_name} ({full_name})...")

        # Load genuine normalized scores
        df = pd.read_csv(score_dir / score_csv, usecols=["genuine_norm"])
        genuine_norm = df["genuine_norm"].values

        # 90th percentile of genuine scores
        p90 = float(np.percentile(genuine_norm, 90))

        # Load impostor normalized scores
        impostor_norm = np.load(score_dir / impostor_npy)

        # FMR=0.001 threshold: the score at which only 0.1% of impostors
        # score above. Sort descending, take index at 0.1% of total count.
        # This is the 99.9th percentile of the impostor distribution.
        fmr_threshold = float(np.percentile(impostor_norm, 99.9))

        thresholds[short_name] = {
            "percentile_90": p90,
            "fmr_001": fmr_threshold,
        }

        logger.info(
            f"  {short_name}: percentile_90={p90:.4f}, fmr_001={fmr_threshold:.4f}"
        )

        # Sanity check: 90th percentile should be well above FMR threshold
        if p90 <= fmr_threshold:
            logger.warning(
                f"  WARNING: percentile_90 ({p90:.4f}) <= fmr_001 ({fmr_threshold:.4f}) "
                f"for {short_name}. This means Class 1 gate <= Class 0 gate, "
                f"which would make label assignment impossible."
            )

    return thresholds


def save_thresholds(thresholds: dict, output_path: str | Path) -> None:
    """Save thresholds to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(thresholds, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Thresholds saved to {output_path}")


def assign_labels(
    durations_csv: str | Path,
    thresholds: dict,
    score_dir: str | Path,
) -> pd.DataFrame:
    """Step 2.3: Assign binary labels based on provider scores + duration.

    Label rules (from blueprint Phase_Step2):
      - Class 1: speech_duration >= 3.0 AND genuine_norm >= percentile_90 for ALL 3 providers
      - Class 0: speech_duration < 1.5 (forced) OR genuine_norm < fmr_001 for ALL 3 providers
      - Excluded: samples matching neither rule

    Args:
        durations_csv: Path to train_pool_durations.csv (Step 2.1 output)
        thresholds: Dict from compute_thresholds()
        score_dir: Path to provider_scores directory

    Returns:
        DataFrame with columns: row_idx, filename, speaker_id, dataset_source,
        label, speech_duration, score_P1, score_P2, score_P3
    """
    score_dir = Path(score_dir)

    # Load durations
    logger.info("Loading durations...")
    dur_df = pd.read_csv(
        durations_csv,
        usecols=["row_idx", "filename", "speaker_id", "dataset_source", "speech_duration_sec"],
    )
    dur_df = dur_df.rename(columns={"speech_duration_sec": "speech_duration"})

    # Load and merge provider scores
    for short_name, full_name, score_csv, _ in PROVIDERS:
        logger.info(f"Loading scores for {short_name}...")
        score_df = pd.read_csv(
            score_dir / score_csv,
            usecols=["row_idx", "genuine_norm"],
        )
        score_df = score_df.rename(columns={"genuine_norm": f"score_{short_name}"})
        dur_df = dur_df.merge(score_df, on="row_idx", how="left")

    logger.info(f"Merged dataset: {len(dur_df)} rows")

    # Verify no NaN in scores
    score_cols = [f"score_{p[0]}" for p in PROVIDERS]
    nan_counts = dur_df[score_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN in score columns: {nan_counts[nan_counts > 0].to_dict()}")

    # Apply label rules
    # Class 1: duration >= 3.0 AND all 3 scores >= their 90th percentile
    class1_duration = dur_df["speech_duration"] >= DURATION_HIGH
    class1_scores = pd.Series(True, index=dur_df.index)
    for short_name, _, _, _ in PROVIDERS:
        class1_scores &= dur_df[f"score_{short_name}"] >= thresholds[short_name]["percentile_90"]
    class1_mask = class1_duration & class1_scores

    # Class 0: duration < 1.5 (forced) OR all 3 scores < their FMR threshold
    class0_duration = dur_df["speech_duration"] < DURATION_LOW
    class0_scores = pd.Series(True, index=dur_df.index)
    for short_name, _, _, _ in PROVIDERS:
        class0_scores &= dur_df[f"score_{short_name}"] < thresholds[short_name]["fmr_001"]
    class0_mask = class0_duration | class0_scores

    # Resolve conflicts: if a sample matches BOTH rules, Class 0 takes priority
    # (a sample with duration < 1.5 could theoretically have high scores)
    conflict_mask = class1_mask & class0_mask
    if conflict_mask.sum() > 0:
        logger.warning(
            f"  {conflict_mask.sum()} samples match both Class 0 and Class 1 rules. "
            f"Assigning to Class 0 (conservative)."
        )
        class1_mask &= ~conflict_mask

    # Assign labels
    dur_df["label"] = -1  # -1 = excluded
    dur_df.loc[class1_mask, "label"] = 1
    dur_df.loc[class0_mask, "label"] = 0

    # Log statistics
    n_class1 = class1_mask.sum()
    n_class0 = class0_mask.sum()
    n_excluded = len(dur_df) - n_class1 - n_class0
    logger.info(f"Label assignment:")
    logger.info(f"  Class 1 (high-performing): {n_class1:,} ({100*n_class1/len(dur_df):.2f}%)")
    logger.info(f"  Class 0 (low-performing):  {n_class0:,} ({100*n_class0/len(dur_df):.2f}%)")
    logger.info(f"  Excluded:                  {n_excluded:,} ({100*n_excluded/len(dur_df):.2f}%)")

    # Break down Class 0 by reason
    n_c0_duration = class0_duration.sum()
    n_c0_scores_only = (class0_scores & ~class0_duration).sum()
    n_c0_both = (class0_duration & class0_scores).sum()
    logger.info(f"  Class 0 breakdown:")
    logger.info(f"    Duration < {DURATION_LOW}s only:       {n_c0_duration - n_c0_both:,}")
    logger.info(f"    All scores < FMR only:     {n_c0_scores_only:,}")
    logger.info(f"    Both duration + scores:    {n_c0_both:,}")

    # Filter to labeled samples only (exclude -1)
    labeled_df = dur_df[dur_df["label"] >= 0].copy()
    labeled_df["label"] = labeled_df["label"].astype(int)

    # Select output columns
    output_cols = [
        "row_idx", "filename", "speaker_id", "dataset_source",
        "label", "speech_duration", "score_P1", "score_P2", "score_P3",
    ]
    return labeled_df[output_cols].reset_index(drop=True)


def balance_training_set(labels_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Step 2.4: Downsample majority class to create balanced 1:1 training set.

    Args:
        labels_df: DataFrame from assign_labels() with 'label' column
        seed: Random seed for reproducibility

    Returns:
        Balanced DataFrame with equal Class 0 and Class 1 counts, shuffled.
    """
    class_counts = labels_df["label"].value_counts()
    n_class0 = class_counts.get(0, 0)
    n_class1 = class_counts.get(1, 0)
    minority_count = min(n_class0, n_class1)

    logger.info(f"Before balancing: Class 0={n_class0:,}, Class 1={n_class1:,}")
    logger.info(f"Minority class: {'Class 0' if n_class0 < n_class1 else 'Class 1'} ({minority_count:,})")

    rng = np.random.RandomState(seed)

    # Downsample each class to minority count
    class0_df = labels_df[labels_df["label"] == 0]
    class1_df = labels_df[labels_df["label"] == 1]

    if n_class0 > minority_count:
        class0_df = class0_df.sample(n=minority_count, random_state=rng)
    if n_class1 > minority_count:
        class1_df = class1_df.sample(n=minority_count, random_state=rng)

    balanced_df = pd.concat([class0_df, class1_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    logger.info(f"After balancing: {len(balanced_df):,} total (2 x {minority_count:,})")
    return balanced_df


def compute_fisher_ratio(
    score_dir: str | Path,
    stats_json: str | Path,
) -> pd.DataFrame:
    """Step 2.5: Compute per-speaker Fisher Discriminant Ratio (d').

    Fisher Ratio is used for feature evaluation in Step 5 (Spearman correlation
    between features and Fisher Ratio). It is NOT used for binary label assignment.

    Formula (ISO/IEC 29794-1 Annex A):
        d'(speaker_i, provider_u) = (mu_genuine_i - mu_impostor) / (sigma_genuine_i + sigma_impostor)

    where:
        mu_genuine_i = mean of genuine_norm scores for speaker i
        sigma_genuine_i = std of genuine_norm scores for speaker i
        mu_impostor = global impostor mean (from score_statistics.json)
        sigma_impostor = global impostor std (from score_statistics.json)

    Uses sum-of-sigmas denominator (not pooled RMS) per blueprint spec.

    Args:
        score_dir: Path to provider_scores directory
        stats_json: Path to score_statistics.json

    Returns:
        DataFrame with columns: row_idx, filename, speaker_id, fisher_P1,
        fisher_P2, fisher_P3, fisher_P4, fisher_P5, fisher_mean
        (fisher_P4 and fisher_P5 are NaN -- no embeddings available)
    """
    score_dir = Path(score_dir)

    # Load global impostor statistics
    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    impostor_stats = {}
    provider_map = {"P1": "P1_ECAPA", "P2": "P2_RESNET", "P3": "P3_ECAPA2"}
    for short_name, full_name in provider_map.items():
        impostor_stats[short_name] = {
            "mu": stats[full_name]["impostor_norm"]["mean"],
            "sigma": stats[full_name]["impostor_norm"]["std"],
        }

    # Load all provider score CSVs and compute per-speaker stats
    all_scores = None
    for short_name, full_name, score_csv, _ in PROVIDERS:
        logger.info(f"Loading scores for {short_name}...")
        df = pd.read_csv(
            score_dir / score_csv,
            usecols=["row_idx", "filename", "speaker_id", "genuine_norm"],
        )
        df = df.rename(columns={"genuine_norm": f"score_{short_name}"})

        if all_scores is None:
            all_scores = df
        else:
            all_scores = all_scores.merge(
                df[["row_idx", f"score_{short_name}"]],
                on="row_idx",
                how="left",
            )

    # Compute per-speaker mean and std of genuine_norm for each provider
    speaker_groups = all_scores.groupby("speaker_id")

    fisher_cols = {}
    for short_name in ["P1", "P2", "P3"]:
        score_col = f"score_{short_name}"
        mu_imp = impostor_stats[short_name]["mu"]
        sigma_imp = impostor_stats[short_name]["sigma"]

        # Per-speaker stats
        speaker_stats = speaker_groups[score_col].agg(["mean", "std"])
        speaker_stats.columns = ["mu_genuine", "sigma_genuine"]

        # Handle speakers with only 1 sample (std = NaN) -> set sigma = 0
        speaker_stats["sigma_genuine"] = speaker_stats["sigma_genuine"].fillna(0.0)

        # Fisher Ratio per speaker
        speaker_stats[f"fisher_{short_name}"] = (
            (speaker_stats["mu_genuine"] - mu_imp)
            / (speaker_stats["sigma_genuine"] + sigma_imp)
        )

        fisher_cols[short_name] = speaker_stats[f"fisher_{short_name}"]

    # Build per-speaker Fisher Ratio dataframe
    fisher_df = pd.DataFrame(fisher_cols)
    fisher_df.columns = ["fisher_P1", "fisher_P2", "fisher_P3"]
    fisher_df["fisher_mean"] = fisher_df[["fisher_P1", "fisher_P2", "fisher_P3"]].mean(axis=1)

    # Map back to per-sample level
    result = all_scores[["row_idx", "filename", "speaker_id"]].copy()
    result = result.merge(fisher_df, left_on="speaker_id", right_index=True, how="left")

    # Add NaN columns for P4/P5 (no embeddings available)
    result["fisher_P4"] = np.nan
    result["fisher_P5"] = np.nan

    # Reorder columns
    output_cols = [
        "row_idx", "filename", "speaker_id",
        "fisher_P1", "fisher_P2", "fisher_P3",
        "fisher_P4", "fisher_P5", "fisher_mean",
    ]
    result = result[output_cols]

    # Log statistics
    for col in ["fisher_P1", "fisher_P2", "fisher_P3", "fisher_mean"]:
        vals = result[col].dropna()
        logger.info(
            f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
            f"min={vals.min():.4f}, max={vals.max():.4f}"
        )

    n_speakers = all_scores["speaker_id"].nunique()
    logger.info(f"Computed Fisher Ratio for {n_speakers:,} speakers across {len(result):,} samples")

    return result
