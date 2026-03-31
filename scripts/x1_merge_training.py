"""X1.9f: Merge expansion pool with original training set.

Steps:
1. Load expansion features (544 S / 161 V) → apply feature selection (430 S / 133 V)
2. Load expansion labels, balance C1 down to match C0 (18,910 per class)
3. Load original training data (20,288 balanced)
4. Concatenate → 58,108 total (29,054 per class)
5. Construct regression targets for expansion samples (same normalization)
6. Save merged X_train, y_train, y_reg_train_fused

Output directory: data/step6/full_feature/training_expanded/ (S)
                  data/step6/full_feature/training_expanded_v/ (V)
                  data/x1_regression/expanded/ (regression targets)
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import _apply_feature_selection

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "step4", "features")

# Inputs
EXPANSION_FEATURES_S = os.path.join(FEATURES_DIR, "features_s_x1_expansion.npy")
EXPANSION_FEATURES_V = os.path.join(FEATURES_DIR, "features_v_x1_expansion.npy")
EXPANSION_LABELS_CSV = os.path.join(DATA_DIR, "step2", "labels",
                                     "x1_expansion_pool_labels.csv")
ORIGINAL_TRAIN_S = os.path.join(DATA_DIR, "step6", "full_feature", "training")
ORIGINAL_TRAIN_V = os.path.join(DATA_DIR, "step6", "full_feature", "training_v")
REG_SCALER = os.path.join(DATA_DIR, "x1_regression", "y_reg_scaler.json")
ORIGINAL_REG_FUSED = os.path.join(DATA_DIR, "x1_regression", "y_reg_train_fused.npy")

# Outputs
OUTPUT_S = os.path.join(DATA_DIR, "step6", "full_feature", "training_expanded")
OUTPUT_V = os.path.join(DATA_DIR, "step6", "full_feature", "training_expanded_v")
OUTPUT_REG = os.path.join(DATA_DIR, "x1_regression", "expanded")

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("X1.9f: Merge Expansion Pool with Original Training Set")
    logger.info("=" * 70)

    # ===== 1. Load expansion labels =====
    logger.info("Loading expansion labels...")
    labels_df = pd.read_csv(EXPANSION_LABELS_CSV)
    n_c0 = (labels_df["label"] == 0).sum()
    n_c1 = (labels_df["label"] == 1).sum()
    logger.info(f"  Expansion labeled: {len(labels_df):,} "
                f"(C1={n_c1:,}, C0={n_c0:,})")

    # ===== 2. Balance expansion: downsample C1 to match C0 =====
    logger.info("Balancing expansion set...")
    rng = np.random.RandomState(RANDOM_STATE)
    minority_count = min(n_c0, n_c1)

    c0_idx = labels_df[labels_df["label"] == 0].index.values
    c1_idx = labels_df[labels_df["label"] == 1].index.values

    if n_c1 > minority_count:
        c1_idx = rng.choice(c1_idx, size=minority_count, replace=False)
    if n_c0 > minority_count:
        c0_idx = rng.choice(c0_idx, size=minority_count, replace=False)

    balanced_idx = np.sort(np.concatenate([c0_idx, c1_idx]))
    balanced_df = labels_df.iloc[balanced_idx].reset_index(drop=True)
    logger.info(f"  Balanced expansion: {len(balanced_df):,} "
                f"(C1={minority_count:,}, C0={minority_count:,})")

    # ===== 3. Load expansion features and apply feature selection =====
    logger.info("Loading expansion features...")
    exp_feat_s_full = np.load(EXPANSION_FEATURES_S)
    exp_feat_v_full = np.load(EXPANSION_FEATURES_V)
    logger.info(f"  Raw shapes: S={exp_feat_s_full.shape}, V={exp_feat_v_full.shape}")

    # Select balanced rows
    exp_feat_s_full = exp_feat_s_full[balanced_idx]
    exp_feat_v_full = exp_feat_v_full[balanced_idx]

    # Check for NaN rows (extraction failures)
    nan_rows_s = np.any(~np.isfinite(exp_feat_s_full), axis=1)
    nan_rows_v = np.any(~np.isfinite(exp_feat_v_full), axis=1)
    nan_rows = nan_rows_s | nan_rows_v
    n_nan = nan_rows.sum()
    if n_nan > 0:
        logger.warning(f"  Removing {n_nan} rows with NaN/Inf features")
        good_mask = ~nan_rows
        exp_feat_s_full = exp_feat_s_full[good_mask]
        exp_feat_v_full = exp_feat_v_full[good_mask]
        balanced_df = balanced_df[good_mask].reset_index(drop=True)

    # Apply feature selection (544 → 430 S, 161 → 133 V)
    logger.info("Applying feature selection...")
    exp_X_s = _apply_feature_selection(exp_feat_s_full, "s")
    exp_X_v = _apply_feature_selection(exp_feat_v_full, "v")
    exp_y = balanced_df["label"].values.astype(np.int32)
    logger.info(f"  Selected shapes: S={exp_X_s.shape}, V={exp_X_v.shape}")

    # ===== 4. Load original training data =====
    logger.info("Loading original training data...")
    orig_X_s = np.load(os.path.join(ORIGINAL_TRAIN_S, "X_train.npy"))
    orig_y_s = np.load(os.path.join(ORIGINAL_TRAIN_S, "y_train.npy"))
    orig_X_v = np.load(os.path.join(ORIGINAL_TRAIN_V, "X_train.npy"))
    orig_y_v = np.load(os.path.join(ORIGINAL_TRAIN_V, "y_train.npy"))
    logger.info(f"  Original: S={orig_X_s.shape}, V={orig_X_v.shape}")
    assert np.array_equal(orig_y_s, orig_y_v), "S and V labels differ!"

    # ===== 5. Concatenate =====
    logger.info("Merging datasets...")
    merged_X_s = np.concatenate([orig_X_s, exp_X_s], axis=0).astype(np.float32)
    merged_X_v = np.concatenate([orig_X_v, exp_X_v], axis=0).astype(np.float32)
    merged_y = np.concatenate([orig_y_s, exp_y], axis=0).astype(np.int32)

    n_total = len(merged_y)
    n_c0_merged = (merged_y == 0).sum()
    n_c1_merged = (merged_y == 1).sum()
    logger.info(f"  Merged: {n_total:,} total "
                f"(C1={n_c1_merged:,}, C0={n_c0_merged:,})")
    logger.info(f"  S shape: {merged_X_s.shape}, V shape: {merged_X_v.shape}")

    # Shuffle
    shuffle_idx = rng.permutation(n_total)
    merged_X_s = merged_X_s[shuffle_idx]
    merged_X_v = merged_X_v[shuffle_idx]
    merged_y = merged_y[shuffle_idx]

    # Verify
    assert merged_X_s.shape == (n_total, 430), f"S shape wrong: {merged_X_s.shape}"
    assert merged_X_v.shape == (n_total, 133), f"V shape wrong: {merged_X_v.shape}"
    assert not np.any(np.isnan(merged_X_s)), "NaN in merged S"
    assert not np.any(np.isnan(merged_X_v)), "NaN in merged V"
    assert set(np.unique(merged_y)) == {0, 1}, f"Labels not binary: {np.unique(merged_y)}"

    # ===== 6. Save classification data =====
    os.makedirs(OUTPUT_S, exist_ok=True)
    os.makedirs(OUTPUT_V, exist_ok=True)

    np.save(os.path.join(OUTPUT_S, "X_train.npy"), merged_X_s)
    np.save(os.path.join(OUTPUT_S, "y_train.npy"), merged_y)
    np.save(os.path.join(OUTPUT_V, "X_train.npy"), merged_X_v)
    np.save(os.path.join(OUTPUT_V, "y_train.npy"), merged_y)
    logger.info(f"  Saved classification data: {OUTPUT_S}, {OUTPUT_V}")

    # ===== 7. Construct regression targets =====
    logger.info("Constructing regression targets...")

    # Load normalization scaler from original X1.7
    with open(REG_SCALER, "r", encoding="utf-8") as f:
        scaler_info = json.load(f)

    # Original regression target
    orig_y_reg = np.load(ORIGINAL_REG_FUSED)
    logger.info(f"  Original reg target: {orig_y_reg.shape}, "
                f"range=[{orig_y_reg.min():.4f}, {orig_y_reg.max():.4f}]")

    # Expansion regression target: fused mean of P1/P2/P3 genuine_norm scores
    exp_scores = balanced_df[["score_P1", "score_P2", "score_P3"]].values
    exp_fused_raw = np.mean(exp_scores, axis=1)

    # Normalize using SAME scaler as original
    y_min = scaler_info["fused"]["y_min"]
    y_max = scaler_info["fused"]["y_max"]
    exp_fused_norm = np.clip(
        (exp_fused_raw - y_min) / (y_max - y_min), 0.0, 1.0
    ).astype(np.float32)

    logger.info(f"  Expansion reg raw: [{exp_fused_raw.min():.3f}, "
                f"{exp_fused_raw.max():.3f}], mean={exp_fused_raw.mean():.3f}")
    logger.info(f"  Expansion reg norm: [{exp_fused_norm.min():.4f}, "
                f"{exp_fused_norm.max():.4f}], mean={exp_fused_norm.mean():.4f}")

    # Merge and shuffle (same order as features)
    merged_y_reg = np.concatenate([orig_y_reg, exp_fused_norm], axis=0)
    merged_y_reg = merged_y_reg[shuffle_idx].astype(np.float32)

    assert merged_y_reg.shape[0] == n_total, "Reg target count mismatch"
    assert not np.any(np.isnan(merged_y_reg)), "NaN in merged reg target"

    os.makedirs(OUTPUT_REG, exist_ok=True)
    np.save(os.path.join(OUTPUT_REG, "y_reg_train_fused.npy"), merged_y_reg)
    logger.info(f"  Saved regression target: {OUTPUT_REG}")

    # ===== 8. Save manifest =====
    # Track which rows are original vs expansion
    source_labels = np.array(
        ["original"] * len(orig_y_s) + ["expansion"] * len(exp_y)
    )[shuffle_idx]
    manifest = pd.DataFrame({
        "source": source_labels,
        "label": merged_y,
        "y_reg_fused": merged_y_reg,
    })
    manifest.to_csv(os.path.join(OUTPUT_S, "training_manifest.csv"), index=False)

    # ===== Summary =====
    logger.info("")
    logger.info("=" * 70)
    logger.info("X1.9f Merge COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total training samples: {n_total:,}")
    logger.info(f"  Class balance: C0={n_c0_merged:,}, C1={n_c1_merged:,}")
    logger.info(f"  Source breakdown:")
    logger.info(f"    Original: {len(orig_y_s):,} ({100*len(orig_y_s)/n_total:.1f}%)")
    logger.info(f"    Expansion: {len(exp_y):,} ({100*len(exp_y)/n_total:.1f}%)")
    logger.info(f"  S features: {merged_X_s.shape}")
    logger.info(f"  V features: {merged_X_v.shape}")
    logger.info(f"  Regression target: [{merged_y_reg.min():.4f}, {merged_y_reg.max():.4f}]")
    logger.info(f"  Scale factor: {n_total / 20288:.2f}x original")


if __name__ == "__main__":
    main()
