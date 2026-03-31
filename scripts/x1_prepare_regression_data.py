"""
Step X1.7: Prepare Regression Targets

Constructs continuous regression targets from genuine S-norm scores for all
training and validation samples. Targets are normalized to [0, 1] for both
VQI-S and VQI-V.

Targets:
  - Fused: mean(P1, P2, P3) genuine S-norm scores
  - Per-provider: P1, P2, P3 individually (ablation)

Usage:
    python scripts/x1_prepare_regression_data.py
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REG_DIR = os.path.join(DATA_DIR, "x1_regression")

logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def prepare_train_targets():
    """Build regression targets for training set (20,288 samples).

    Uses score_P1/P2/P3 columns from training_set_final.csv which are
    genuine S-norm scores already aligned with X_train.npy row order.
    """
    logger.info("Loading training set manifest...")
    df = pd.read_csv(os.path.join(DATA_DIR, "step2", "training_set_final.csv"))
    assert len(df) == 20288, f"Expected 20288, got {len(df)}"

    # Extract per-provider scores
    p1 = df["score_P1"].values.astype(np.float64)
    p2 = df["score_P2"].values.astype(np.float64)
    p3 = df["score_P3"].values.astype(np.float64)

    assert not np.any(np.isnan(p1)), "NaN in train P1 scores"
    assert not np.any(np.isnan(p2)), "NaN in train P2 scores"
    assert not np.any(np.isnan(p3)), "NaN in train P3 scores"

    fused = np.mean(np.column_stack([p1, p2, p3]), axis=1)

    logger.info("  P1: [%.3f, %.3f], mean=%.3f", p1.min(), p1.max(), p1.mean())
    logger.info("  P2: [%.3f, %.3f], mean=%.3f", p2.min(), p2.max(), p2.mean())
    logger.info("  P3: [%.3f, %.3f], mean=%.3f", p3.min(), p3.max(), p3.mean())
    logger.info("  Fused: [%.3f, %.3f], mean=%.3f", fused.min(), fused.max(), fused.mean())

    return {
        "fused": fused,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "filenames": df["filename"].values,
        "labels": df["label"].values,
    }


def prepare_val_targets():
    """Build regression targets for validation set (50,000 samples).

    Merges scores from step1/provider_scores/scores_val_set_P{1,2,3}_*.csv
    and aligns with validation feature matrix row order.
    """
    logger.info("Loading validation set...")

    # Load validation manifest for filenames
    val_manifest = pd.read_csv(os.path.join(DATA_DIR, "step2", "validation_set.csv"))
    filenames = val_manifest["filename"].values

    # Load provider scores for validation set
    scores = {}
    for prov, suffix in [("P1", "P1_ECAPA_ecapa"), ("P2", "P2_RESNET_resnet"), ("P3", "P3_ECAPA2_ecapa2")]:
        csv_path = os.path.join(DATA_DIR, "step1", "provider_scores", f"scores_val_set_{suffix}.csv")
        df_prov = pd.read_csv(csv_path)
        # Align by row_idx (same order as validation features)
        scores[prov] = df_prov["genuine_norm"].values

    p1 = scores["P1"]
    p2 = scores["P2"]
    p3 = scores["P3"]

    # Some validation samples may have NaN scores (no same-speaker pairs)
    nan_mask = np.isnan(p1) | np.isnan(p2) | np.isnan(p3)
    n_nan = int(nan_mask.sum())
    logger.info("  Val NaN count: %d / %d (%.1f%%)", n_nan, len(p1), 100.0 * n_nan / len(p1))

    fused = np.mean(np.column_stack([p1, p2, p3]), axis=1)

    logger.info("  P1: [%.3f, %.3f] (excl NaN)", np.nanmin(p1), np.nanmax(p1))
    logger.info("  Fused: [%.3f, %.3f] (excl NaN)", np.nanmin(fused), np.nanmax(fused))

    return {
        "fused": fused,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "nan_mask": nan_mask,
        "filenames": filenames,
    }


def normalize_targets(train_raw, val_raw):
    """Min-max normalize to [0, 1] using training set statistics.

    Returns normalized arrays and scaler parameters.
    """
    y_min = float(np.nanmin(train_raw))
    y_max = float(np.nanmax(train_raw))
    assert y_max > y_min, f"Degenerate range: [{y_min}, {y_max}]"

    train_norm = np.clip((train_raw - y_min) / (y_max - y_min), 0.0, 1.0)
    val_norm = np.clip((val_raw - y_min) / (y_max - y_min), 0.0, 1.0)

    return train_norm, val_norm, {"y_min": y_min, "y_max": y_max}


def compute_stats(name, arr):
    """Compute summary statistics for a target array."""
    valid = arr[~np.isnan(arr)]
    return {
        "name": name,
        "count": len(valid),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "skewness": float(stats.skew(valid)),
        "kurtosis": float(stats.kurtosis(valid)),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(REG_DIR, exist_ok=True)

    # 1. Prepare raw targets
    logger.info("=" * 60)
    logger.info("X1.7: Regression Target Preparation")
    logger.info("=" * 60)

    train_data = prepare_train_targets()
    val_data = prepare_val_targets()
    _flush()

    # 2. Normalize and save
    scaler_info = {}
    all_stats = []

    for target_name in ["fused", "p1", "p2", "p3"]:
        logger.info("Processing target: %s", target_name)

        train_raw = train_data[target_name]
        val_raw = val_data[target_name]

        # Raw stats
        all_stats.append(compute_stats(f"train_{target_name}_raw", train_raw))
        all_stats.append(compute_stats(f"val_{target_name}_raw", val_raw))

        # Normalize
        train_norm, val_norm, params = normalize_targets(train_raw, val_raw)
        scaler_info[target_name] = params

        # Save
        np.save(os.path.join(REG_DIR, f"y_reg_train_{target_name}.npy"), train_norm.astype(np.float32))
        np.save(os.path.join(REG_DIR, f"y_reg_val_{target_name}.npy"), val_norm.astype(np.float32))

        # Normalized stats
        all_stats.append(compute_stats(f"train_{target_name}_norm", train_norm))
        all_stats.append(compute_stats(f"val_{target_name}_norm", val_norm))

        logger.info("  Train norm: [%.4f, %.4f], mean=%.4f, std=%.4f",
                    train_norm.min(), train_norm.max(), train_norm.mean(), train_norm.std())
        logger.info("  Val norm: [%.4f, %.4f], mean=%.4f (excl NaN)",
                    np.nanmin(val_norm), np.nanmax(val_norm), np.nanmean(val_norm))
        _flush()

    # 3. Save scaler
    scaler_path = os.path.join(REG_DIR, "y_reg_scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=2)
    logger.info("Saved scaler: %s", scaler_path)

    # 4. Save stats
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(REG_DIR, "regression_target_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info("Saved stats: %s", stats_path)

    # 5. Save validation NaN mask (for filtering labeled samples)
    np.save(os.path.join(REG_DIR, "val_nan_mask.npy"), val_data["nan_mask"])

    # 6. Verification
    logger.info("")
    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    # Check fused has lower variance than any single provider (averaging smooths noise)
    fused_std = train_data["fused"].std()
    p1_std = train_data["p1"].std()
    p2_std = train_data["p2"].std()
    p3_std = train_data["p3"].std()
    logger.info("Std check — Fused: %.4f, P1: %.4f, P2: %.4f, P3: %.4f", fused_std, p1_std, p2_std, p3_std)
    if fused_std < min(p1_std, p2_std, p3_std):
        logger.info("  PASS: Fused has lower variance than all individual providers")
    else:
        logger.warning("  WARN: Fused std not strictly lowest (expected from averaging)")

    # Check all normalized targets in [0, 1]
    for target_name in ["fused", "p1", "p2", "p3"]:
        t = np.load(os.path.join(REG_DIR, f"y_reg_train_{target_name}.npy"))
        assert t.min() >= 0.0 and t.max() <= 1.0, f"Train {target_name} out of [0,1]"
        assert not np.any(np.isnan(t)), f"NaN in train {target_name}"

    # Check shapes match X_train
    X_train = np.load(os.path.join(DATA_DIR, "step6", "full_feature", "training", "X_train.npy"))
    y_reg = np.load(os.path.join(REG_DIR, "y_reg_train_fused.npy"))
    assert X_train.shape[0] == y_reg.shape[0], \
        f"Shape mismatch: X_train={X_train.shape[0]}, y_reg={y_reg.shape[0]}"
    logger.info("Shape check: X_train=%s, y_reg=%s — MATCH", X_train.shape, y_reg.shape)

    # Print correlation between binary labels and regression targets
    y_bin = train_data["labels"]
    y_fused_norm = np.load(os.path.join(REG_DIR, "y_reg_train_fused.npy"))
    corr = np.corrcoef(y_bin, y_fused_norm)[0, 1]
    logger.info("Correlation (binary label vs fused reg target): %.4f", corr)

    # Summary table
    logger.info("")
    logger.info("Output files:")
    for f in sorted(os.listdir(REG_DIR)):
        fpath = os.path.join(REG_DIR, f)
        size = os.path.getsize(fpath)
        logger.info("  %s (%s)", f, f"{size:,} bytes")

    logger.info("")
    logger.info("=" * 60)
    logger.info("X1.7 Regression Target Preparation COMPLETE")
    logger.info("=" * 60)
