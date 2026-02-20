"""Step 8: Batch VQI scoring for test splits.

Loads extracted features and trained RF models, computes VQI-S and VQI-V
scores for each file in a test split.

Usage:
    python scripts/compute_vqi_scores.py --split test_voxceleb1
    python scripts/compute_vqi_scores.py --split test_vctk
    python scripts/compute_vqi_scores.py --split test_cnceleb
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from vqi.prediction.random_forest import (
    load_model as load_model_s,
    get_selected_feature_names as get_selected_s,
    predict_scores_batch as predict_batch_s,
)
from vqi.prediction.random_forest_v import (
    load_model as load_model_v,
    get_selected_feature_names as get_selected_v,
    predict_scores_batch as predict_batch_v,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(_IMPL_DIR, "models")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
EVAL_V_DIR = os.path.join(DATA_DIR, "evaluation_v")
OUTPUT_DIR = os.path.join(DATA_DIR, "test_scores")


def main():
    parser = argparse.ArgumentParser(description="Compute VQI scores for test split")
    parser.add_argument("--split", required=True,
                        help="Split name (e.g., test_voxceleb1)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load split CSV (for filenames and speaker IDs)
    split_csv = os.path.join(SPLITS_DIR, f"{args.split}.csv")
    df = pd.read_csv(split_csv)
    logger.info(f"Split '{args.split}': {len(df)} files")

    # Load features
    feat_s_path = os.path.join(FEATURES_DIR, f"features_s_{args.split}.npy")
    feat_v_path = os.path.join(FEATURES_DIR, f"features_v_{args.split}.npy")

    if not os.path.exists(feat_s_path):
        logger.error(f"Features not found: {feat_s_path}")
        logger.error("Run extract_features_test.py first.")
        sys.exit(1)

    feat_s_all = np.load(feat_s_path)  # (N, 544)
    feat_v_all = np.load(feat_v_path)  # (N, 161)
    logger.info(f"Loaded features: S={feat_s_all.shape}, V={feat_v_all.shape}")

    # Load feature name mappings
    with open(os.path.join(FEATURES_DIR, "feature_names_s.json"), "r", encoding="utf-8") as f:
        all_names_s = json.load(f)
    with open(os.path.join(FEATURES_DIR, "feature_names_v.json"), "r", encoding="utf-8") as f:
        all_names_v = json.load(f)

    # Load selected features
    selected_s = get_selected_s(os.path.join(EVAL_DIR, "selected_features.txt"))
    selected_v = get_selected_v(os.path.join(EVAL_V_DIR, "selected_features.txt"))
    logger.info(f"Selected features: S={len(selected_s)}, V={len(selected_v)}")

    # Map selected names to column indices
    name_to_idx_s = {name: i for i, name in enumerate(all_names_s)}
    name_to_idx_v = {name: i for i, name in enumerate(all_names_v)}

    idx_s = [name_to_idx_s[n] for n in selected_s]
    idx_v = [name_to_idx_v[n] for n in selected_v]

    X_s = feat_s_all[:, idx_s]  # (N, N_selected)
    X_v = feat_v_all[:, idx_v]  # (N, N_selected_V)
    logger.info(f"Selected feature matrices: S={X_s.shape}, V={X_v.shape}")

    # Replace NaN/Inf with 0 for prediction
    nan_mask_s = ~np.isfinite(X_s)
    nan_mask_v = ~np.isfinite(X_v)
    if nan_mask_s.any():
        logger.warning(f"Replacing {nan_mask_s.sum()} NaN/Inf in VQI-S features")
        X_s[nan_mask_s] = 0.0
    if nan_mask_v.any():
        logger.warning(f"Replacing {nan_mask_v.sum()} NaN/Inf in VQI-V features")
        X_v[nan_mask_v] = 0.0

    # Load models
    model_s = load_model_s(os.path.join(MODELS_DIR, "vqi_rf_model.joblib"))
    model_v = load_model_v(os.path.join(MODELS_DIR, "vqi_v_rf_model.joblib"))

    # Predict scores
    t0 = time.time()
    scores_s = predict_batch_s(model_s, X_s)  # (N,) int [0-100]
    scores_v = predict_batch_v(model_v, X_v)  # (N,) int [0-100]
    predict_time = time.time() - t0
    logger.info(f"Prediction time: {predict_time:.2f}s ({predict_time/len(df)*1000:.2f}ms/file)")

    # Build output DataFrame
    out_df = pd.DataFrame({
        "filename": df["filename"],
        "speaker_id": df["speaker_id"],
        "vqi_s": scores_s,
        "vqi_v": scores_v,
    })

    output_path = os.path.join(OUTPUT_DIR, f"vqi_scores_{args.split}.csv")
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {output_path}")

    # Summary statistics
    logger.info("=" * 60)
    logger.info(f"VQI-S: min={scores_s.min()}, max={scores_s.max()}, "
                f"mean={scores_s.mean():.1f}, std={scores_s.std():.1f}")
    logger.info(f"VQI-V: min={scores_v.min()}, max={scores_v.max()}, "
                f"mean={scores_v.mean():.1f}, std={scores_v.std():.1f}")

    # Verify no NaN in scores
    assert not np.any(np.isnan(scores_s)), "NaN in VQI-S scores!"
    assert not np.any(np.isnan(scores_v)), "NaN in VQI-V scores!"
    logger.info("No NaN in scores - PASS")


if __name__ == "__main__":
    main()
