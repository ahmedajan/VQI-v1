"""
Step 7.16: Run VQI-S x VQI-V Feature-Level Cross-Analysis

Orchestrates 6 experiments (A-F) with checkpoint/resume support.

Usage:
    python scripts/run_cross_analysis.py [--resume]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.cross_analysis_sv import (
    experiment_a_combined_model,
    experiment_b_cross_correlation,
    experiment_c_importance_redistribution,
    experiment_d_ablation,
    experiment_e_cross_prediction,
    experiment_f_validation_comparison,
    compute_verdict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "step7", "cross_validation", "cross_analysis")

X_S_PATH = os.path.join(DATA_DIR, "step6", "full_feature", "training", "X_train.npy")
X_V_PATH = os.path.join(DATA_DIR, "step6", "full_feature", "training_v", "X_train.npy")
Y_PATH = os.path.join(DATA_DIR, "step6", "full_feature", "training", "y_train.npy")
S_NAMES_PATH = os.path.join(DATA_DIR, "step6", "full_feature", "training", "feature_names.txt")
V_NAMES_PATH = os.path.join(DATA_DIR, "step6", "full_feature", "training_v", "feature_names.txt")
IMP_S_CSV = os.path.join(DATA_DIR, "step6", "full_feature", "training", "feature_importances.csv")
IMP_V_CSV = os.path.join(DATA_DIR, "step6", "full_feature", "training_v", "feature_importances.csv")
MODEL_S_PATH = os.path.join(PROJECT_ROOT, "models", "vqi_rf_model.joblib")
MODEL_V_PATH = os.path.join(PROJECT_ROOT, "models", "vqi_v_rf_model.joblib")

# Validation data paths
FEATURES_S_VAL = os.path.join(DATA_DIR, "step4", "features", "features_s_val.npy")
FEATURES_V_VAL = os.path.join(DATA_DIR, "step4", "features", "features_v_val.npy")
FEATURE_NAMES_S_JSON = os.path.join(DATA_DIR, "step4", "features", "feature_names_s.json")
FEATURE_NAMES_V_JSON = os.path.join(DATA_DIR, "step4", "features", "feature_names_v.json")
SELECTED_S_TXT = os.path.join(DATA_DIR, "step5", "evaluation", "selected_features.txt")
SELECTED_V_TXT = os.path.join(DATA_DIR, "step5", "evaluation_v", "selected_features.txt")

# Optional: validation labels
THRESHOLDS_YAML = os.path.join(DATA_DIR, "step2", "label_thresholds.yaml")
VALIDATION_CSV = os.path.join(DATA_DIR, "step2", "validation_set.csv")
PROVIDER_SCORES_DIR = os.path.join(DATA_DIR, "step1", "provider_scores")

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "_checkpoint_cross_analysis.yaml")

STAGES = ["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f", "verdict"]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 7.16: VQI-S x VQI-V Cross-Analysis")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(OUTPUT_DIR, "cross_analysis.log"),
                mode="a",
                encoding="utf-8",
            ),
        ],
    )

    logger.info("=" * 70)
    logger.info("Step 7.16: VQI-S x VQI-V Feature-Level Cross-Analysis")
    logger.info("=" * 70)

    # Load checkpoint
    state = {}
    if args.resume:
        state = _load_checkpoint(CHECKPOINT_PATH)
        if state:
            logger.info("Resuming from checkpoint: completed=%s", state.get("completed", []))
        else:
            logger.info("No checkpoint found, starting fresh")
    completed = set(state.get("completed", []))

    t_start = time.time()

    # --- Load training data ---
    logger.info("Loading training data...")
    X_s = np.load(X_S_PATH)
    X_v = np.load(X_V_PATH)
    y = np.load(Y_PATH)
    with open(S_NAMES_PATH, "r", encoding="utf-8") as f:
        s_names = [l.strip() for l in f if l.strip()]
    with open(V_NAMES_PATH, "r", encoding="utf-8") as f:
        v_names = [l.strip() for l in f if l.strip()]

    logger.info(
        "Data loaded: X_s=%s, X_v=%s, y=%s",
        X_s.shape, X_v.shape, y.shape,
    )
    assert X_s.shape[0] == X_v.shape[0] == len(y)
    assert X_s.shape[1] == len(s_names) == 430
    assert X_v.shape[1] == len(v_names) == 133

    # ========================================================
    # Experiment A: Combined Model Performance
    # ========================================================
    if "exp_a" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 1/7] Experiment A: Combined Model Performance")
        logger.info("=" * 50)
        t0 = time.time()
        exp_a = experiment_a_combined_model(
            X_s, X_v, y, s_names, v_names, OUTPUT_DIR,
            MODEL_S_PATH, MODEL_V_PATH,
        )
        elapsed = time.time() - t0
        logger.info("[Stage 1/7] Done in %.1f seconds", elapsed)
        completed.add("exp_a")
        state["completed"] = sorted(completed)
        state["exp_a_model_path"] = exp_a["model_path"]
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 1/7] Experiment A: loaded from checkpoint")

    # ========================================================
    # Experiment B: Cross-Correlation Matrix
    # ========================================================
    if "exp_b" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 2/7] Experiment B: Cross-Correlation Matrix")
        logger.info("=" * 50)
        t0 = time.time()
        exp_b = experiment_b_cross_correlation(X_s, X_v, s_names, v_names, OUTPUT_DIR)
        elapsed = time.time() - t0
        logger.info("[Stage 2/7] Done in %.1f seconds", elapsed)
        completed.add("exp_b")
        state["completed"] = sorted(completed)
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 2/7] Experiment B: loaded from checkpoint")

    # ========================================================
    # Experiment C: Feature Importance Redistribution
    # ========================================================
    if "exp_c" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 3/7] Experiment C: Importance Redistribution")
        logger.info("=" * 50)
        import joblib
        combined_model_path = state.get(
            "exp_a_model_path",
            os.path.join(PROJECT_ROOT, "models", "vqi_combined_rf_model.joblib"),
        )
        clf_combined = joblib.load(combined_model_path)
        combined_names = s_names + v_names
        t0 = time.time()
        exp_c = experiment_c_importance_redistribution(
            clf_combined, combined_names, s_names, v_names,
            IMP_S_CSV, IMP_V_CSV, OUTPUT_DIR,
        )
        elapsed = time.time() - t0
        logger.info("[Stage 3/7] Done in %.1f seconds", elapsed)
        completed.add("exp_c")
        state["completed"] = sorted(completed)
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 3/7] Experiment C: loaded from checkpoint")

    # ========================================================
    # Experiment D: Ablation / Unique Contribution
    # ========================================================
    if "exp_d" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 4/7] Experiment D: Ablation / Unique Contribution")
        logger.info("=" * 50)
        import joblib
        combined_model_path = state.get(
            "exp_a_model_path",
            os.path.join(PROJECT_ROOT, "models", "vqi_combined_rf_model.joblib"),
        )
        clf_combined = joblib.load(combined_model_path)
        combined_names = s_names + v_names
        t0 = time.time()
        exp_d = experiment_d_ablation(
            X_s, X_v, y, clf_combined, combined_names,
            s_names, v_names, OUTPUT_DIR,
        )
        elapsed = time.time() - t0
        logger.info("[Stage 4/7] Done in %.1f seconds", elapsed)
        completed.add("exp_d")
        state["completed"] = sorted(completed)
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 4/7] Experiment D: loaded from checkpoint")

    # ========================================================
    # Experiment E: Cross-Prediction
    # ========================================================
    if "exp_e" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 5/7] Experiment E: Cross-Prediction")
        logger.info("=" * 50)
        t0 = time.time()
        exp_e = experiment_e_cross_prediction(
            X_s, X_v, y, MODEL_S_PATH, MODEL_V_PATH, OUTPUT_DIR,
        )
        elapsed = time.time() - t0
        logger.info("[Stage 5/7] Done in %.1f seconds", elapsed)
        completed.add("exp_e")
        state["completed"] = sorted(completed)
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 5/7] Experiment E: loaded from checkpoint")

    # ========================================================
    # Experiment F: Validation Set Comparison
    # ========================================================
    if "exp_f" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 6/7] Experiment F: Validation Set Comparison")
        logger.info("=" * 50)
        combined_model_path = state.get(
            "exp_a_model_path",
            os.path.join(PROJECT_ROOT, "models", "vqi_combined_rf_model.joblib"),
        )
        t0 = time.time()
        exp_f = experiment_f_validation_comparison(
            features_s_val_npy=FEATURES_S_VAL,
            features_v_val_npy=FEATURES_V_VAL,
            feature_names_s_json=FEATURE_NAMES_S_JSON,
            feature_names_v_json=FEATURE_NAMES_V_JSON,
            selected_s_txt=SELECTED_S_TXT,
            selected_v_txt=SELECTED_V_TXT,
            model_s_path=MODEL_S_PATH,
            model_v_path=MODEL_V_PATH,
            combined_model_path=combined_model_path,
            s_names=s_names,
            v_names=v_names,
            output_dir=OUTPUT_DIR,
            thresholds_yaml=THRESHOLDS_YAML if os.path.exists(THRESHOLDS_YAML) else None,
            validation_csv=VALIDATION_CSV if os.path.exists(VALIDATION_CSV) else None,
            provider_scores_dir=PROVIDER_SCORES_DIR if os.path.isdir(PROVIDER_SCORES_DIR) else None,
        )
        elapsed = time.time() - t0
        logger.info("[Stage 6/7] Done in %.1f seconds", elapsed)
        completed.add("exp_f")
        state["completed"] = sorted(completed)
        _save_checkpoint(CHECKPOINT_PATH, state)
    else:
        logger.info("[Stage 6/7] Experiment F: loaded from checkpoint")

    # ========================================================
    # Verdict
    # ========================================================
    if "verdict" not in completed:
        logger.info("=" * 50)
        logger.info("[Stage 7/7] Computing verdict...")
        logger.info("=" * 50)

        # Load experiment results from saved files
        with open(os.path.join(OUTPUT_DIR, "combined_training_metrics.yaml"), "r", encoding="utf-8") as f:
            exp_a_metrics = yaml.safe_load(f)
        with open(os.path.join(OUTPUT_DIR, "ablation_results.yaml"), "r", encoding="utf-8") as f:
            exp_d_data = yaml.safe_load(f)
        with open(os.path.join(OUTPUT_DIR, "cross_prediction.yaml"), "r", encoding="utf-8") as f:
            exp_e_data = yaml.safe_load(f)

        verdict = compute_verdict(exp_a_metrics, exp_d_data, exp_e_data)

        with open(os.path.join(OUTPUT_DIR, "verdict.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(verdict, f, default_flow_style=False, sort_keys=False)

        completed.add("verdict")
        state["completed"] = sorted(completed)
        state["verdict"] = verdict["verdict"]
        _save_checkpoint(CHECKPOINT_PATH, state)

        logger.info("VERDICT: %s", verdict["verdict"])
        logger.info("  %s", verdict["explanation"])
    else:
        logger.info("[Stage 7/7] Verdict: loaded from checkpoint")

    # ========================================================
    # Summary
    # ========================================================
    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Step 7.16 complete in %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    # Remove checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint removed (clean completion)")


if __name__ == "__main__":
    main()
