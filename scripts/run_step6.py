"""
Step 6 Orchestrator: Model Training

Runs VQI-S (6.1-6.6) then VQI-V (6.7-6.12) training pipelines with checkpoint support.

Usage:
    python run_step6.py [--resume] [--skip-s] [--skip-v]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.train_rf import run_vqi_s_pipeline
from vqi.training.train_rf_v import run_vqi_v_pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def verify_outputs(output_dir: str, model_path: str, score_type: str) -> bool:
    """Verify training outputs are complete and valid."""
    import joblib
    import pandas as pd

    ok = True
    prefix = f"VQI-{score_type.upper()}"

    # Check X_train.npy
    x_path = os.path.join(output_dir, "X_train.npy")
    if os.path.exists(x_path):
        X = np.load(x_path)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logging.error("%s: NaN/Inf in X_train", prefix)
            ok = False
        logging.info("%s: X_train.npy -> shape %s, OK", prefix, X.shape)
    else:
        logging.error("%s: X_train.npy NOT FOUND", prefix)
        ok = False

    # Check y_train.npy
    y_path = os.path.join(output_dir, "y_train.npy")
    if os.path.exists(y_path):
        y = np.load(y_path)
        unique = set(np.unique(y))
        if unique != {0, 1}:
            logging.error("%s: Unexpected labels: %s", prefix, unique)
            ok = False
        logging.info("%s: y_train.npy -> shape %s, labels %s, OK", prefix, y.shape, unique)
    else:
        logging.error("%s: y_train.npy NOT FOUND", prefix)
        ok = False

    # Check grid_search_results.csv
    gs_path = os.path.join(output_dir, "grid_search_results.csv")
    if os.path.exists(gs_path):
        df = pd.read_csv(gs_path)
        logging.info(
            "%s: grid_search_results.csv -> %d configs, best OOB_err=%.4f",
            prefix, len(df), df["oob_error"].min(),
        )
    else:
        logging.error("%s: grid_search_results.csv NOT FOUND", prefix)
        ok = False

    # Check model file
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        logging.info(
            "%s: Model loaded (%d trees, %d features)",
            prefix, clf.n_estimators, clf.n_features_in_,
        )
    else:
        logging.error("%s: Model NOT FOUND at %s", prefix, model_path)
        ok = False

    # Check training_metrics.yaml
    metrics_path = os.path.join(output_dir, "training_metrics.yaml")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        logging.info(
            "%s: OOB_err=%.4f, train_acc=%.4f",
            prefix, metrics["oob_error"], metrics["training_accuracy"],
        )
    else:
        logging.error("%s: training_metrics.yaml NOT FOUND", prefix)
        ok = False

    # Check feature_importances.csv
    imp_path = os.path.join(output_dir, "feature_importances.csv")
    if os.path.exists(imp_path):
        df = pd.read_csv(imp_path)
        if not np.all(df["importance"].values >= 0):
            logging.error("%s: Negative importance values", prefix)
            ok = False
        logging.info("%s: feature_importances.csv -> %d features, OK", prefix, len(df))
    else:
        logging.error("%s: feature_importances.csv NOT FOUND", prefix)
        ok = False

    # Check oob_convergence.csv
    conv_path = os.path.join(output_dir, "oob_convergence.csv")
    if os.path.exists(conv_path):
        df = pd.read_csv(conv_path)
        logging.info("%s: oob_convergence.csv -> %d points, OK", prefix, len(df))
    else:
        logging.error("%s: oob_convergence.csv NOT FOUND", prefix)
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="Step 6: Model Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-s", action="store_true", help="Skip VQI-S pipeline")
    parser.add_argument("--skip-v", action="store_true", help="Skip VQI-V pipeline")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("step6")

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, "data")
    features_dir = os.path.join(data_dir, "features")
    training_csv = os.path.join(data_dir, "training_set_final.csv")

    # VQI-S paths
    train_s_dir = os.path.join(data_dir, "training")
    model_s_path = os.path.join(PROJECT_ROOT, "models", "vqi_rf_model.joblib")

    # VQI-V paths
    train_v_dir = os.path.join(data_dir, "training_v")
    model_v_path = os.path.join(PROJECT_ROOT, "models", "vqi_v_rf_model.joblib")

    os.makedirs(train_s_dir, exist_ok=True)
    os.makedirs(train_v_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

    t0 = time.time()

    # ---- VQI-S Pipeline ----
    if not args.skip_s:
        logger.info("=" * 60)
        logger.info("Starting VQI-S Model Training (430 features)")
        logger.info("=" * 60)

        results_s = run_vqi_s_pipeline(
            features_npy=os.path.join(features_dir, "features_s_train.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_s.json"),
            selected_features_txt=os.path.join(data_dir, "evaluation", "selected_features.txt"),
            training_csv=training_csv,
            output_dir=train_s_dir,
            model_path=model_s_path,
            checkpoint_path=os.path.join(train_s_dir, "_checkpoint_step6_s.yaml"),
            resume=args.resume,
        )

        logger.info(
            "VQI-S: %d features, OOB_err=%.4f, train_acc=%.4f, best=%s",
            results_s["n_features"],
            results_s["oob_error"],
            results_s["training_accuracy"],
            results_s["best_params"],
        )

        # Verify
        s_ok = verify_outputs(train_s_dir, model_s_path, "s")
        if not s_ok:
            logger.error("VQI-S verification FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping VQI-S pipeline (--skip-s)")

    t_s = time.time() - t0

    # ---- VQI-V Pipeline ----
    if not args.skip_v:
        logger.info("=" * 60)
        logger.info("Starting VQI-V Model Training (133 features)")
        logger.info("=" * 60)

        results_v = run_vqi_v_pipeline(
            features_npy=os.path.join(features_dir, "features_v_train.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_v.json"),
            selected_features_txt=os.path.join(data_dir, "evaluation_v", "selected_features.txt"),
            training_csv=training_csv,
            output_dir=train_v_dir,
            model_path=model_v_path,
            checkpoint_path=os.path.join(train_v_dir, "_checkpoint_step6_v.yaml"),
            resume=args.resume,
        )

        logger.info(
            "VQI-V: %d features, OOB_err=%.4f, train_acc=%.4f, best=%s",
            results_v["n_features"],
            results_v["oob_error"],
            results_v["training_accuracy"],
            results_v["best_params"],
        )

        # Verify
        v_ok = verify_outputs(train_v_dir, model_v_path, "v")
        if not v_ok:
            logger.error("VQI-V verification FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping VQI-V pipeline (--skip-v)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Step 6 COMPLETE in %.1f seconds (%.1f min) [S: %.1f min, V: %.1f min]",
        elapsed, elapsed / 60,
        t_s / 60, (elapsed - t_s) / 60,
    )
    logger.info("=" * 60)

    # Clean up checkpoints on success
    for cp in [
        os.path.join(train_s_dir, "_checkpoint_step6_s.yaml"),
        os.path.join(train_v_dir, "_checkpoint_step6_v.yaml"),
    ]:
        if os.path.exists(cp):
            os.remove(cp)
            logger.info("Removed checkpoint: %s", cp)


if __name__ == "__main__":
    main()
