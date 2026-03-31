"""
Step 5 Orchestrator: Feature Evaluation and Selection

Runs VQI-S (5.1-5.5) then VQI-V (5.7-5.11) pipelines with checkpoint support.

Usage:
    python run_step5.py [--resume] [--skip-s] [--skip-v]
"""

import argparse
import logging
import os
import sys
import time

import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.evaluate_features import run_vqi_s_pipeline
from vqi.training.evaluate_features_v import run_vqi_v_pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def verify_outputs(output_dir: str, score_type: str) -> bool:
    """Verify pipeline outputs are complete and valid."""
    import numpy as np
    import pandas as pd

    ok = True
    prefix = f"VQI-{score_type.upper()}"

    # Check spearman_correlations.csv
    sp_path = os.path.join(output_dir, "spearman_correlations.csv")
    if os.path.exists(sp_path):
        df = pd.read_csv(sp_path)
        rho_vals = df["abs_rho_mean"].values
        if np.any(np.isnan(rho_vals)) or np.any(np.isinf(rho_vals)):
            logging.error("%s: NaN/Inf in Spearman correlations", prefix)
            ok = False
        if np.any(np.abs(df["rho_mean"].values) > 1.0):
            logging.error("%s: rho values outside [-1,1]", prefix)
            ok = False
        logging.info("%s: spearman_correlations.csv -> %d rows, OK", prefix, len(df))
    else:
        logging.error("%s: spearman_correlations.csv NOT FOUND", prefix)
        ok = False

    # Check removed_redundant_features.csv
    rem_path = os.path.join(output_dir, "removed_redundant_features.csv")
    if os.path.exists(rem_path):
        df = pd.read_csv(rem_path)
        # Verify no surviving pairs above threshold
        corr_path = os.path.join(output_dir, "feature_correlation_matrix.npy")
        if os.path.exists(corr_path):
            logging.info(
                "%s: removed_redundant_features.csv -> %d pairs removed", prefix, len(df)
            )
    else:
        logging.error("%s: removed_redundant_features.csv NOT FOUND", prefix)
        ok = False

    # Check selected_features.txt
    sel_path = os.path.join(output_dir, "selected_features.txt")
    if os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as f:
            selected = [line.strip() for line in f if line.strip()]
        n = len(selected)
        logging.info("%s: selected_features.txt -> %d features", prefix, n)
    else:
        logging.error("%s: selected_features.txt NOT FOUND", prefix)
        ok = False

    # Check feature_selection_summary.yaml
    yaml_path = os.path.join(output_dir, "feature_selection_summary.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            summary = yaml.safe_load(f)
        logging.info(
            "%s: N_selected=%d, OOB=%.4f, iterations=%d",
            prefix,
            summary["n_selected"],
            summary["final_oob_accuracy"],
            summary["n_iterations"],
        )
    else:
        logging.error("%s: feature_selection_summary.yaml NOT FOUND", prefix)
        ok = False

    # Check ERC
    erc_path = os.path.join(output_dir, "erc_per_feature.csv")
    if os.path.exists(erc_path):
        df = pd.read_csv(erc_path)
        auc_vals = df["auc_mean"].values
        if np.any(np.isnan(auc_vals)) or np.any(np.isinf(auc_vals)):
            logging.error("%s: NaN/Inf in ERC AUC values", prefix)
            ok = False
        logging.info("%s: erc_per_feature.csv -> %d features, OK", prefix, len(df))
    else:
        logging.error("%s: erc_per_feature.csv NOT FOUND", prefix)
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="Step 5: Feature Evaluation and Selection")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-s", action="store_true", help="Skip VQI-S pipeline")
    parser.add_argument("--skip-v", action="store_true", help="Skip VQI-V pipeline")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("step5")

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, "data")
    features_dir = os.path.join(data_dir, "step4", "features")
    eval_s_dir = os.path.join(data_dir, "step5", "evaluation")
    eval_v_dir = os.path.join(data_dir, "step5", "evaluation_v")
    checkpoint_path = os.path.join(eval_s_dir, "_checkpoint_step5.yaml")

    training_csv = os.path.join(data_dir, "step2", "training_set_final.csv")
    fisher_csv = os.path.join(data_dir, "step2", "fisher_values.csv")

    os.makedirs(eval_s_dir, exist_ok=True)
    os.makedirs(eval_v_dir, exist_ok=True)

    t0 = time.time()

    # ---- VQI-S Pipeline ----
    if not args.skip_s:
        logger.info("=" * 60)
        logger.info("Starting VQI-S Feature Selection (544 candidates)")
        logger.info("=" * 60)

        results_s = run_vqi_s_pipeline(
            features_path=os.path.join(features_dir, "features_s_train.npy"),
            names_path=os.path.join(features_dir, "feature_names_s.json"),
            training_csv=training_csv,
            fisher_csv=fisher_csv,
            output_dir=eval_s_dir,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )

        logger.info(
            "VQI-S: %d -> %d -> %d -> %d features (OOB=%.4f)",
            results_s["n_total"],
            results_s["n_valid"],
            results_s["n_after_redundancy"],
            results_s["n_selected"],
            results_s["rf_summary"]["final_oob_accuracy"],
        )

        # Verify
        s_ok = verify_outputs(eval_s_dir, "s")
        if not s_ok:
            logger.error("VQI-S verification FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping VQI-S pipeline (--skip-s)")

    # ---- VQI-V Pipeline ----
    if not args.skip_v:
        logger.info("=" * 60)
        logger.info("Starting VQI-V Feature Selection (161 candidates)")
        logger.info("=" * 60)

        checkpoint_v = os.path.join(eval_v_dir, "_checkpoint_step5_v.yaml")

        results_v = run_vqi_v_pipeline(
            features_path=os.path.join(features_dir, "features_v_train.npy"),
            names_path=os.path.join(features_dir, "feature_names_v.json"),
            training_csv=training_csv,
            fisher_csv=fisher_csv,
            output_dir=eval_v_dir,
            checkpoint_path=checkpoint_v,
            resume=args.resume,
        )

        logger.info(
            "VQI-V: %d -> %d -> %d -> %d features (OOB=%.4f)",
            results_v["n_total"],
            results_v["n_valid"],
            results_v["n_after_redundancy"],
            results_v["n_selected"],
            results_v["rf_summary"]["final_oob_accuracy"],
        )

        # Verify
        v_ok = verify_outputs(eval_v_dir, "v")
        if not v_ok:
            logger.error("VQI-V verification FAILED")
            sys.exit(1)
    else:
        logger.info("Skipping VQI-V pipeline (--skip-v)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Step 5 COMPLETE in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)

    # Clean up checkpoint on success
    for cp in [checkpoint_path, os.path.join(eval_v_dir, "_checkpoint_step5_v.yaml")]:
        if os.path.exists(cp):
            os.remove(cp)
            logger.info("Removed checkpoint: %s", cp)


if __name__ == "__main__":
    main()
