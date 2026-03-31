"""
Step 7 Orchestrator: Model Validation

Runs the full validation pipeline:
  Phase A: Embedding extraction + provider scores for val_set (via subprocess)
  Phase B: Merge scores + compute labels + predict VQI scores
  Phase C: VQI-S validation analysis
  Phase D: VQI-V validation analysis
  Phase E: Dual-score scatter analysis

Usage:
    python run_step7.py [--resume] [--skip-embeddings] [--skip-s] [--skip-v]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.validate_model import (
    run_vqi_s_validation,
    merge_provider_scores,
    compute_validation_labels,
    predict_vqi_scores,
    compute_quadrant_analysis,
    PROVIDER_NAMES,
)
from vqi.training.validate_model_v import run_vqi_v_validation


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def check_embeddings_exist(embeddings_dir: str, split: str, providers: list) -> bool:
    """Check if embeddings already exist for all providers."""
    for pn in providers:
        npy_path = os.path.join(embeddings_dir, f"{split}_{pn}.npy")
        if not os.path.exists(npy_path):
            return False
    return True


def check_scores_exist(scores_dir: str, split: str, providers: list) -> bool:
    """Check if provider score CSVs exist."""
    short_map = {"P1_ECAPA": "ecapa", "P2_RESNET": "resnet", "P3_ECAPA2": "ecapa2"}
    for pn in providers:
        csv_path = os.path.join(scores_dir, f"scores_{split}_{pn}_{short_map[pn]}.csv")
        if not os.path.exists(csv_path):
            return False
    return True


def run_embedding_extraction(scripts_dir: str, resume: bool = False):
    """Run extract_embeddings.py for val_set via subprocess."""
    cmd = [
        sys.executable,
        os.path.join(scripts_dir, "extract_embeddings.py"),
        "--split", "val_set",
        "--providers", "P1_ECAPA", "P2_RESNET", "P3_ECAPA2",
    ]
    if resume:
        cmd.append("--resume")

    logging.getLogger("step7").info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=os.path.dirname(scripts_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Embedding extraction failed with code {result.returncode}")


def run_score_computation(scripts_dir: str, resume: bool = False):
    """Run compute_scores.py for val_set via subprocess."""
    cmd = [
        sys.executable,
        os.path.join(scripts_dir, "compute_scores.py"),
        "--split", "val_set",
        "--providers", "P1_ECAPA", "P2_RESNET", "P3_ECAPA2",
    ]
    if resume:
        cmd.append("--resume")

    logging.getLogger("step7").info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=os.path.dirname(scripts_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Score computation failed with code {result.returncode}")


def verify_validation_outputs(output_dir: str, score_type: str) -> bool:
    """Verify validation outputs for a score type."""
    ok = True
    prefix = f"VQI-{score_type.upper()}"
    logger = logging.getLogger("step7")

    suffix = "_v" if score_type == "v" else ""

    # Check results CSV
    csv_path = os.path.join(output_dir, f"validation_results{suffix}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        score_col = f"vqi_{score_type}_score"
        if score_col in df.columns:
            scores = df[score_col].values
            if np.any(np.isnan(scores)):
                logger.error("%s: NaN in VQI scores", prefix)
                ok = False
            if scores.min() < 0 or scores.max() > 100:
                logger.error("%s: Scores out of [0,100] range", prefix)
                ok = False
            logger.info(
                "%s: validation_results%s.csv -> %d rows, scores [%d, %d], mean=%.1f",
                prefix, suffix, len(df), scores.min(), scores.max(), scores.mean(),
            )
        else:
            logger.error("%s: Missing column %s", prefix, score_col)
            ok = False
    else:
        logger.error("%s: validation_results%s.csv NOT FOUND", prefix, suffix)
        ok = False

    # Check metrics YAML
    yaml_path = os.path.join(output_dir, f"validation_metrics{suffix}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        logger.info(
            "%s: Acc=%.4f, F1=%.4f, AUC=%.4f",
            prefix, metrics.get("accuracy", 0), metrics.get("f1_score", 0),
            metrics.get("auc_roc", 0),
        )
    else:
        logger.error("%s: validation_metrics%s.yaml NOT FOUND", prefix, suffix)
        ok = False

    return ok


def run_dual_score_analysis(
    val_s_dir: str,
    val_v_dir: str,
    provider_scores_dir: str,
    thresholds_yaml: str,
    output_dir: str,
    split_name: str = "val_set",
):
    """Run 7.14: dual-score scatter analysis."""
    logger = logging.getLogger("step7")
    logger.info("=" * 60)
    logger.info("Phase E: Dual-Score Analysis (7.14)")
    logger.info("=" * 60)

    # Load VQI-S results
    s_df = pd.read_csv(os.path.join(val_s_dir, "validation_results.csv"))
    vqi_s_scores = s_df["vqi_s_score"].values.astype(int)

    # Load VQI-V results
    v_df = pd.read_csv(os.path.join(val_v_dir, "validation_results_v.csv"))
    vqi_v_scores = v_df["vqi_v_score"].values.astype(int)

    # Labels from the S validation (same for both)
    labels = s_df["label"].values

    # Load genuine scores for per-quadrant stats
    genuine_scores = {}
    short_map = {"P1_ECAPA": "ecapa", "P2_RESNET": "resnet", "P3_ECAPA2": "ecapa2"}
    for pn in PROVIDER_NAMES:
        col = f"score_{pn}"
        if col in s_df.columns:
            genuine_scores[pn] = s_df[col].values.astype(np.float32)

    # Determine thresholds (50, or Youden's J if differs by >5)
    threshold_s = 50
    threshold_v = 50

    # Check Youden's J from VQI-S metrics
    s_metrics_path = os.path.join(val_s_dir, "validation_metrics.yaml")
    v_metrics_path = os.path.join(val_v_dir, "validation_metrics_v.yaml")

    if os.path.exists(s_metrics_path):
        with open(s_metrics_path, "r", encoding="utf-8") as f:
            s_metrics = yaml.safe_load(f)
        youden_s = s_metrics.get("youden_j_threshold", 50)
        if abs(youden_s - 50) > 5:
            threshold_s = int(round(youden_s))
            logger.info("Using Youden's J for VQI-S threshold: %d", threshold_s)

    if os.path.exists(v_metrics_path):
        with open(v_metrics_path, "r", encoding="utf-8") as f:
            v_metrics = yaml.safe_load(f)
        youden_v = v_metrics.get("youden_j_threshold", 50)
        if abs(youden_v - 50) > 5:
            threshold_v = int(round(youden_v))
            logger.info("Using Youden's J for VQI-V threshold: %d", threshold_v)

    # Compute quadrant analysis
    quad_df = compute_quadrant_analysis(
        vqi_s_scores, vqi_v_scores, labels, genuine_scores,
        threshold_s, threshold_v,
    )

    os.makedirs(output_dir, exist_ok=True)
    quad_csv = os.path.join(output_dir, "quadrant_analysis.csv")
    quad_df.to_csv(quad_csv, index=False, encoding="utf-8")
    logger.info("Quadrant analysis saved to %s", quad_csv)

    # Save combined dual-score data for visualization
    dual_df = pd.DataFrame({
        "filename": s_df["filename"],
        "speaker_id": s_df["speaker_id"],
        "vqi_s_score": vqi_s_scores,
        "vqi_v_score": vqi_v_scores,
        "label": labels,
    })
    for pn in PROVIDER_NAMES:
        col = f"score_{pn}"
        if col in s_df.columns:
            dual_df[col] = s_df[col]

    dual_csv = os.path.join(output_dir, "dual_score_data.csv")
    dual_df.to_csv(dual_csv, index=False, encoding="utf-8")
    logger.info("Dual-score data saved to %s (%d rows)", dual_csv, len(dual_df))

    # Save threshold info
    thresh_info = {
        "threshold_s": threshold_s,
        "threshold_v": threshold_v,
        "method_s": "youden_j" if threshold_s != 50 else "midpoint",
        "method_v": "youden_j" if threshold_v != 50 else "midpoint",
    }
    thresh_path = os.path.join(output_dir, "dual_score_thresholds.yaml")
    with open(thresh_path, "w", encoding="utf-8") as f:
        yaml.dump(thresh_info, f, default_flow_style=False, sort_keys=False)

    # Acceptance checks
    q1_rate = quad_df.loc[quad_df["quadrant"].str.contains("Q1"), "class1_rate"].values
    q3_rate = quad_df.loc[quad_df["quadrant"].str.contains("Q3"), "class1_rate"].values

    if len(q1_rate) > 0 and len(q3_rate) > 0:
        q1_ok = not np.isnan(q1_rate[0])
        q3_ok = not np.isnan(q3_rate[0])
        if q1_ok and q3_ok:
            check = q1_rate[0] > q3_rate[0]
            logger.info(
                "Q1 Class1 rate (%.4f) > Q3 Class1 rate (%.4f) -> %s",
                q1_rate[0], q3_rate[0], "PASS" if check else "FAIL",
            )

    return quad_df, thresh_info


def main():
    parser = argparse.ArgumentParser(description="Step 7: Model Validation")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding extraction (assume already done)")
    parser.add_argument("--skip-scores", action="store_true",
                        help="Skip score computation (assume already done)")
    parser.add_argument("--skip-s", action="store_true", help="Skip VQI-S validation")
    parser.add_argument("--skip-v", action="store_true", help="Skip VQI-V validation")
    parser.add_argument("--skip-dual", action="store_true", help="Skip dual-score analysis")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("step7")

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, "data")
    features_dir = os.path.join(data_dir, "step4", "features")
    embeddings_dir = os.path.join(data_dir, "step1", "embeddings")
    scores_dir = os.path.join(data_dir, "step1", "provider_scores")
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")

    validation_csv = os.path.join(data_dir, "step2", "validation_set.csv")
    thresholds_yaml = os.path.join(data_dir, "step2", "label_thresholds.yaml")

    # Output dirs
    val_s_dir = os.path.join(data_dir, "step7", "full_feature", "validation")
    val_v_dir = os.path.join(data_dir, "step7", "full_feature", "validation_v")
    os.makedirs(val_s_dir, exist_ok=True)
    os.makedirs(val_v_dir, exist_ok=True)

    t0 = time.time()

    # ---- Phase A: Embedding extraction + score computation ----
    providers = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]

    if not args.skip_embeddings:
        if check_embeddings_exist(embeddings_dir, "val_set", providers):
            logger.info("Phase A.1: val_set embeddings already exist, skipping extraction")
        else:
            logger.info("=" * 60)
            logger.info("Phase A.1: Extracting val_set embeddings (~45 min)")
            logger.info("=" * 60)
            run_embedding_extraction(scripts_dir, resume=args.resume)
    else:
        logger.info("Phase A.1: Skipping embedding extraction (--skip-embeddings)")

    if not args.skip_scores:
        if check_scores_exist(scores_dir, "val_set", providers):
            logger.info("Phase A.2: val_set scores already exist, skipping computation")
        else:
            logger.info("=" * 60)
            logger.info("Phase A.2: Computing val_set provider scores (~5 min)")
            logger.info("=" * 60)
            run_score_computation(scripts_dir, resume=args.resume)
    else:
        logger.info("Phase A.2: Skipping score computation (--skip-scores)")

    t_a = time.time() - t0

    # ---- Phase B-C: VQI-S Validation ----
    results_s = None
    if not args.skip_s:
        logger.info("=" * 60)
        logger.info("Phase B-C: VQI-S Validation (7.1-7.7)")
        logger.info("=" * 60)

        results_s = run_vqi_s_validation(
            validation_csv=validation_csv,
            features_npy=os.path.join(features_dir, "features_s_val.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_s.json"),
            selected_features_txt=os.path.join(data_dir, "step5", "evaluation", "selected_features.txt"),
            model_path=os.path.join(PROJECT_ROOT, "models", "vqi_rf_model.joblib"),
            training_dir=os.path.join(data_dir, "step6", "full_feature", "training"),
            provider_scores_dir=scores_dir,
            thresholds_yaml=thresholds_yaml,
            output_dir=val_s_dir,
            checkpoint_path=os.path.join(val_s_dir, "_checkpoint_step7_s.yaml"),
            split_name="val_set",
            resume=args.resume,
        )

        s_ok = verify_validation_outputs(val_s_dir, "s")
        if not s_ok:
            logger.error("VQI-S validation verification FAILED")
            sys.exit(1)

        logger.info("VQI-S Results: Acc=%.4f, AUC=%.4f, CDF shift=%s",
                     results_s.get("confusion", {}).get("accuracy", 0),
                     results_s.get("confusion", {}).get("auc_roc", 0),
                     results_s.get("all_cdf_pass", "N/A"))
    else:
        logger.info("Skipping VQI-S validation (--skip-s)")

    t_s = time.time() - t0 - t_a

    # ---- Phase D: VQI-V Validation ----
    results_v = None
    if not args.skip_v:
        logger.info("=" * 60)
        logger.info("Phase D: VQI-V Validation (7.8-7.15)")
        logger.info("=" * 60)

        results_v = run_vqi_v_validation(
            validation_csv=validation_csv,
            features_npy=os.path.join(features_dir, "features_v_val.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_v.json"),
            selected_features_txt=os.path.join(data_dir, "step5", "evaluation_v", "selected_features.txt"),
            model_path=os.path.join(PROJECT_ROOT, "models", "vqi_v_rf_model.joblib"),
            training_dir=os.path.join(data_dir, "step6", "full_feature", "training_v"),
            provider_scores_dir=scores_dir,
            thresholds_yaml=thresholds_yaml,
            output_dir=val_v_dir,
            checkpoint_path=os.path.join(val_v_dir, "_checkpoint_step7_v.yaml"),
            split_name="val_set",
            resume=args.resume,
        )

        v_ok = verify_validation_outputs(val_v_dir, "v")
        if not v_ok:
            logger.error("VQI-V validation verification FAILED")
            sys.exit(1)

        logger.info("VQI-V Results: Acc=%.4f, AUC=%.4f, CDF shift=%s",
                     results_v.get("confusion", {}).get("accuracy", 0),
                     results_v.get("confusion", {}).get("auc_roc", 0),
                     results_v.get("all_cdf_pass", "N/A"))
    else:
        logger.info("Skipping VQI-V validation (--skip-v)")

    t_v = time.time() - t0 - t_a - t_s

    # ---- Phase E: Dual-score analysis ----
    if not args.skip_dual and not args.skip_s and not args.skip_v:
        quad_df, thresh_info = run_dual_score_analysis(
            val_s_dir=val_s_dir,
            val_v_dir=val_v_dir,
            provider_scores_dir=scores_dir,
            thresholds_yaml=thresholds_yaml,
            output_dir=val_s_dir,  # Save in validation/ dir
            split_name="val_set",
        )
    elif not args.skip_dual:
        logger.info("Skipping dual-score analysis (requires both S and V)")

    # ---- Summary ----
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Step 7 COMPLETE in %.1f seconds (%.1f min) "
        "[A: %.1f min, S: %.1f min, V: %.1f min]",
        elapsed, elapsed / 60,
        t_a / 60, t_s / 60, t_v / 60,
    )
    logger.info("=" * 60)

    if results_s:
        logger.info("VQI-S: Acc=%.4f, F1=%.4f, AUC=%.4f, CDF_all_pass=%s",
                     results_s.get("confusion", {}).get("accuracy", 0),
                     results_s.get("confusion", {}).get("f1_score", 0),
                     results_s.get("confusion", {}).get("auc_roc", 0),
                     results_s.get("all_cdf_pass", "N/A"))
    if results_v:
        logger.info("VQI-V: Acc=%.4f, F1=%.4f, AUC=%.4f, CDF_all_pass=%s",
                     results_v.get("confusion", {}).get("accuracy", 0),
                     results_v.get("confusion", {}).get("f1_score", 0),
                     results_v.get("confusion", {}).get("auc_roc", 0),
                     results_v.get("all_cdf_pass", "N/A"))

    # Clean up checkpoints
    for cp in [
        os.path.join(val_s_dir, "_checkpoint_step7_s.yaml"),
        os.path.join(val_v_dir, "_checkpoint_step7_v.yaml"),
    ]:
        if os.path.exists(cp):
            os.remove(cp)
            logger.info("Removed checkpoint: %s", cp)


if __name__ == "__main__":
    main()
