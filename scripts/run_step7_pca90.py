"""
Step 7 PCA-90% Orchestrator: Model Validation with PCA-90% models.

Reuses the full-feature Step 7 pipeline (stages 1-2 model-agnostic)
but injects PCA transform in Stage 3 (prediction).

Phases:
  A: Skip (embeddings + scores already exist from full-feature run)
  B-C: VQI-S PCA-90% validation
  D: VQI-V PCA-90% validation
  E: Dual-score analysis

Usage:
    python scripts/run_step7_pca90.py [--skip-s] [--skip-v] [--skip-dual]
"""

import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.validate_model import (
    merge_provider_scores,
    compute_validation_labels,
    compute_bin_distribution,
    assign_bins,
    compute_cdf_per_bin,
    check_cdf_shift,
    compute_confusion_metrics,
    compute_quadrant_analysis,
    load_cv_stability,
    PROVIDER_NAMES,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def predict_vqi_scores_pca(
    features_npy, feature_names_json, selected_features_txt,
    scaler_path, pca_path, model_path, score_type,
):
    """Predict VQI scores using PCA-90% pipeline.

    Pipeline: select features -> StandardScaler -> PCA -> RF predict.
    """
    logger = logging.getLogger("step7_pca90")

    # Load full features
    X_full = np.load(features_npy)
    with open(feature_names_json, "r", encoding="utf-8") as f:
        all_names = json.load(f)

    # Load selected feature names
    with open(selected_features_txt, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    # Map to indices
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = np.array([name_to_idx[name] for name in selected_names])
    X_selected = X_full[:, indices].astype(np.float32)

    # Replace NaN/Inf
    nan_mask = ~np.isfinite(X_selected)
    if nan_mask.any():
        logger.warning("Replacing %d NaN/Inf in VQI-%s features", nan_mask.sum(), score_type.upper())
        X_selected[nan_mask] = 0.0

    # Load PCA pipeline
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    logger.info("PCA-%s: %d components", score_type.upper(), pca.n_components_)

    # Transform
    X_pca = pca.transform(scaler.transform(X_selected))

    # Load model and predict
    clf = joblib.load(model_path)
    logger.info(
        "Loaded VQI-%s PCA model: %d trees, %d features",
        score_type.upper(), clf.n_estimators, clf.n_features_in_,
    )

    probas = clf.predict_proba(X_pca)[:, 1]
    scores = np.round(probas * 100).astype(int)
    scores = np.clip(scores, 0, 100)

    logger.info(
        "VQI-%s PCA scores: N=%d, min=%d, max=%d, mean=%.1f, median=%d",
        score_type.upper(), len(scores),
        int(scores.min()), int(scores.max()),
        float(scores.mean()), int(np.median(scores)),
    )
    return scores, selected_names


def run_pca90_validation(
    score_type, validation_csv, features_npy, feature_names_json,
    selected_features_txt, scaler_path, pca_path, model_path,
    training_dir, provider_scores_dir, thresholds_yaml,
    output_dir, split_name="val_set",
):
    """Run the full PCA-90% validation pipeline for one score type."""
    logger = logging.getLogger("step7_pca90")
    prefix = f"VQI-{score_type.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "_v" if score_type == "v" else ""

    # ---- Stage 1: Merge provider scores ----
    logger.info("[%s] Stage 1/8: Merging provider scores...", prefix)
    val_df = merge_provider_scores(validation_csv, provider_scores_dir, split_name)
    val_df.to_csv(os.path.join(output_dir, "validation_merged.csv"), index=False, encoding="utf-8")

    # ---- Stage 2: Compute labels ----
    logger.info("[%s] Stage 2/8: Computing validation labels...", prefix)
    val_df = compute_validation_labels(val_df, thresholds_yaml)
    val_df.to_csv(os.path.join(output_dir, "validation_labeled.csv"), index=False, encoding="utf-8")

    n1 = int((val_df["label"] == 1).sum())
    n0 = int((val_df["label"] == 0).sum())
    logger.info("[%s] Labels: Class1=%d, Class0=%d", prefix, n1, n0)

    results = {"n_class_1": n1, "n_class_0": n0}

    # ---- Stage 3: Predict VQI scores with PCA pipeline ----
    logger.info("[%s] Stage 3/8: Predicting VQI scores (PCA-90%%)...", prefix)
    scores, selected_names = predict_vqi_scores_pca(
        features_npy, feature_names_json, selected_features_txt,
        scaler_path, pca_path, model_path, score_type,
    )
    score_col = f"vqi_{score_type}_score"
    val_df[score_col] = scores

    results_csv = os.path.join(output_dir, f"validation_results{suffix}.csv")
    val_df.to_csv(results_csv, index=False, encoding="utf-8")

    results["n_features"] = len(selected_names)
    results["score_min"] = int(scores.min())
    results["score_max"] = int(scores.max())
    results["score_mean"] = round(float(scores.mean()), 2)
    results["score_median"] = int(np.median(scores))

    labels = val_df["label"].values

    # ---- Stage 4: Bin distribution ----
    logger.info("[%s] Stage 4/8: Computing bin distribution...", prefix)
    bin_df = compute_bin_distribution(scores)
    bin_df.to_csv(os.path.join(output_dir, "bin_distribution.csv"), index=False, encoding="utf-8")
    results["bin_distribution"] = bin_df.to_dict("records")

    # ---- Stage 5: CDF analysis ----
    logger.info("[%s] Stage 5/8: CDF analysis per quality bin...", prefix)
    cdf_results = {}
    cdf_shifts = {}

    for pn in PROVIDER_NAMES:
        col = f"score_{pn}"
        if col not in val_df.columns:
            continue
        genuine = val_df[col].values.astype(np.float32)
        cdf_data = compute_cdf_per_bin(scores, genuine, pn)
        cdf_results[pn] = {
            name: {"n": d["n"], "mean": d["mean"]}
            for name, d in cdf_data.items()
        }
        cdf_shifts[pn] = check_cdf_shift(cdf_data, pn)

        # Save CDF data for visualization
        for bname, bdata in cdf_data.items():
            np.savez_compressed(
                os.path.join(output_dir, f"cdf_{pn}_{bname.replace(' ', '_')}.npz"),
                x=bdata["x"], cdf=bdata["cdf"],
            )

    results["cdf_shifts"] = cdf_shifts
    results["cdf_summary"] = cdf_results
    results["all_cdf_pass"] = all(cdf_shifts.values())

    # ---- Stage 6: Confusion matrix ----
    logger.info("[%s] Stage 6/8: Computing confusion matrix and metrics...", prefix)
    metrics = compute_confusion_metrics(scores, labels, threshold=50)
    results["confusion"] = metrics

    metrics_summary = {k: v for k, v in metrics.items() if k not in ("fpr", "tpr", "roc_thresholds")}
    yaml_path = os.path.join(output_dir, f"validation_metrics{suffix}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metrics_summary, f, default_flow_style=False, sort_keys=False)

    np.savez_compressed(
        os.path.join(output_dir, "roc_data.npz"),
        fpr=np.array(metrics["fpr"]),
        tpr=np.array(metrics["tpr"]),
        thresholds=np.array(metrics["roc_thresholds"]),
    )

    # ---- Stage 7: OOB convergence ----
    logger.info("[%s] Stage 7/8: Loading OOB convergence...", prefix)
    oob_conv_csv = os.path.join(training_dir, "oob_convergence.csv")
    if os.path.exists(oob_conv_csv):
        conv_df = pd.read_csv(oob_conv_csv)
        oob_meta_path = os.path.join(training_dir, "oob_convergence_meta.yaml")
        with open(oob_meta_path, "r", encoding="utf-8") as f:
            conv_meta = yaml.safe_load(f)
        results["oob_convergence_point"] = conv_meta["convergence_point"]
        results["oob_min_error"] = conv_meta["min_oob_error"]
    else:
        # PCA training dirs may not have OOB convergence - generate from training_metrics
        logger.info("[%s] No OOB convergence CSV; generating from grid_search_results...", prefix)
        gs_path = os.path.join(training_dir, "grid_search_results.csv")
        gs_df = pd.read_csv(gs_path)

        # Build pseudo-convergence from grid search (sorted by n_estimators)
        conv_rows = []
        for _, row in gs_df.iterrows():
            conv_rows.append({
                "n_estimators": int(row["n_estimators"]),
                "oob_error": float(row["oob_error"]),
            })
        conv_df = pd.DataFrame(conv_rows).sort_values("n_estimators").drop_duplicates("n_estimators")

        conv_csv_out = os.path.join(output_dir, "oob_convergence.csv")
        conv_df.to_csv(conv_csv_out, index=False, encoding="utf-8")

        # Synthetic meta
        best_row = gs_df.iloc[0]
        conv_meta = {
            "convergence_point": int(best_row["n_estimators"]),
            "min_oob_error": float(best_row["oob_error"]),
        }
        meta_out = os.path.join(output_dir, "oob_convergence_meta.yaml")
        with open(meta_out, "w", encoding="utf-8") as f:
            yaml.dump(conv_meta, f, default_flow_style=False, sort_keys=False)

        results["oob_convergence_point"] = conv_meta["convergence_point"]
        results["oob_min_error"] = conv_meta["min_oob_error"]

    # ---- Stage 8: CV stability ----
    logger.info("[%s] Stage 8/8: Loading CV stability...", prefix)
    cv_data = load_cv_stability(training_dir)
    results["cv_stability"] = cv_data

    logger.info("[%s] PCA-90%% validation pipeline complete.", prefix)
    return results


def verify_validation_outputs(output_dir, score_type):
    """Verify validation outputs for a score type."""
    ok = True
    logger = logging.getLogger("step7_pca90")
    prefix = f"VQI-{score_type.upper()}"
    suffix = "_v" if score_type == "v" else ""

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
        logger.error("%s: validation_results%s.csv NOT FOUND", prefix, suffix)
        ok = False

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


def run_dual_score_analysis(val_s_dir, val_v_dir, output_dir):
    """Run dual-score quadrant analysis on PCA-90% results."""
    logger = logging.getLogger("step7_pca90")
    logger.info("=" * 60)
    logger.info("Phase E: Dual-Score Analysis (PCA-90%%)")
    logger.info("=" * 60)

    s_df = pd.read_csv(os.path.join(val_s_dir, "validation_results.csv"))
    v_df = pd.read_csv(os.path.join(val_v_dir, "validation_results_v.csv"))

    vqi_s_scores = s_df["vqi_s_score"].values.astype(int)
    vqi_v_scores = v_df["vqi_v_score"].values.astype(int)
    labels = s_df["label"].values

    genuine_scores = {}
    for pn in PROVIDER_NAMES:
        col = f"score_{pn}"
        if col in s_df.columns:
            genuine_scores[pn] = s_df[col].values.astype(np.float32)

    # Thresholds
    threshold_s, threshold_v = 50, 50

    s_metrics_path = os.path.join(val_s_dir, "validation_metrics.yaml")
    v_metrics_path = os.path.join(val_v_dir, "validation_metrics_v.yaml")

    if os.path.exists(s_metrics_path):
        with open(s_metrics_path, "r", encoding="utf-8") as f:
            s_metrics = yaml.safe_load(f)
        youden_s = s_metrics.get("youden_j_threshold", 50)
        if abs(youden_s - 50) > 5:
            threshold_s = int(round(youden_s))

    if os.path.exists(v_metrics_path):
        with open(v_metrics_path, "r", encoding="utf-8") as f:
            v_metrics = yaml.safe_load(f)
        youden_v = v_metrics.get("youden_j_threshold", 50)
        if abs(youden_v - 50) > 5:
            threshold_v = int(round(youden_v))

    quad_df = compute_quadrant_analysis(
        vqi_s_scores, vqi_v_scores, labels, genuine_scores,
        threshold_s, threshold_v,
    )

    os.makedirs(output_dir, exist_ok=True)
    quad_df.to_csv(os.path.join(output_dir, "quadrant_analysis.csv"), index=False, encoding="utf-8")

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

    dual_df.to_csv(os.path.join(output_dir, "dual_score_data.csv"), index=False, encoding="utf-8")

    thresh_info = {
        "threshold_s": threshold_s,
        "threshold_v": threshold_v,
        "method_s": "youden_j" if threshold_s != 50 else "midpoint",
        "method_v": "youden_j" if threshold_v != 50 else "midpoint",
    }
    with open(os.path.join(output_dir, "dual_score_thresholds.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(thresh_info, f, default_flow_style=False, sort_keys=False)

    # Acceptance check
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
    parser = argparse.ArgumentParser(description="Step 7 PCA-90%: Model Validation")
    parser.add_argument("--skip-s", action="store_true", help="Skip VQI-S validation")
    parser.add_argument("--skip-v", action="store_true", help="Skip VQI-V validation")
    parser.add_argument("--skip-dual", action="store_true", help="Skip dual-score analysis")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("step7_pca90")

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, "data")
    features_dir = os.path.join(data_dir, "features")
    scores_dir = os.path.join(data_dir, "provider_scores")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    validation_csv = os.path.join(data_dir, "validation_set.csv")
    thresholds_yaml = os.path.join(data_dir, "label_thresholds.yaml")

    # Output dirs
    val_s_dir = os.path.join(data_dir, "validation_pca90")
    val_v_dir = os.path.join(data_dir, "validation_pca90_v")
    os.makedirs(val_s_dir, exist_ok=True)
    os.makedirs(val_v_dir, exist_ok=True)

    t0 = time.time()

    # ---- VQI-S PCA-90% Validation ----
    results_s = None
    if not args.skip_s:
        logger.info("=" * 60)
        logger.info("Phase B-C: VQI-S PCA-90%% Validation")
        logger.info("=" * 60)

        results_s = run_pca90_validation(
            score_type="s",
            validation_csv=validation_csv,
            features_npy=os.path.join(features_dir, "features_s_val.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_s.json"),
            selected_features_txt=os.path.join(data_dir, "evaluation", "selected_features.txt"),
            scaler_path=os.path.join(models_dir, "vqi_pca_scaler_s.joblib"),
            pca_path=os.path.join(models_dir, "vqi_pca_transformer_s.joblib"),
            model_path=os.path.join(models_dir, "vqi_rf_pca_model.joblib"),
            training_dir=os.path.join(data_dir, "training_pca"),
            provider_scores_dir=scores_dir,
            thresholds_yaml=thresholds_yaml,
            output_dir=val_s_dir,
        )

        s_ok = verify_validation_outputs(val_s_dir, "s")
        if not s_ok:
            logger.error("VQI-S PCA-90%% validation verification FAILED")
            sys.exit(1)

    t_s = time.time() - t0

    # ---- VQI-V PCA-90% Validation ----
    results_v = None
    if not args.skip_v:
        logger.info("=" * 60)
        logger.info("Phase D: VQI-V PCA-90%% Validation")
        logger.info("=" * 60)

        results_v = run_pca90_validation(
            score_type="v",
            validation_csv=validation_csv,
            features_npy=os.path.join(features_dir, "features_v_val.npy"),
            feature_names_json=os.path.join(features_dir, "feature_names_v.json"),
            selected_features_txt=os.path.join(data_dir, "evaluation_v", "selected_features.txt"),
            scaler_path=os.path.join(models_dir, "vqi_pca_scaler_v.joblib"),
            pca_path=os.path.join(models_dir, "vqi_pca_transformer_v.joblib"),
            model_path=os.path.join(models_dir, "vqi_v_rf_pca_model.joblib"),
            training_dir=os.path.join(data_dir, "training_pca_v"),
            provider_scores_dir=scores_dir,
            thresholds_yaml=thresholds_yaml,
            output_dir=val_v_dir,
        )

        v_ok = verify_validation_outputs(val_v_dir, "v")
        if not v_ok:
            logger.error("VQI-V PCA-90%% validation verification FAILED")
            sys.exit(1)

    t_v = time.time() - t0 - t_s

    # ---- Dual-score analysis ----
    if not args.skip_dual and not args.skip_s and not args.skip_v:
        run_dual_score_analysis(val_s_dir, val_v_dir, output_dir=val_s_dir)

    # ---- Summary ----
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Step 7 PCA-90%% COMPLETE in %.1f seconds (%.1f min) [S: %.1f min, V: %.1f min]",
        elapsed, elapsed / 60, t_s / 60, t_v / 60,
    )
    logger.info("=" * 60)

    if results_s:
        logger.info("VQI-S PCA-90%%: Acc=%.4f, F1=%.4f, AUC=%.4f, CDF_all_pass=%s",
                     results_s.get("confusion", {}).get("accuracy", 0),
                     results_s.get("confusion", {}).get("f1_score", 0),
                     results_s.get("confusion", {}).get("auc_roc", 0),
                     results_s.get("all_cdf_pass", "N/A"))
    if results_v:
        logger.info("VQI-V PCA-90%%: Acc=%.4f, F1=%.4f, AUC=%.4f, CDF_all_pass=%s",
                     results_v.get("confusion", {}).get("accuracy", 0),
                     results_v.get("confusion", {}).get("f1_score", 0),
                     results_v.get("confusion", {}).get("auc_roc", 0),
                     results_v.get("all_cdf_pass", "N/A"))


if __name__ == "__main__":
    main()
