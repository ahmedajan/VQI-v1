"""
Step 7: Model Validation — Shared Pipeline

Validates trained RF models on the held-out 50K validation set.
Used for both VQI-S (Signal Quality) and VQI-V (Voice Distinctiveness).

Pipeline:
  1. Merge provider scores into validation data
  2. Compute binary labels using thresholds from Step 2
  3. Predict VQI scores using trained model
  4. Score distribution by quality bin
  5. CDF of genuine scores per quality bin
  6. Confusion matrix + metrics
  7. Dual-score scatter analysis (VQI-S vs VQI-V)

VQI-S wrappers at bottom; VQI-V wrappers in validate_model_v.py.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Quality bins per blueprint Section 7.2
QUALITY_BINS = [
    ("Very Low", 0, 20),
    ("Low", 21, 40),
    ("Medium", 41, 60),
    ("High", 61, 80),
    ("Very High", 81, 100),
]

PROVIDER_NAMES = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]
PROVIDER_SHORTS = {"P1_ECAPA": "ecapa", "P2_RESNET": "resnet", "P3_ECAPA2": "ecapa2"}


# ---------------------------------------------------------------------------
# 7.1 / 7.8: Merge provider scores into validation data
# ---------------------------------------------------------------------------

def merge_provider_scores(
    validation_csv: str,
    provider_scores_dir: str,
    split_name: str = "val_set",
) -> pd.DataFrame:
    """Merge validation CSV with provider score CSVs.

    Reads score_P{1,2,3}_{short}.csv files produced by compute_scores.py
    for the val_set split, and joins them with validation_set.csv by filename.

    Returns DataFrame with columns:
        filename, speaker_id, dataset_source,
        score_P1, score_P2, score_P3 (genuine_norm values)
    """
    val_df = pd.read_csv(validation_csv)
    logger.info("Loaded validation set: %d rows", len(val_df))

    for pn in PROVIDER_NAMES:
        short = PROVIDER_SHORTS[pn]
        score_csv = os.path.join(
            provider_scores_dir, f"scores_{split_name}_{pn}_{short}.csv"
        )
        if not os.path.exists(score_csv):
            raise FileNotFoundError(f"Provider score file not found: {score_csv}")

        score_df = pd.read_csv(score_csv)
        logger.info(
            "Loaded %s scores: %d rows, valid genuine_norm: %d",
            pn, len(score_df), score_df["genuine_norm"].notna().sum(),
        )

        # Map filename -> genuine_norm
        score_map = dict(zip(score_df["filename"], score_df["genuine_norm"]))
        col = f"score_{pn[-2:]}" if pn == "P1_ECAPA" else f"score_{pn[:2]}"
        # Use standardized column names
        col = f"score_{pn}"
        val_df[col] = val_df["filename"].map(score_map)

        n_mapped = val_df[col].notna().sum()
        logger.info("  Mapped %d/%d scores for %s", n_mapped, len(val_df), pn)

    return val_df


# ---------------------------------------------------------------------------
# 7.1 / 7.8: Compute binary labels for validation
# ---------------------------------------------------------------------------

def compute_validation_labels(
    val_df: pd.DataFrame,
    thresholds_yaml: str,
) -> pd.DataFrame:
    """Assign binary labels to validation samples using Step 2 rules.

    Class 1: all 3 genuine_norm >= P90 threshold
    Class 0: all 3 genuine_norm < FMR@0.001 threshold
    Others: label = NaN (excluded from confusion matrix)

    Returns val_df with new 'label' column.
    """
    with open(thresholds_yaml, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)

    p90 = {
        "P1": thresholds["P1"]["percentile_90"],
        "P2": thresholds["P2"]["percentile_90"],
        "P3": thresholds["P3"]["percentile_90"],
    }
    fmr = {
        "P1": thresholds["P1"]["fmr_001"],
        "P2": thresholds["P2"]["fmr_001"],
        "P3": thresholds["P3"]["fmr_001"],
    }

    s1 = val_df["score_P1_ECAPA"]
    s2 = val_df["score_P2_RESNET"]
    s3 = val_df["score_P3_ECAPA2"]

    # Class 1: all >= P90
    class1_mask = (s1 >= p90["P1"]) & (s2 >= p90["P2"]) & (s3 >= p90["P3"])
    # Class 0: all < FMR
    class0_mask = (s1 < fmr["P1"]) & (s2 < fmr["P2"]) & (s3 < fmr["P3"])

    # Samples with any NaN score are excluded
    has_scores = s1.notna() & s2.notna() & s3.notna()
    class1_mask = class1_mask & has_scores
    class0_mask = class0_mask & has_scores

    val_df["label"] = np.nan
    val_df.loc[class1_mask, "label"] = 1
    val_df.loc[class0_mask, "label"] = 0

    n1 = int(class1_mask.sum())
    n0 = int(class0_mask.sum())
    n_nan_scores = int((~has_scores).sum())
    n_excluded = len(val_df) - n1 - n0

    logger.info(
        "Validation labels: Class1=%d, Class0=%d, Excluded=%d (NaN scores=%d)",
        n1, n0, n_excluded, n_nan_scores,
    )
    return val_df


# ---------------------------------------------------------------------------
# 7.1 / 7.8: Predict VQI scores
# ---------------------------------------------------------------------------

def predict_vqi_scores(
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    model_path: str,
    score_type: str,
) -> Tuple[np.ndarray, List[str]]:
    """Predict VQI scores for validation features.

    Returns:
        scores: (N,) int array in [0, 100]
        selected_names: list of selected feature names
    """
    # Load full features
    X_full = np.load(features_npy)
    with open(feature_names_json, "r", encoding="utf-8") as f:
        all_names = json.load(f)
    assert X_full.shape[1] == len(all_names), (
        f"Feature columns ({X_full.shape[1]}) != names ({len(all_names)})"
    )

    # Load selected feature names
    with open(selected_features_txt, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    # Map to indices
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = np.array([name_to_idx[name] for name in selected_names])
    X_selected = X_full[:, indices].astype(np.float32)

    # Verify
    assert not np.any(np.isnan(X_selected)), "NaN in validation features"
    assert not np.any(np.isinf(X_selected)), "Inf in validation features"

    # Load model and predict
    clf = joblib.load(model_path)
    logger.info(
        "Loaded VQI-%s model: %d trees, %d features",
        score_type.upper(), clf.n_estimators, clf.n_features_in_,
    )
    assert clf.n_features_in_ == len(selected_names), (
        f"Model expects {clf.n_features_in_} features, got {len(selected_names)}"
    )

    # P(Class1) -> score [0, 100]
    probas = clf.predict_proba(X_selected)[:, 1]
    scores = np.round(probas * 100).astype(int)
    scores = np.clip(scores, 0, 100)

    logger.info(
        "VQI-%s scores: N=%d, min=%d, max=%d, mean=%.1f, median=%d",
        score_type.upper(), len(scores),
        int(scores.min()), int(scores.max()),
        float(scores.mean()), int(np.median(scores)),
    )
    return scores, selected_names


# ---------------------------------------------------------------------------
# 7.2 / 7.9: Score distribution by quality bin
# ---------------------------------------------------------------------------

def compute_bin_distribution(scores: np.ndarray) -> pd.DataFrame:
    """Assign each score to a quality bin and compute distribution."""
    bins = []
    for name, lo, hi in QUALITY_BINS:
        mask = (scores >= lo) & (scores <= hi)
        count = int(mask.sum())
        pct = count / len(scores) * 100
        bins.append({"bin": name, "lo": lo, "hi": hi, "count": count, "pct": round(pct, 2)})
    return pd.DataFrame(bins)


def assign_bins(scores: np.ndarray) -> np.ndarray:
    """Return bin name for each score."""
    bin_names = np.empty(len(scores), dtype=object)
    for name, lo, hi in QUALITY_BINS:
        mask = (scores >= lo) & (scores <= hi)
        bin_names[mask] = name
    return bin_names


# ---------------------------------------------------------------------------
# 7.3 / 7.10: CDF of genuine scores per quality bin
# ---------------------------------------------------------------------------

def compute_cdf_per_bin(
    scores: np.ndarray,
    genuine_scores: np.ndarray,
    provider_name: str,
) -> Dict:
    """Compute CDF data for genuine scores grouped by VQI quality bin.

    Returns dict: {bin_name: {"x": array, "cdf": array, "n": int, "mean": float}}
    """
    bin_names = assign_bins(scores)
    result = {}

    for name, _, _ in QUALITY_BINS:
        mask = (bin_names == name) & (~np.isnan(genuine_scores))
        if mask.sum() == 0:
            result[name] = {"x": np.array([]), "cdf": np.array([]), "n": 0, "mean": np.nan}
            continue

        vals = np.sort(genuine_scores[mask])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        result[name] = {
            "x": vals,
            "cdf": cdf,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
        }

    return result


def check_cdf_shift(cdf_data: Dict, provider_name: str) -> bool:
    """Verify that 'Very High' CDF is RIGHT of 'Very Low'.

    Returns True if mean genuine score of Very High > Very Low.
    """
    vh = cdf_data.get("Very High", {})
    vl = cdf_data.get("Very Low", {})

    if vh.get("n", 0) == 0 or vl.get("n", 0) == 0:
        logger.warning(
            "%s: Cannot verify CDF shift (Very High n=%d, Very Low n=%d)",
            provider_name, vh.get("n", 0), vl.get("n", 0),
        )
        return False

    shift_ok = vh["mean"] > vl["mean"]
    logger.info(
        "%s CDF shift: Very High mean=%.4f, Very Low mean=%.4f -> %s",
        provider_name, vh["mean"], vl["mean"],
        "PASS" if shift_ok else "FAIL",
    )
    return shift_ok


# ---------------------------------------------------------------------------
# 7.4 / 7.11: Confusion matrix and metrics
# ---------------------------------------------------------------------------

def compute_confusion_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: int = 50,
) -> Dict:
    """Compute confusion matrix and classification metrics.

    Only includes samples with non-NaN labels.

    Returns dict with: cm, accuracy, precision, recall, f1, auc_roc,
        fpr, tpr, thresholds, youden_j_threshold, n_labeled.
    """
    # Filter to labeled samples
    valid = ~np.isnan(labels)
    y_true = labels[valid].astype(int)
    y_scores = scores[valid]

    # Binary predictions at threshold
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC (use continuous scores, not binary)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # Youden's J statistic
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    youden_threshold = float(roc_thresholds[best_j_idx])

    logger.info(
        "Confusion (threshold=%d): Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%.4f",
        threshold, acc, prec, rec, f1, auc,
    )
    logger.info("  CM: %s", cm.tolist())
    logger.info("  Youden's J threshold: %.1f", youden_threshold)

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "auc_roc": round(float(auc), 6),
        "threshold": threshold,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "youden_j_threshold": round(float(youden_threshold), 2),
        "n_labeled": int(valid.sum()),
        "n_class_0": int((y_true == 0).sum()),
        "n_class_1": int((y_true == 1).sum()),
    }


# ---------------------------------------------------------------------------
# 7.14: Dual-score scatter / quadrant analysis
# ---------------------------------------------------------------------------

def compute_quadrant_analysis(
    vqi_s_scores: np.ndarray,
    vqi_v_scores: np.ndarray,
    labels: np.ndarray,
    genuine_scores: Dict[str, np.ndarray],
    threshold_s: int = 50,
    threshold_v: int = 50,
) -> pd.DataFrame:
    """Compute per-quadrant statistics for the 2D scatter.

    Args:
        vqi_s_scores: (N,) VQI-S scores
        vqi_v_scores: (N,) VQI-V scores
        labels: (N,) binary labels (may contain NaN)
        genuine_scores: dict {provider_name: (N,) genuine_norm array}
        threshold_s: VQI-S threshold
        threshold_v: VQI-V threshold

    Returns DataFrame with per-quadrant stats.
    """
    quadrants = {
        "Q1 (High S, High V)": (vqi_s_scores >= threshold_s) & (vqi_v_scores >= threshold_v),
        "Q2 (Low S, High V)": (vqi_s_scores < threshold_s) & (vqi_v_scores >= threshold_v),
        "Q3 (Low S, Low V)": (vqi_s_scores < threshold_s) & (vqi_v_scores < threshold_v),
        "Q4 (High S, Low V)": (vqi_s_scores >= threshold_s) & (vqi_v_scores < threshold_v),
    }

    rows = []
    for qname, mask in quadrants.items():
        count = int(mask.sum())
        pct = count / len(vqi_s_scores) * 100

        # Labeled subset
        labeled_mask = mask & ~np.isnan(labels)
        n_labeled = int(labeled_mask.sum())
        if n_labeled > 0:
            q_labels = labels[labeled_mask]
            class1_rate = float(np.mean(q_labels == 1))
        else:
            class1_rate = np.nan

        # Mean genuine scores per provider
        row = {
            "quadrant": qname,
            "count": count,
            "pct_of_total": round(pct, 2),
            "n_labeled": n_labeled,
            "class1_rate": round(class1_rate, 4) if not np.isnan(class1_rate) else np.nan,
            "failure_rate": round(1 - class1_rate, 4) if not np.isnan(class1_rate) else np.nan,
            "mean_vqi_s": round(float(vqi_s_scores[mask].mean()), 2) if count > 0 else np.nan,
            "mean_vqi_v": round(float(vqi_v_scores[mask].mean()), 2) if count > 0 else np.nan,
        }

        for pn, gscores in genuine_scores.items():
            valid = mask & ~np.isnan(gscores)
            if valid.sum() > 0:
                row[f"mean_genuine_{pn}"] = round(float(gscores[valid].mean()), 4)
            else:
                row[f"mean_genuine_{pn}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Quadrant analysis:\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 7.5 / 7.12: OOB convergence (reuse Step 6 data)
# ---------------------------------------------------------------------------

def load_oob_convergence(training_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """Load OOB convergence data from Step 6."""
    conv_path = os.path.join(training_dir, "oob_convergence.csv")
    meta_path = os.path.join(training_dir, "oob_convergence_meta.yaml")

    conv_df = pd.read_csv(conv_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    logger.info(
        "OOB convergence: %d points, convergence at %d trees, min OOB=%.4f",
        len(conv_df), meta["convergence_point"], meta["min_oob_error"],
    )
    return conv_df, meta


# ---------------------------------------------------------------------------
# 7.6 / 7.13: CV stability (reuse Step 6 data)
# ---------------------------------------------------------------------------

def load_cv_stability(training_dir: str) -> Dict:
    """Load CV stability from Step 6 grid search results."""
    gs_path = os.path.join(training_dir, "grid_search_results.csv")
    gs_df = pd.read_csv(gs_path)

    # Get the best config (first row with CV data)
    cv_rows = gs_df[gs_df["cv_accuracy_mean"].notna()]
    if len(cv_rows) == 0:
        logger.warning("No CV data found in grid search results")
        return {"cv_accuracy_mean": None, "cv_accuracy_std": None, "stable": False}

    best = cv_rows.iloc[0]
    cv_mean = float(best["cv_accuracy_mean"])
    cv_std = float(best["cv_accuracy_std"])
    stable = cv_std < 0.03

    logger.info(
        "CV stability: mean=%.4f, std=%.4f -> %s (threshold < 0.03)",
        cv_mean, cv_std, "PASS" if stable else "FAIL",
    )

    return {
        "cv_accuracy_mean": round(cv_mean, 6),
        "cv_accuracy_std": round(cv_std, 6),
        "stable": stable,
        "all_cv_rows": cv_rows[["n_estimators", "max_features", "cv_accuracy_mean", "cv_accuracy_std"]].to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

STAGES = [
    "merge_scores",
    "compute_labels",
    "predict_scores",
    "bin_distribution",
    "cdf_analysis",
    "confusion_metrics",
    "oob_convergence",
    "cv_stability",
]


def _load_checkpoint(checkpoint_path: str) -> Dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_checkpoint(checkpoint_path: str, state: Dict):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared validation pipeline
# ---------------------------------------------------------------------------

def _run_validation_pipeline(
    score_type: str,
    validation_csv: str,
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    model_path: str,
    training_dir: str,
    provider_scores_dir: str,
    thresholds_yaml: str,
    output_dir: str,
    checkpoint_path: str,
    split_name: str = "val_set",
    resume: bool = False,
) -> Dict:
    """Run the full validation pipeline for one score type.

    Returns dict with all validation results.
    """
    prefix = f"VQI-{score_type.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    state = {}
    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state:
            logger.info("[%s] Resuming from checkpoint: completed=%s", prefix, state.get("completed", []))

    completed = set(state.get("completed", []))
    results = state.get("results", {})

    # ---- Stage 1: Merge provider scores ----
    merged_csv = os.path.join(output_dir, "validation_merged.csv")
    if "merge_scores" not in completed:
        logger.info("[%s] Stage 1/8: Merging provider scores...", prefix)
        val_df = merge_provider_scores(validation_csv, provider_scores_dir, split_name)
        val_df.to_csv(merged_csv, index=False, encoding="utf-8")
        completed.add("merge_scores")
        state["completed"] = sorted(completed)
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 1/8 complete.", prefix)
    else:
        logger.info("[%s] Stage 1/8: Loading from checkpoint...", prefix)
        val_df = pd.read_csv(merged_csv)

    # ---- Stage 2: Compute labels ----
    labeled_csv = os.path.join(output_dir, "validation_labeled.csv")
    if "compute_labels" not in completed:
        logger.info("[%s] Stage 2/8: Computing validation labels...", prefix)
        val_df = compute_validation_labels(val_df, thresholds_yaml)
        val_df.to_csv(labeled_csv, index=False, encoding="utf-8")
        n1 = int((val_df["label"] == 1).sum())
        n0 = int((val_df["label"] == 0).sum())
        results["n_class_1"] = n1
        results["n_class_0"] = n0
        results["n_excluded"] = len(val_df) - n1 - n0
        completed.add("compute_labels")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 2/8 complete.", prefix)
    else:
        logger.info("[%s] Stage 2/8: Loading from checkpoint...", prefix)
        val_df = pd.read_csv(labeled_csv)

    # ---- Stage 3: Predict VQI scores ----
    results_csv = os.path.join(output_dir, f"validation_results{'_v' if score_type == 'v' else ''}.csv")
    if "predict_scores" not in completed:
        logger.info("[%s] Stage 3/8: Predicting VQI scores...", prefix)
        scores, selected_names = predict_vqi_scores(
            features_npy, feature_names_json, selected_features_txt,
            model_path, score_type,
        )
        score_col = f"vqi_{score_type}_score"
        val_df[score_col] = scores
        val_df.to_csv(results_csv, index=False, encoding="utf-8")

        results["n_features"] = len(selected_names)
        results["score_min"] = int(scores.min())
        results["score_max"] = int(scores.max())
        results["score_mean"] = round(float(scores.mean()), 2)
        results["score_median"] = int(np.median(scores))

        completed.add("predict_scores")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 3/8 complete.", prefix)
    else:
        logger.info("[%s] Stage 3/8: Loading from checkpoint...", prefix)
        val_df = pd.read_csv(results_csv)

    score_col = f"vqi_{score_type}_score"
    scores = val_df[score_col].values.astype(int)
    labels = val_df["label"].values

    # ---- Stage 4: Bin distribution ----
    if "bin_distribution" not in completed:
        logger.info("[%s] Stage 4/8: Computing bin distribution...", prefix)
        bin_df = compute_bin_distribution(scores)
        bin_csv = os.path.join(output_dir, "bin_distribution.csv")
        bin_df.to_csv(bin_csv, index=False, encoding="utf-8")
        results["bin_distribution"] = bin_df.to_dict("records")
        completed.add("bin_distribution")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 4/8 complete.", prefix)

    # ---- Stage 5: CDF analysis ----
    if "cdf_analysis" not in completed:
        logger.info("[%s] Stage 5/8: CDF analysis per quality bin...", prefix)
        cdf_results = {}
        cdf_shifts = {}

        for pn in PROVIDER_NAMES:
            col = f"score_{pn}"
            if col not in val_df.columns:
                logger.warning("  %s column not found, skipping CDF for %s", col, pn)
                continue
            genuine = val_df[col].values.astype(np.float32)
            cdf_data = compute_cdf_per_bin(scores, genuine, pn)
            cdf_results[pn] = {
                name: {"n": d["n"], "mean": d["mean"]}
                for name, d in cdf_data.items()
            }
            cdf_shifts[pn] = check_cdf_shift(cdf_data, pn)

        results["cdf_shifts"] = cdf_shifts
        results["cdf_summary"] = cdf_results
        results["all_cdf_pass"] = all(cdf_shifts.values())

        # Save CDF data for visualization
        cdf_save = {}
        for pn in PROVIDER_NAMES:
            col = f"score_{pn}"
            if col not in val_df.columns:
                continue
            genuine = val_df[col].values.astype(np.float32)
            cdf_data = compute_cdf_per_bin(scores, genuine, pn)
            for bname, bdata in cdf_data.items():
                key = f"{pn}_{bname}"
                np.savez_compressed(
                    os.path.join(output_dir, f"cdf_{pn}_{bname.replace(' ', '_')}.npz"),
                    x=bdata["x"], cdf=bdata["cdf"],
                )

        completed.add("cdf_analysis")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 5/8 complete.", prefix)

    # ---- Stage 6: Confusion matrix ----
    if "confusion_metrics" not in completed:
        logger.info("[%s] Stage 6/8: Computing confusion matrix and metrics...", prefix)
        metrics = compute_confusion_metrics(scores, labels, threshold=50)
        results["confusion"] = metrics

        # Save metrics YAML
        # Separate the ROC curve arrays (large) from the summary metrics
        metrics_summary = {k: v for k, v in metrics.items() if k not in ("fpr", "tpr", "roc_thresholds")}
        yaml_path = os.path.join(output_dir, f"validation_metrics{'_v' if score_type == 'v' else ''}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(metrics_summary, f, default_flow_style=False, sort_keys=False)

        # Save ROC data for plotting
        np.savez_compressed(
            os.path.join(output_dir, "roc_data.npz"),
            fpr=np.array(metrics["fpr"]),
            tpr=np.array(metrics["tpr"]),
            thresholds=np.array(metrics["roc_thresholds"]),
        )

        completed.add("confusion_metrics")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 6/8 complete.", prefix)

    # ---- Stage 7: OOB convergence (reuse Step 6) ----
    if "oob_convergence" not in completed:
        logger.info("[%s] Stage 7/8: Loading OOB convergence from Step 6...", prefix)
        conv_df, conv_meta = load_oob_convergence(training_dir)
        results["oob_convergence_point"] = conv_meta["convergence_point"]
        results["oob_min_error"] = conv_meta["min_oob_error"]
        completed.add("oob_convergence")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 7/8 complete.", prefix)

    # ---- Stage 8: CV stability (reuse Step 6) ----
    if "cv_stability" not in completed:
        logger.info("[%s] Stage 8/8: Loading CV stability from Step 6...", prefix)
        cv_data = load_cv_stability(training_dir)
        results["cv_stability"] = cv_data
        completed.add("cv_stability")
        state["completed"] = sorted(completed)
        state["results"] = results
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 8/8 complete.", prefix)

    logger.info("[%s] Validation pipeline complete.", prefix)
    return results


# ---------------------------------------------------------------------------
# VQI-S wrappers
# ---------------------------------------------------------------------------

def run_vqi_s_validation(
    validation_csv: str,
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    model_path: str,
    training_dir: str,
    provider_scores_dir: str,
    thresholds_yaml: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    split_name: str = "val_set",
    resume: bool = False,
) -> Dict:
    """Run VQI-S validation pipeline. Sub-tasks 7.1-7.7."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "_checkpoint_step7_s.yaml")

    return _run_validation_pipeline(
        score_type="s",
        validation_csv=validation_csv,
        features_npy=features_npy,
        feature_names_json=feature_names_json,
        selected_features_txt=selected_features_txt,
        model_path=model_path,
        training_dir=training_dir,
        provider_scores_dir=provider_scores_dir,
        thresholds_yaml=thresholds_yaml,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        resume=resume,
    )
