"""
Step 6: Random Forest Model Training

Shared pipeline for both VQI-S (Signal Quality) and VQI-V (Voice Distinctiveness):
  1. Prepare training data (select features, extract X/y)
  2. Hyperparameter grid search (OOB + CV)
  3. Train final model with best params
  4. Extract feature importances
  5. OOB convergence analysis

VQI-S wrappers at bottom; VQI-V wrappers in train_rf_v.py.
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
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Fixed RF parameters (per blueprint Section F)
FIXED_PARAMS = dict(
    criterion="gini",
    min_samples_leaf=5,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    bootstrap=True,
)

# Grid search space
N_ESTIMATORS_GRID = [200, 300, 400, 500, 750, 1000]
MAX_FEATURES_GRID = [5, 8, 10, 12, "sqrt"]


# ---------------------------------------------------------------------------
# Sub-task 6.1 / 6.7: Prepare Training Data
# ---------------------------------------------------------------------------

def prepare_training_data(
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    training_csv: str,
    output_dir: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Select features and prepare X_train, y_train arrays.

    Returns:
        X_train: (N, N_selected) float32
        y_train: (N,) int32
        selected_names: list of N_selected feature names
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load full feature matrix
    X_full = np.load(features_npy)
    with open(feature_names_json, "r", encoding="utf-8") as f:
        all_names = json.load(f)
    assert X_full.shape[1] == len(all_names), (
        f"Columns ({X_full.shape[1]}) != names ({len(all_names)})"
    )

    # Load selected feature names
    with open(selected_features_txt, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]
    n_selected = len(selected_names)

    # Map names to column indices
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = []
    for name in selected_names:
        if name not in name_to_idx:
            raise ValueError(f"Selected feature '{name}' not found in feature names")
        indices.append(name_to_idx[name])
    indices = np.array(indices)

    X_train = X_full[:, indices].astype(np.float32)

    # Load labels
    train_df = pd.read_csv(training_csv)
    assert len(train_df) == X_full.shape[0], (
        f"CSV rows ({len(train_df)}) != feature rows ({X_full.shape[0]})"
    )
    y_train = train_df["label"].values.astype(np.int32)

    # Verify
    assert not np.any(np.isnan(X_train)), "NaN found in X_train"
    assert not np.any(np.isinf(X_train)), "Inf found in X_train"
    assert set(np.unique(y_train)) == {0, 1}, f"Unexpected labels: {np.unique(y_train)}"

    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    logger.info(
        "Prepared X_train: %s, y_train: %s (Class0=%d, Class1=%d)",
        X_train.shape, y_train.shape, n0, n1,
    )

    # Save
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    with open(os.path.join(output_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(name + "\n")

    logger.info("Saved X_train.npy, y_train.npy, feature_names.txt to %s", output_dir)
    return X_train, y_train, selected_names


# ---------------------------------------------------------------------------
# Sub-task 6.2 / 6.8: Hyperparameter Grid Search
# ---------------------------------------------------------------------------

def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: str,
    n_cv_top: int = 5,
) -> Dict:
    """Grid search over n_estimators x max_features using OOB error.

    For the top n_cv_top configs by OOB, also runs 5-fold CV.

    Returns:
        best_params dict with 'n_estimators' and 'max_features' keys.
    """
    rows = []
    logger.info(
        "Starting grid search: %d x %d = %d configs",
        len(N_ESTIMATORS_GRID), len(MAX_FEATURES_GRID),
        len(N_ESTIMATORS_GRID) * len(MAX_FEATURES_GRID),
    )

    for n_est in N_ESTIMATORS_GRID:
        for max_feat in MAX_FEATURES_GRID:
            params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": max_feat}
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            oob_err = 1.0 - clf.oob_score_
            rows.append({
                "n_estimators": n_est,
                "max_features": str(max_feat),
                "oob_error": round(oob_err, 6),
                "oob_accuracy": round(clf.oob_score_, 6),
                "cv_accuracy_mean": None,
                "cv_accuracy_std": None,
            })
            logger.info(
                "  n_est=%d, max_feat=%s -> OOB_err=%.4f",
                n_est, max_feat, oob_err,
            )

    # Sort by OOB error ascending
    rows.sort(key=lambda r: r["oob_error"])

    # Run 5-fold CV for top N configs
    logger.info("Running 5-fold CV for top %d configs...", n_cv_top)
    for row in rows[:n_cv_top]:
        n_est = row["n_estimators"]
        max_feat = row["max_features"]
        # Convert string back to int or 'sqrt'
        mf = int(max_feat) if max_feat.isdigit() else max_feat
        params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": mf}
        clf = RandomForestClassifier(**params)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        row["cv_accuracy_mean"] = round(float(np.mean(cv_scores)), 6)
        row["cv_accuracy_std"] = round(float(np.std(cv_scores)), 6)
        logger.info(
            "  n_est=%d, max_feat=%s -> CV=%.4f +/- %.4f",
            n_est, max_feat, row["cv_accuracy_mean"], row["cv_accuracy_std"],
        )

    # Save grid results
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Grid search results saved to %s (%d rows)", csv_path, len(df))

    best = rows[0]
    best_mf = int(best["max_features"]) if best["max_features"].isdigit() else best["max_features"]
    best_params = {
        "n_estimators": best["n_estimators"],
        "max_features": best_mf,
    }
    logger.info(
        "Best config: n_est=%d, max_feat=%s, OOB_err=%.4f",
        best_params["n_estimators"], best_params["max_features"], best["oob_error"],
    )
    return best_params


# ---------------------------------------------------------------------------
# Sub-task 6.3 / 6.9: Train Final Model
# ---------------------------------------------------------------------------

def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict,
    output_dir: str,
    model_path: str,
) -> RandomForestClassifier:
    """Train final RF with best hyperparameters, save model and metrics."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    params = {**FIXED_PARAMS, **best_params}
    logger.info("Training final model with params: %s", params)

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    oob_err = 1.0 - clf.oob_score_
    train_acc = clf.score(X_train, y_train)
    train_preds = clf.predict(X_train)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_train, train_preds)

    logger.info("Final model: OOB_err=%.4f, train_acc=%.4f", oob_err, train_acc)
    logger.info("Confusion matrix:\n%s", cm)

    # Save model
    joblib.dump(clf, model_path)
    logger.info("Model saved to %s", model_path)

    # Save metrics
    metrics = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_features": best_params["max_features"],
        "oob_error": round(float(oob_err), 6),
        "oob_accuracy": round(float(clf.oob_score_), 6),
        "training_accuracy": round(float(train_acc), 6),
        "n_samples": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "n_class_0": int(np.sum(y_train == 0)),
        "n_class_1": int(np.sum(y_train == 1)),
        "confusion_matrix": cm.tolist(),
    }
    # Also compute class-specific metrics
    report = classification_report(y_train, train_preds, output_dict=True)
    metrics["precision_0"] = round(float(report["0"]["precision"]), 4)
    metrics["recall_0"] = round(float(report["0"]["recall"]), 4)
    metrics["precision_1"] = round(float(report["1"]["precision"]), 4)
    metrics["recall_1"] = round(float(report["1"]["recall"]), 4)

    yaml_path = os.path.join(output_dir, "training_metrics.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
    logger.info("Training metrics saved to %s", yaml_path)

    return clf


# ---------------------------------------------------------------------------
# Sub-task 6.4 / 6.10: Extract Feature Importances
# ---------------------------------------------------------------------------

def extract_importances(
    clf: RandomForestClassifier,
    feature_names: List[str],
    output_dir: str,
) -> pd.DataFrame:
    """Extract and save sorted feature importances."""
    importances = clf.feature_importances_
    assert len(importances) == len(feature_names)

    # Per-tree importances for variance
    tree_importances = np.array([t.feature_importances_ for t in clf.estimators_])

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "importance_std": np.std(tree_importances, axis=0),
        "rank": 0,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    csv_path = os.path.join(output_dir, "feature_importances.csv")
    df.to_csv(csv_path, index=False)
    logger.info(
        "Feature importances saved to %s (%d features, sum=%.4f)",
        csv_path, len(df), float(importances.sum()),
    )
    logger.info("Top 10 features:")
    for _, row in df.head(10).iterrows():
        logger.info("  #%d: %s (%.4f)", row["rank"], row["feature"], row["importance"])

    return df


# ---------------------------------------------------------------------------
# Sub-task 6.6 / 6.12: OOB Convergence Analysis
# ---------------------------------------------------------------------------

def oob_convergence_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_max_features,
    output_dir: str,
    convergence_threshold: float = 0.001,
    min_trees: int = 200,
) -> pd.DataFrame:
    """Train RFs with increasing n_estimators and record OOB convergence.

    Returns DataFrame with columns: n_estimators, oob_error, oob_accuracy.
    """
    n_est_list = [10, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
    rows = []

    logger.info("OOB convergence analysis with max_features=%s", best_max_features)
    for n_est in n_est_list:
        params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": best_max_features}
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        oob_err = 1.0 - clf.oob_score_
        rows.append({
            "n_estimators": n_est,
            "oob_error": round(oob_err, 6),
            "oob_accuracy": round(clf.oob_score_, 6),
        })
        logger.info("  n_est=%d -> OOB_err=%.4f", n_est, oob_err)

    df = pd.DataFrame(rows)

    # Identify convergence point
    min_oob = df["oob_error"].min()
    converged = df[
        (df["oob_error"] <= min_oob + convergence_threshold)
        & (df["n_estimators"] >= min_trees)
    ]
    if len(converged) > 0:
        conv_point = int(converged.iloc[0]["n_estimators"])
    else:
        conv_point = int(df.iloc[-1]["n_estimators"])
    logger.info(
        "Convergence point: %d trees (min OOB=%.4f, threshold=%.4f)",
        conv_point, min_oob, convergence_threshold,
    )

    # Save
    csv_path = os.path.join(output_dir, "oob_convergence.csv")
    df.to_csv(csv_path, index=False)
    logger.info("OOB convergence saved to %s", csv_path)

    # Save convergence metadata
    meta = {
        "convergence_point": conv_point,
        "min_oob_error": round(float(min_oob), 6),
        "best_max_features": best_max_features,
    }
    meta_path = os.path.join(output_dir, "oob_convergence_meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return df


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

STAGES = ["prepare", "grid_search", "train_final", "importances", "convergence"]


def _load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint state from YAML."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_checkpoint(checkpoint_path: str, state: Dict):
    """Save checkpoint state to YAML."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared pipeline: chains all sub-tasks with checkpointing
# ---------------------------------------------------------------------------

def _run_training_pipeline(
    score_type: str,
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    training_csv: str,
    output_dir: str,
    model_path: str,
    checkpoint_path: str,
    resume: bool = False,
) -> Dict:
    """Run the full training pipeline for one score type.

    Args:
        score_type: 's' or 'v'
        Other args: paths to input/output files.

    Returns:
        dict with pipeline results.
    """
    prefix = f"VQI-{score_type.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    state = {}
    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state:
            logger.info("[%s] Resuming from checkpoint: completed=%s", prefix, state.get("completed", []))

    completed = set(state.get("completed", []))

    # Stage 1: Prepare data
    x_path = os.path.join(output_dir, "X_train.npy")
    y_path = os.path.join(output_dir, "y_train.npy")
    names_path = os.path.join(output_dir, "feature_names.txt")

    if "prepare" not in completed:
        logger.info("[%s] Stage 1/5: Preparing training data...", prefix)
        X_train, y_train, feature_names = prepare_training_data(
            features_npy, feature_names_json, selected_features_txt,
            training_csv, output_dir,
        )
        completed.add("prepare")
        state["completed"] = sorted(completed)
        state["n_features"] = len(feature_names)
        state["n_samples"] = int(len(y_train))
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 1/5 complete. Checkpoint saved.", prefix)
    else:
        logger.info("[%s] Stage 1/5: Loading from checkpoint...", prefix)
        X_train = np.load(x_path)
        y_train = np.load(y_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = [line.strip() for line in f if line.strip()]

    # Stage 2: Grid search
    if "grid_search" not in completed:
        logger.info("[%s] Stage 2/5: Hyperparameter grid search...", prefix)
        best_params = hyperparameter_search(X_train, y_train, output_dir)
        completed.add("grid_search")
        state["completed"] = sorted(completed)
        state["best_params"] = best_params
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 2/5 complete. Checkpoint saved.", prefix)
    else:
        best_params = state["best_params"]
        logger.info("[%s] Stage 2/5: Using cached best_params=%s", prefix, best_params)

    # Stage 3: Train final model
    if "train_final" not in completed:
        logger.info("[%s] Stage 3/5: Training final model...", prefix)
        clf = train_final_model(X_train, y_train, best_params, output_dir, model_path)
        completed.add("train_final")
        state["completed"] = sorted(completed)
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 3/5 complete. Checkpoint saved.", prefix)
    else:
        logger.info("[%s] Stage 3/5: Loading model from %s", prefix, model_path)
        clf = joblib.load(model_path)

    # Stage 4: Feature importances
    if "importances" not in completed:
        logger.info("[%s] Stage 4/5: Extracting feature importances...", prefix)
        imp_df = extract_importances(clf, feature_names, output_dir)
        completed.add("importances")
        state["completed"] = sorted(completed)
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 4/5 complete. Checkpoint saved.", prefix)

    # Stage 5: OOB convergence
    if "convergence" not in completed:
        logger.info("[%s] Stage 5/5: OOB convergence analysis...", prefix)
        conv_df = oob_convergence_analysis(
            X_train, y_train, best_params["max_features"], output_dir,
        )
        completed.add("convergence")
        state["completed"] = sorted(completed)
        _save_checkpoint(checkpoint_path, state)
        logger.info("[%s] Stage 5/5 complete. Checkpoint saved.", prefix)

    # Load metrics for return
    metrics_path = os.path.join(output_dir, "training_metrics.yaml")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f)

    results = {
        "score_type": score_type,
        "n_features": len(feature_names),
        "n_samples": int(len(y_train)),
        "best_params": best_params,
        "oob_error": metrics["oob_error"],
        "oob_accuracy": metrics["oob_accuracy"],
        "training_accuracy": metrics["training_accuracy"],
        "model_path": model_path,
    }

    logger.info(
        "[%s] Pipeline complete: %d features, OOB_err=%.4f, train_acc=%.4f",
        prefix, results["n_features"], results["oob_error"], results["training_accuracy"],
    )
    return results


# ---------------------------------------------------------------------------
# VQI-S wrappers
# ---------------------------------------------------------------------------

def run_vqi_s_pipeline(
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    training_csv: str,
    output_dir: str,
    model_path: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict:
    """Run VQI-S (Signal Quality) training pipeline. Sub-tasks 6.1-6.4, 6.6."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "_checkpoint_step6_s.yaml")

    return _run_training_pipeline(
        score_type="s",
        features_npy=features_npy,
        feature_names_json=feature_names_json,
        selected_features_txt=selected_features_txt,
        training_csv=training_csv,
        output_dir=output_dir,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
