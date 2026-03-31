"""
Step X1.1: Prepare Unified Training/Evaluation Data

Provides data loading utilities for all X1 model enhancement scripts.
Constructs validation arrays from step4 features + step5 feature selection + step7 labels.
Test data returns features + pair definitions (test evaluation is pair-based, not sample-level).

Usage:
    python scripts/x1_prepare_data.py
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURES_DIR = os.path.join(DATA_DIR, "step4", "features")

logger = logging.getLogger(__name__)

# Dataset names for test evaluation
TEST_DATASETS = ["voxceleb1", "vctk", "cnceleb", "vpqad", "vseadc"]


def _apply_feature_selection(X_full, score_type):
    """Select columns from full feature matrix using step5 selected features.

    Args:
        X_full: (N, 544) for S or (N, 161) for V — full feature matrix from step4
        score_type: 's' or 'v'

    Returns:
        X_selected: (N, 430) for S or (N, 133) for V
    """
    sv = "s" if score_type == "s" else "v"

    names_json = os.path.join(FEATURES_DIR, f"feature_names_{sv}.json")
    with open(names_json, "r", encoding="utf-8") as f:
        all_names = json.load(f)

    eval_dir = "evaluation" if score_type == "s" else "evaluation_v"
    sel_txt = os.path.join(DATA_DIR, "step5", eval_dir, "selected_features.txt")
    with open(sel_txt, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    assert X_full.shape[1] == len(all_names), (
        f"Feature columns ({X_full.shape[1]}) != names ({len(all_names)})"
    )

    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = np.array([name_to_idx[name] for name in selected_names])
    X_selected = X_full[:, indices].astype(np.float32)

    assert not np.any(np.isnan(X_selected)), f"NaN in selected features ({score_type})"
    assert not np.any(np.isinf(X_selected)), f"Inf in selected features ({score_type})"

    return X_selected


def load_training_data(score_type="s"):
    """Load X_train and y_train for VQI-S or VQI-V.

    Returns:
        X_train: (20288, 430) for S or (20288, 133) for V
        y_train: (20288,) binary {0, 1}
    """
    subdir = "training" if score_type == "s" else "training_v"
    base = os.path.join(DATA_DIR, "step6", "full_feature", subdir)

    X = np.load(os.path.join(base, "X_train.npy"))
    y = np.load(os.path.join(base, "y_train.npy"))

    assert X.shape[0] == y.shape[0], f"Row mismatch: X={X.shape[0]}, y={y.shape[0]}"
    assert not np.any(np.isnan(X)), "NaN in X_train"
    assert not np.any(np.isinf(X)), "Inf in X_train"
    assert set(np.unique(y)) <= {0, 1}, f"y_train not binary: {np.unique(y)}"

    return X.astype(np.float32), y.astype(np.int32)


def load_validation_data(score_type="s"):
    """Construct X_val and y_val from step4 features + step5 selection + step7 labels.

    Returns:
        X_val: (n_labeled, 430) for S or (n_labeled, 133) for V
        y_val: (n_labeled,) binary {0, 1}
    """
    sv = "s" if score_type == "s" else "v"

    # Load full validation features from step4
    feat_path = os.path.join(FEATURES_DIR, f"features_{sv}_val.npy")
    X_full = np.load(feat_path)

    # Apply feature selection
    X_selected = _apply_feature_selection(X_full, score_type)

    # Load labels from step7 validation results
    suffix = "_v" if score_type == "v" else ""
    val_csv = os.path.join(
        DATA_DIR, "step7", "full_feature",
        f"validation{'_v' if score_type == 'v' else ''}",
        f"validation_results{suffix}.csv"
    )
    df = pd.read_csv(val_csv)
    labels = df["label"].values

    # Filter to labeled samples only (drop NaN labels)
    labeled_mask = ~np.isnan(labels)
    X_val = X_selected[labeled_mask]
    y_val = labels[labeled_mask].astype(np.int32)

    assert set(np.unique(y_val)) <= {0, 1}, f"y_val not binary: {np.unique(y_val)}"
    assert not np.any(np.isnan(X_val)), "NaN in X_val"
    assert not np.any(np.isinf(X_val)), "Inf in X_val"

    return X_val, y_val


def load_test_features(score_type, dataset):
    """Load test features for a specific dataset (feature-selected).

    Note: Some test files may have NaN features (extraction failures).
    These are kept as-is — evaluation handles NaN at the pair level.

    Returns:
        X_test: (n_files, 430) for S or (n_files, 133) for V
    """
    sv = "s" if score_type == "s" else "v"
    feat_path = os.path.join(FEATURES_DIR, f"features_{sv}_test_{dataset}.npy")
    X_full = np.load(feat_path)

    # Relax NaN check for _apply_feature_selection — do selection manually
    names_json = os.path.join(FEATURES_DIR, f"feature_names_{sv}.json")
    with open(names_json, "r", encoding="utf-8") as f:
        all_names = json.load(f)

    eval_dir = "evaluation" if score_type == "s" else "evaluation_v"
    sel_txt = os.path.join(DATA_DIR, "step5", eval_dir, "selected_features.txt")
    with open(sel_txt, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = np.array([name_to_idx[name] for name in selected_names])
    X_selected = X_full[:, indices].astype(np.float32)

    n_nan = int(np.any(np.isnan(X_selected), axis=1).sum())
    if n_nan > 0:
        logger.info("Test %s/%s: %d/%d rows have NaN (extraction failures)",
                     score_type, dataset, n_nan, X_selected.shape[0])

    return X_selected


def load_test_pairs(dataset):
    """Load pair definitions for a test dataset.

    Returns:
        pairs: (n_pairs, 2) int array of file indices
        labels: (n_pairs,) binary {0=impostor, 1=genuine}
    """
    pair_path = os.path.join(
        DATA_DIR, "step8", "full_feature", "test_scores",
        f"pair_definitions_{dataset}.npz"
    )
    data = np.load(pair_path, allow_pickle=True)
    return data["pairs"], data["labels"]


def load_all_data(score_type="s"):
    """Load all training + validation data, plus test features/pairs per dataset.

    Returns dict:
        train: (X_train, y_train)
        val: (X_val, y_val)
        test: {dataset: (X_test, pairs, labels)}
    """
    result = {
        "train": load_training_data(score_type),
        "val": load_validation_data(score_type),
        "test": {},
    }
    for ds in TEST_DATASETS:
        X_test = load_test_features(score_type, ds)
        pairs, labels = load_test_pairs(ds)
        result["test"][ds] = (X_test, pairs, labels)
    return result


def _print_summary(label, X, y):
    """Print shape and class distribution."""
    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    print(f"  {label}: X={X.shape}, y={y.shape}, "
          f"class0={c0}, class1={c1}, "
          f"X range=[{X.min():.4f}, {X.max():.4f}]")


if __name__ == "__main__":
    import joblib

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("X1.1 Data Verification")
    print("=" * 70)

    for st, label in [("s", "VQI-S"), ("v", "VQI-V")]:
        print(f"\n--- {label} ---")

        # Training
        X_train, y_train = load_training_data(st)
        _print_summary("Training", X_train, y_train)

        # Validation
        X_val, y_val = load_validation_data(st)
        _print_summary("Validation", X_val, y_val)

        # Test features per dataset
        for ds in TEST_DATASETS:
            X_test = load_test_features(st, ds)
            pairs, pair_labels = load_test_pairs(ds)
            n_gen = int((pair_labels == 1).sum())
            n_imp = int((pair_labels == 0).sum())
            print(f"  Test ({ds}): X={X_test.shape}, "
                  f"pairs={pairs.shape}, gen={n_gen}, imp={n_imp}")

        # RF model check
        suffix = "_v" if st == "v" else ""
        model_name = f"vqi{'_v' if st == 'v' else ''}_rf_model.joblib"
        model_path = os.path.join(MODELS_DIR, model_name)
        clf = joblib.load(model_path)
        sample = X_train[:1]
        proba = clf.predict_proba(sample)[:, 1]
        score = int(round(proba[0] * 100))
        print(f"  RF model: {clf.n_estimators} trees, "
              f"n_features={clf.n_features_in_}, "
              f"sample score={score}")

    print("\n" + "=" * 70)
    print("All verifications PASSED")
    print("=" * 70)
