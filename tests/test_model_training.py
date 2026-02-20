"""
Unit tests for Step 6: Model Training (train_rf, random_forest).

Uses small synthetic data (200 samples, 10 features) for fast execution.
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure project root on path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.train_rf import (
    prepare_training_data,
    hyperparameter_search,
    train_final_model,
    extract_importances,
    oob_convergence_analysis,
    _run_training_pipeline,
    N_ESTIMATORS_GRID,
    MAX_FEATURES_GRID,
)
from vqi.prediction.random_forest import (
    load_model as load_model_s,
    predict_score as predict_score_s,
    predict_scores_batch as predict_scores_batch_s,
)
from vqi.prediction.random_forest_v import (
    predict_score as predict_score_v,
    predict_scores_batch as predict_scores_batch_v,
)


@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic dataset: 200 samples, 15 total features, 10 selected."""
    np.random.seed(42)
    n_samples = 200
    n_total = 15
    n_selected = 10

    X_full = np.random.randn(n_samples, n_total).astype(np.float32)
    # Make labels somewhat correlated with first feature
    probs = 1.0 / (1.0 + np.exp(-X_full[:, 0]))
    y = (probs > 0.5).astype(np.int32)

    all_names = [f"feat_{i}" for i in range(n_total)]
    selected_names = all_names[:n_selected]

    # Save files
    features_npy = str(tmp_path / "features.npy")
    np.save(features_npy, X_full)

    names_json = str(tmp_path / "feature_names.json")
    with open(names_json, "w", encoding="utf-8") as f:
        json.dump(all_names, f)

    selected_txt = str(tmp_path / "selected_features.txt")
    with open(selected_txt, "w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(name + "\n")

    training_csv = str(tmp_path / "training.csv")
    df = pd.DataFrame({
        "filename": [f"file_{i}.wav" for i in range(n_samples)],
        "label": y,
    })
    df.to_csv(training_csv, index=False)

    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "features_npy": features_npy,
        "names_json": names_json,
        "selected_txt": selected_txt,
        "training_csv": training_csv,
        "output_dir": output_dir,
        "tmp_path": str(tmp_path),
        "n_samples": n_samples,
        "n_total": n_total,
        "n_selected": n_selected,
        "X_full": X_full,
        "y": y,
        "all_names": all_names,
        "selected_names": selected_names,
    }


# ---- Data preparation tests ----

class TestPrepareTrainingData:
    def test_correct_shape(self, synthetic_data):
        d = synthetic_data
        X, y, names = prepare_training_data(
            d["features_npy"], d["names_json"], d["selected_txt"],
            d["training_csv"], d["output_dir"],
        )
        assert X.shape == (d["n_samples"], d["n_selected"])
        assert y.shape == (d["n_samples"],)
        assert len(names) == d["n_selected"]

    def test_no_nan_inf(self, synthetic_data):
        d = synthetic_data
        X, y, _ = prepare_training_data(
            d["features_npy"], d["names_json"], d["selected_txt"],
            d["training_csv"], d["output_dir"],
        )
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_label_balance(self, synthetic_data):
        d = synthetic_data
        _, y, _ = prepare_training_data(
            d["features_npy"], d["names_json"], d["selected_txt"],
            d["training_csv"], d["output_dir"],
        )
        assert set(np.unique(y)) == {0, 1}

    def test_feature_name_ordering(self, synthetic_data):
        d = synthetic_data
        _, _, names = prepare_training_data(
            d["features_npy"], d["names_json"], d["selected_txt"],
            d["training_csv"], d["output_dir"],
        )
        assert names == d["selected_names"]

    def test_output_files_created(self, synthetic_data):
        d = synthetic_data
        prepare_training_data(
            d["features_npy"], d["names_json"], d["selected_txt"],
            d["training_csv"], d["output_dir"],
        )
        assert os.path.exists(os.path.join(d["output_dir"], "X_train.npy"))
        assert os.path.exists(os.path.join(d["output_dir"], "y_train.npy"))
        assert os.path.exists(os.path.join(d["output_dir"], "feature_names.txt"))


# ---- Grid search tests ----

class TestHyperparameterSearch:
    def test_returns_valid_config(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        best = hyperparameter_search(X, y, d["output_dir"], n_cv_top=2)
        assert "n_estimators" in best
        assert "max_features" in best
        assert best["n_estimators"] in N_ESTIMATORS_GRID
        assert best["max_features"] in MAX_FEATURES_GRID or isinstance(best["max_features"], int)

    def test_csv_expected_columns(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        hyperparameter_search(X, y, d["output_dir"], n_cv_top=2)
        df = pd.read_csv(os.path.join(d["output_dir"], "grid_search_results.csv"))
        expected_cols = {"n_estimators", "max_features", "oob_error", "oob_accuracy",
                         "cv_accuracy_mean", "cv_accuracy_std"}
        assert expected_cols.issubset(set(df.columns))

    def test_best_oob_better_than_random(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        hyperparameter_search(X, y, d["output_dir"], n_cv_top=2)
        df = pd.read_csv(os.path.join(d["output_dir"], "grid_search_results.csv"))
        assert df["oob_error"].min() < 0.50  # Better than random


# ---- Training tests ----

class TestTrainFinalModel:
    def test_model_is_rf(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        model_path = os.path.join(d["tmp_path"], "model.joblib")
        clf = train_final_model(X, y, {"n_estimators": 50, "max_features": 5},
                                d["output_dir"], model_path)
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(clf, RandomForestClassifier)

    def test_model_saves_loads(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        model_path = os.path.join(d["tmp_path"], "model.joblib")
        clf = train_final_model(X, y, {"n_estimators": 50, "max_features": 5},
                                d["output_dir"], model_path)
        clf2 = joblib.load(model_path)
        assert clf2.n_estimators == 50

    def test_training_metrics_yaml(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        model_path = os.path.join(d["tmp_path"], "model.joblib")
        train_final_model(X, y, {"n_estimators": 50, "max_features": 5},
                          d["output_dir"], model_path)
        yaml_path = os.path.join(d["output_dir"], "training_metrics.yaml")
        assert os.path.exists(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        expected_keys = {"oob_error", "oob_accuracy", "training_accuracy",
                         "n_samples", "n_features", "confusion_matrix"}
        assert expected_keys.issubset(set(metrics.keys()))


# ---- Importances tests ----

class TestExtractImportances:
    def test_sorted_descending(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        df = extract_importances(clf, d["selected_names"], d["output_dir"])
        imps = df["importance"].values
        assert np.all(imps[:-1] >= imps[1:])

    def test_sum_approximately_one(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        df = extract_importances(clf, d["selected_names"], d["output_dir"])
        assert abs(df["importance"].sum() - 1.0) < 0.01

    def test_correct_feature_count(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        df = extract_importances(clf, d["selected_names"], d["output_dir"])
        assert len(df) == d["n_selected"]


# ---- Score mapping tests ----

class TestScoreMapping:
    def test_output_in_range(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        score = predict_score_s(clf, X[0])
        assert 0 <= score <= 100
        assert isinstance(score, int)

    def test_batch_scores_in_range(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        scores = predict_scores_batch_s(clf, X)
        assert scores.shape == (d["n_samples"],)
        assert np.all(scores >= 0) and np.all(scores <= 100)

    def test_v_score_in_range(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        score = predict_score_v(clf, X[0])
        assert 0 <= score <= 100


# ---- Convergence tests ----

class TestConvergence:
    def test_oob_decreases_with_trees(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        df = oob_convergence_analysis(X, y, 5, d["output_dir"])
        # OOB at 1000 trees should be <= OOB at 10 trees (with tolerance)
        oob_10 = df[df["n_estimators"] == 10]["oob_error"].values[0]
        oob_1000 = df[df["n_estimators"] == 1000]["oob_error"].values[0]
        assert oob_1000 <= oob_10 + 0.05  # Allow small tolerance

    def test_csv_expected_columns(self, synthetic_data):
        d = synthetic_data
        X = d["X_full"][:, :d["n_selected"]].astype(np.float32)
        y = d["y"]
        oob_convergence_analysis(X, y, 5, d["output_dir"])
        df = pd.read_csv(os.path.join(d["output_dir"], "oob_convergence.csv"))
        assert "n_estimators" in df.columns
        assert "oob_error" in df.columns
        assert "oob_accuracy" in df.columns


# ---- End-to-end pipeline test ----

class TestEndToEnd:
    def test_pipeline_produces_all_files(self, synthetic_data):
        d = synthetic_data
        model_path = os.path.join(d["tmp_path"], "model.joblib")
        cp_path = os.path.join(d["output_dir"], "_checkpoint.yaml")

        results = _run_training_pipeline(
            score_type="s",
            features_npy=d["features_npy"],
            feature_names_json=d["names_json"],
            selected_features_txt=d["selected_txt"],
            training_csv=d["training_csv"],
            output_dir=d["output_dir"],
            model_path=model_path,
            checkpoint_path=cp_path,
            resume=False,
        )

        expected_files = [
            "X_train.npy", "y_train.npy", "feature_names.txt",
            "grid_search_results.csv", "training_metrics.yaml",
            "feature_importances.csv", "oob_convergence.csv",
            "oob_convergence_meta.yaml",
        ]
        for fname in expected_files:
            path = os.path.join(d["output_dir"], fname)
            assert os.path.exists(path), f"Missing: {fname}"

        assert os.path.exists(model_path)
        assert results["oob_error"] < 0.50
        assert results["training_accuracy"] > 0.50
