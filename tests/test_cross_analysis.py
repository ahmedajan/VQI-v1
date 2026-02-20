"""
Unit tests for Step 7.16: VQI-S x VQI-V Cross-Analysis.

Tests cover all 6 experiments (A-F) plus the verdict logic.
Run with: pytest tests/test_cross_analysis.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# Ensure project root on path
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vqi.training.cross_analysis_sv import (
    FIXED_PARAMS,
    experiment_a_combined_model,
    experiment_b_cross_correlation,
    experiment_c_importance_redistribution,
    experiment_d_ablation,
    experiment_e_cross_prediction,
    compute_verdict,
)


# ---------------------------------------------------------------------------
# Fixtures: small synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data(tmp_path):
    """Create small synthetic training data and models for testing."""
    np.random.seed(42)
    n = 200
    n_s = 20
    n_v = 10

    X_s = np.random.randn(n, n_s).astype(np.float32)
    X_v = np.random.randn(n, n_v).astype(np.float32)
    # Make some features predictive
    y = ((X_s[:, 0] + X_v[:, 0]) > 0).astype(np.int32)

    s_names = [f"S_feat_{i}" for i in range(n_s)]
    v_names = [f"V_feat_{i}" for i in range(n_v)]

    # Save feature names
    s_names_path = str(tmp_path / "s_names.txt")
    v_names_path = str(tmp_path / "v_names.txt")
    with open(s_names_path, "w") as f:
        f.write("\n".join(s_names))
    with open(v_names_path, "w") as f:
        f.write("\n".join(v_names))

    # Train and save solo models
    import joblib
    model_s_path = str(tmp_path / "model_s.joblib")
    model_v_path = str(tmp_path / "model_v.joblib")

    clf_s = RandomForestClassifier(
        **FIXED_PARAMS, n_estimators=50, max_features=5,
    )
    clf_s.fit(X_s, y)
    joblib.dump(clf_s, model_s_path)

    clf_v = RandomForestClassifier(
        **FIXED_PARAMS, n_estimators=50, max_features=3,
    )
    clf_v.fit(X_v, y)
    joblib.dump(clf_v, model_v_path)

    # Save importances CSVs
    imp_s_df = pd.DataFrame({
        "feature": s_names,
        "importance": clf_s.feature_importances_,
        "importance_std": np.zeros(n_s),
        "rank": range(1, n_s + 1),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_s_df["rank"] = range(1, n_s + 1)

    imp_v_df = pd.DataFrame({
        "feature": v_names,
        "importance": clf_v.feature_importances_,
        "importance_std": np.zeros(n_v),
        "rank": range(1, n_v + 1),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_v_df["rank"] = range(1, n_v + 1)

    imp_s_csv = str(tmp_path / "imp_s.csv")
    imp_v_csv = str(tmp_path / "imp_v.csv")
    imp_s_df.to_csv(imp_s_csv, index=False)
    imp_v_df.to_csv(imp_v_csv, index=False)

    return {
        "X_s": X_s,
        "X_v": X_v,
        "y": y,
        "s_names": s_names,
        "v_names": v_names,
        "model_s_path": model_s_path,
        "model_v_path": model_v_path,
        "imp_s_csv": imp_s_csv,
        "imp_v_csv": imp_v_csv,
        "output_dir": str(tmp_path / "output"),
        "clf_s": clf_s,
        "clf_v": clf_v,
    }


# ---------------------------------------------------------------------------
# Test Experiment A
# ---------------------------------------------------------------------------

class TestExperimentA:
    def test_combined_model_trains(self, synthetic_data, tmp_path):
        """Combined model trains and produces valid CV comparison."""
        result = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        assert "clf_combined" in result
        assert "cv_scores" in result
        assert "metrics" in result
        assert len(result["combined_names"]) == 30  # 20 S + 10 V

    def test_cv_scores_shape(self, synthetic_data):
        result = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        for config in ["S-only", "V-only", "Combined"]:
            assert len(result["cv_scores"][config]) == 5
            for score in result["cv_scores"][config]:
                assert 0.0 <= score <= 1.0

    def test_output_files_created(self, synthetic_data):
        experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        out = synthetic_data["output_dir"]
        assert os.path.exists(os.path.join(out, "cv_comparison.csv"))
        assert os.path.exists(os.path.join(out, "combined_training_metrics.yaml"))
        assert os.path.exists(os.path.join(out, "combined_grid_search.csv"))


# ---------------------------------------------------------------------------
# Test Experiment B
# ---------------------------------------------------------------------------

class TestExperimentB:
    def test_cross_correlation_shape(self, synthetic_data):
        result = experiment_b_cross_correlation(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
        )

        corr = result["corr_matrix"]
        assert corr.shape == (20, 10)  # n_s x n_v

    def test_correlations_in_range(self, synthetic_data):
        result = experiment_b_cross_correlation(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
        )

        corr = result["corr_matrix"]
        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)
        assert not np.any(np.isnan(corr))

    def test_top_pairs_output(self, synthetic_data):
        experiment_b_cross_correlation(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
        )

        csv_path = os.path.join(synthetic_data["output_dir"], "top_correlated_pairs.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) <= 50
        assert "s_feature" in df.columns
        assert "v_feature" in df.columns
        assert "spearman_rho" in df.columns


# ---------------------------------------------------------------------------
# Test Experiment C
# ---------------------------------------------------------------------------

class TestExperimentC:
    def test_importance_redistribution(self, synthetic_data):
        # First train combined model
        exp_a = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        result = experiment_c_importance_redistribution(
            clf_combined=exp_a["clf_combined"],
            combined_names=exp_a["combined_names"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            imp_s_csv=synthetic_data["imp_s_csv"],
            imp_v_csv=synthetic_data["imp_v_csv"],
            output_dir=synthetic_data["output_dir"],
        )

        red = result["redistribution"]
        # Shares should sum to ~100%
        assert abs(red["s_share_pct"] + red["v_share_pct"] - 100.0) < 0.5

    def test_importances_sum_to_one(self, synthetic_data):
        exp_a = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        imp_sum = exp_a["clf_combined"].feature_importances_.sum()
        assert abs(imp_sum - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Test Experiment D
# ---------------------------------------------------------------------------

class TestExperimentD:
    def test_block_permutation_reduces_accuracy(self, synthetic_data):
        """Permuting features should reduce accuracy (sanity check)."""
        exp_a = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        result = experiment_d_ablation(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            clf_combined=exp_a["clf_combined"],
            combined_names=exp_a["combined_names"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            n_permutation_seeds=3,
        )

        bp = result["results"]["block_permutation"]
        # At least one block should have positive drop on average
        assert bp["mean_drop_permute_s"] >= -0.1 or bp["mean_drop_permute_v"] >= -0.1

    def test_incremental_features_nondecreasing(self, synthetic_data):
        """Accuracy should generally increase with more features."""
        exp_a = experiment_a_combined_model(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
        )

        result = experiment_d_ablation(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            clf_combined=exp_a["clf_combined"],
            combined_names=exp_a["combined_names"],
            s_names=synthetic_data["s_names"],
            v_names=synthetic_data["v_names"],
            output_dir=synthetic_data["output_dir"],
            n_permutation_seeds=3,
        )

        incr_df = result["incr_df"]
        # Last row should have >= first row accuracy (with small tolerance)
        assert incr_df.iloc[-1]["oob_accuracy"] >= incr_df.iloc[0]["oob_accuracy"] - 0.1


# ---------------------------------------------------------------------------
# Test Experiment E
# ---------------------------------------------------------------------------

class TestExperimentE:
    def test_cross_prediction_metrics(self, synthetic_data):
        result = experiment_e_cross_prediction(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
            output_dir=synthetic_data["output_dir"],
        )

        cp = result["cross_pred"]
        assert 0 <= cp["agreement_rate"] <= 1
        assert -1 <= cp["cohen_kappa"] <= 1
        assert -1 <= cp["spearman_proba_rho"] <= 1
        assert 0 <= cp["s_to_v_cross_prediction_oob"] <= 1
        assert 0 <= cp["v_to_s_cross_prediction_oob"] <= 1

    def test_probas_saved(self, synthetic_data):
        experiment_e_cross_prediction(
            X_s=synthetic_data["X_s"],
            X_v=synthetic_data["X_v"],
            y=synthetic_data["y"],
            model_s_path=synthetic_data["model_s_path"],
            model_v_path=synthetic_data["model_v_path"],
            output_dir=synthetic_data["output_dir"],
        )

        npz_path = os.path.join(synthetic_data["output_dir"], "cross_prediction_probas.npz")
        assert os.path.exists(npz_path)
        data = np.load(npz_path)
        assert "proba_s" in data
        assert "proba_v" in data
        assert len(data["proba_s"]) == 200


# ---------------------------------------------------------------------------
# Test Verdict Logic
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_both_needed(self):
        exp_a = {
            "cv_results": [
                {"config": "S-only", "cv_folds": [0.78, 0.79, 0.78, 0.80, 0.79], "cv_mean": 0.788, "cv_std": 0.01},
                {"config": "V-only", "cv_folds": [0.77, 0.78, 0.77, 0.79, 0.78], "cv_mean": 0.778, "cv_std": 0.01},
                {"config": "Combined", "cv_folds": [0.81, 0.82, 0.81, 0.83, 0.82], "cv_mean": 0.818, "cv_std": 0.01},
            ]
        }
        exp_d = {"block_permutation": {"s_unique_pct": 5.0, "v_unique_pct": 3.0}}
        exp_e = {"s_to_v_cross_prediction_oob": 0.65, "v_to_s_cross_prediction_oob": 0.63}

        v = compute_verdict(exp_a, exp_d, exp_e)
        assert v["verdict"] == "BOTH_NEEDED"

    def test_one_suffices(self):
        exp_a = {
            "cv_results": [
                {"config": "S-only", "cv_folds": [0.82, 0.82, 0.82, 0.82, 0.82], "cv_mean": 0.82, "cv_std": 0.0},
                {"config": "V-only", "cv_folds": [0.80, 0.80, 0.80, 0.80, 0.80], "cv_mean": 0.80, "cv_std": 0.0},
                {"config": "Combined", "cv_folds": [0.825, 0.825, 0.825, 0.825, 0.825], "cv_mean": 0.825, "cv_std": 0.0},
            ]
        }
        exp_d = {"block_permutation": {"s_unique_pct": 1.0, "v_unique_pct": 0.5}}
        exp_e = {"s_to_v_cross_prediction_oob": 0.80, "v_to_s_cross_prediction_oob": 0.78}

        v = compute_verdict(exp_a, exp_d, exp_e)
        assert v["verdict"] == "ONE_SUFFICES"

    def test_truly_independent(self):
        exp_a = {
            "cv_results": [
                {"config": "S-only", "cv_folds": [0.78, 0.78, 0.78, 0.78, 0.78], "cv_mean": 0.78, "cv_std": 0.0},
                {"config": "V-only", "cv_folds": [0.77, 0.77, 0.77, 0.77, 0.77], "cv_mean": 0.77, "cv_std": 0.0},
                {"config": "Combined", "cv_folds": [0.785, 0.785, 0.785, 0.785, 0.785], "cv_mean": 0.785, "cv_std": 0.0},
            ]
        }
        exp_d = {"block_permutation": {"s_unique_pct": 3.0, "v_unique_pct": 2.5}}
        exp_e = {"s_to_v_cross_prediction_oob": 0.55, "v_to_s_cross_prediction_oob": 0.58}

        v = compute_verdict(exp_a, exp_d, exp_e)
        assert v["verdict"] == "TRULY_INDEPENDENT"
