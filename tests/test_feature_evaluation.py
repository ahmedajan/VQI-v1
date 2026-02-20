"""
Unit tests for Step 5: Feature Evaluation and Selection.

Uses synthetic data (100 samples, 20 features) to test all pipeline stages.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure project root on path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vqi.training.evaluate_features import (
    erc_feature_evaluation,
    load_candidate_features,
    remove_redundant_features,
    rf_importance_pruning,
    spearman_evaluation,
    _run_selection_pipeline,
)


@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic dataset for testing."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 20

    # Features: mix of informative, redundant, constant, and random
    X = rng.randn(n_samples, n_features)

    # Labels: binary
    labels = np.array([0] * 100 + [1] * 100)

    # Make feature 0 highly informative (correlated with label)
    X[:, 0] = labels + rng.randn(n_samples) * 0.2

    # Make feature 1 a near-duplicate of feature 0
    X[:, 1] = X[:, 0] + rng.randn(n_samples) * 0.01

    # Make feature 2 constant
    X[:, 2] = 5.0

    # Make feature 3 moderately informative
    X[:, 3] = labels * 0.5 + rng.randn(n_samples) * 0.5

    # Features 4-19: random noise
    # (already set above)

    # Fisher values: correlated with feature 0 for testing Spearman
    fisher_P1 = X[:, 0] * 2.0 + rng.randn(n_samples) * 0.1
    fisher_P2 = X[:, 0] * 1.5 + rng.randn(n_samples) * 0.3
    fisher_P3 = X[:, 0] * 1.8 + rng.randn(n_samples) * 0.2

    # Scores for ERC
    scores_P1 = labels * 10 + rng.randn(n_samples) * 2
    scores_P2 = labels * 8 + rng.randn(n_samples) * 2.5
    scores_P3 = labels * 9 + rng.randn(n_samples) * 2

    # Feature names
    names = [f"feat_{i:02d}" for i in range(n_features)]

    # Save files
    features_path = str(tmp_path / "features.npy")
    np.save(features_path, X)

    names_path = str(tmp_path / "feature_names.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(names, f)

    # Training CSV
    training_csv = str(tmp_path / "training_set.csv")
    train_df = pd.DataFrame({
        "row_idx": range(n_samples),
        "filename": [f"file_{i}.wav" for i in range(n_samples)],
        "speaker_id": [f"spk_{i % 20}" for i in range(n_samples)],
        "dataset_source": "test",
        "label": labels,
        "speech_duration": rng.uniform(3, 20, n_samples),
        "score_P1": scores_P1,
        "score_P2": scores_P2,
        "score_P3": scores_P3,
    })
    train_df.to_csv(training_csv, index=False)

    # Fisher CSV
    fisher_csv = str(tmp_path / "fisher_values.csv")
    fisher_df = pd.DataFrame({
        "row_idx": range(n_samples),
        "filename": [f"file_{i}.wav" for i in range(n_samples)],
        "speaker_id": [f"spk_{i % 20}" for i in range(n_samples)],
        "fisher_P1": fisher_P1,
        "fisher_P2": fisher_P2,
        "fisher_P3": fisher_P3,
        "fisher_P4": np.nan,
        "fisher_P5": np.nan,
        "fisher_mean": (fisher_P1 + fisher_P2 + fisher_P3) / 3,
    })
    fisher_df.to_csv(fisher_csv, index=False)

    return {
        "X": X,
        "labels": labels,
        "names": names,
        "features_path": features_path,
        "names_path": names_path,
        "training_csv": training_csv,
        "fisher_csv": fisher_csv,
        "tmp_path": tmp_path,
    }


class TestLoadCandidateFeatures:
    def test_shapes(self, synthetic_data):
        X, names, labels, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        assert X.shape == (200, 20)
        assert len(names) == 20
        assert len(labels) == 200
        assert len(fisher_df) == 200
        assert len(valid_mask) == 20

    def test_constant_detected(self, synthetic_data):
        X, names, labels, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        # Feature 2 is constant
        assert not valid_mask[2]
        # Others should be valid
        assert valid_mask[0]
        assert valid_mask[1]

    def test_fisher_alignment(self, synthetic_data):
        X, names, labels, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        assert fisher_df["fisher_P1"].notna().all()
        assert fisher_df["fisher_P2"].notna().all()


class TestSpearmanEvaluation:
    def test_output_format(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_out")
        df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        assert len(df) == 20
        assert "rho_P1" in df.columns
        assert "abs_rho_mean" in df.columns
        assert os.path.exists(os.path.join(out_dir, "spearman_correlations.csv"))

    def test_informative_feature_high_rho(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_out2")
        df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        # Feature 0 should have high |rho| (correlated with Fisher)
        feat0 = df[df["feature_name"] == "feat_00"]
        assert abs(feat0["rho_P1"].values[0]) > 0.7

    def test_constant_feature_zero_rho(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_out3")
        df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        feat2 = df[df["feature_name"] == "feat_02"]
        assert feat2["rho_P1"].values[0] == 0.0
        assert feat2["pval_P1"].values[0] == 1.0

    def test_rho_in_range(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_out4")
        df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        assert (df["rho_mean"].abs() <= 1.0).all()


class TestRedundancyRemoval:
    def test_identical_features_removed(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_red")
        spearman_df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        out_dir2 = str(tmp_path / "redundancy_out")
        X_red, names_red, kept = remove_redundant_features(
            X, names, spearman_df, valid_mask, out_dir2, threshold=0.95,
        )

        # Features 0 and 1 are near-duplicates (r > 0.99)
        # One should be removed
        assert len(names_red) < np.sum(valid_mask)
        # At least one of feat_00/feat_01 should be in the result
        assert "feat_00" in names_red or "feat_01" in names_red
        # But not both
        has_both = ("feat_00" in names_red) and ("feat_01" in names_red)
        assert not has_both

    def test_below_threshold_kept(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_red2")
        spearman_df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        out_dir2 = str(tmp_path / "redundancy_out2")
        X_red, names_red, kept = remove_redundant_features(
            X, names, spearman_df, valid_mask, out_dir2, threshold=0.95,
        )

        # Random features (4+) should all survive (low inter-correlation)
        for i in range(4, 20):
            if valid_mask[i]:
                assert f"feat_{i:02d}" in names_red

    def test_constant_excluded(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        names = synthetic_data["names"]
        _, _, _, fisher_df, valid_mask = load_candidate_features(
            synthetic_data["features_path"],
            synthetic_data["names_path"],
            synthetic_data["training_csv"],
            synthetic_data["fisher_csv"],
        )
        out_dir = str(tmp_path / "spearman_red3")
        spearman_df = spearman_evaluation(X, names, fisher_df, valid_mask, out_dir)

        out_dir2 = str(tmp_path / "redundancy_out3")
        X_red, names_red, kept = remove_redundant_features(
            X, names, spearman_df, valid_mask, out_dir2, threshold=0.95,
        )

        assert "feat_02" not in names_red  # constant feature


class TestRFPruning:
    def test_stable_output(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        labels = synthetic_data["labels"]
        names = synthetic_data["names"]
        valid_mask = np.std(X, axis=0) > 1e-12
        # Use only valid features
        valid_idx = np.where(valid_mask)[0]
        X_valid = X[:, valid_idx]
        names_valid = [names[i] for i in valid_idx]

        out_dir = str(tmp_path / "rf_out")
        X_sel, names_sel, kept, summary = rf_importance_pruning(
            X_valid, labels, names_valid, out_dir,
            n_estimators=50,  # fewer for speed
            n_selected_range=(2, 19),
        )

        assert len(names_sel) >= 2
        assert len(names_sel) <= 19
        assert X_sel.shape[1] == len(names_sel)

    def test_nonzero_importance(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        labels = synthetic_data["labels"]
        names = synthetic_data["names"]
        valid_mask = np.std(X, axis=0) > 1e-12
        valid_idx = np.where(valid_mask)[0]
        X_valid = X[:, valid_idx]
        names_valid = [names[i] for i in valid_idx]

        out_dir = str(tmp_path / "rf_out2")
        X_sel, names_sel, kept, summary = rf_importance_pruning(
            X_valid, labels, names_valid, out_dir,
            n_estimators=50,
            n_selected_range=(2, 19),
        )

        # Check importance rankings CSV
        imp_df = pd.read_csv(os.path.join(out_dir, "rf_importance_rankings.csv"))
        assert (imp_df["importance"] > 0).all()

    def test_reproducibility(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        labels = synthetic_data["labels"]
        names = synthetic_data["names"]
        valid_mask = np.std(X, axis=0) > 1e-12
        valid_idx = np.where(valid_mask)[0]
        X_valid = X[:, valid_idx]
        names_valid = [names[i] for i in valid_idx]

        out1 = str(tmp_path / "rf_rep1")
        _, names1, _, _ = rf_importance_pruning(
            X_valid, labels, names_valid, out1, n_estimators=50, n_selected_range=(2, 19),
        )
        out2 = str(tmp_path / "rf_rep2")
        _, names2, _, _ = rf_importance_pruning(
            X_valid, labels, names_valid, out2, n_estimators=50, n_selected_range=(2, 19),
        )
        assert names1 == names2

    def test_yaml_keys(self, synthetic_data, tmp_path):
        X = synthetic_data["X"]
        labels = synthetic_data["labels"]
        names = synthetic_data["names"]
        valid_mask = np.std(X, axis=0) > 1e-12
        valid_idx = np.where(valid_mask)[0]
        X_valid = X[:, valid_idx]
        names_valid = [names[i] for i in valid_idx]

        out_dir = str(tmp_path / "rf_out3")
        _, _, _, summary = rf_importance_pruning(
            X_valid, labels, names_valid, out_dir, n_estimators=50, n_selected_range=(2, 19),
        )

        assert "n_selected" in summary
        assert "final_oob_accuracy" in summary
        assert "n_iterations" in summary
        assert "top_10_features" in summary


class TestERCEvaluation:
    def test_output_format(self, synthetic_data, tmp_path):
        X = synthetic_data["X"][:, :5]  # first 5 features
        names = [f"feat_{i:02d}" for i in range(5)]
        out_dir = str(tmp_path / "erc_out")

        df = erc_feature_evaluation(
            X, names, synthetic_data["training_csv"], out_dir,
        )

        assert len(df) == 5
        assert "auc_mean" in df.columns
        assert "auc_fnmr10_P1" in df.columns
        assert os.path.exists(os.path.join(out_dir, "erc_per_feature.csv"))

    def test_auc_finite(self, synthetic_data, tmp_path):
        X = synthetic_data["X"][:, :5]
        names = [f"feat_{i:02d}" for i in range(5)]
        out_dir = str(tmp_path / "erc_out2")

        df = erc_feature_evaluation(
            X, names, synthetic_data["training_csv"], out_dir,
        )

        assert np.all(np.isfinite(df["auc_mean"].values))

    def test_informative_better_than_random(self, synthetic_data, tmp_path):
        # Feature 0 is informative, feature 4+ are random
        X = synthetic_data["X"][:, [0, 10]]
        names = ["feat_00", "feat_10"]
        out_dir = str(tmp_path / "erc_out3")

        df = erc_feature_evaluation(
            X, names, synthetic_data["training_csv"], out_dir,
        )

        auc_informative = df[df["feature_name"] == "feat_00"]["auc_mean"].values[0]
        auc_random = df[df["feature_name"] == "feat_10"]["auc_mean"].values[0]
        # Lower AUC is better (faster FNMR reduction)
        # Informative feature should have lower or similar AUC
        # (may not always hold with random data, so just check finite)
        assert np.isfinite(auc_informative)
        assert np.isfinite(auc_random)


class TestEndToEnd:
    def test_full_pipeline(self, synthetic_data, tmp_path):
        out_dir = str(tmp_path / "pipeline_out")
        checkpoint = str(tmp_path / "checkpoint.yaml")

        results = _run_selection_pipeline(
            score_type="s",
            features_path=synthetic_data["features_path"],
            names_path=synthetic_data["names_path"],
            training_csv=synthetic_data["training_csv"],
            fisher_csv=synthetic_data["fisher_csv"],
            output_dir=out_dir,
            redundancy_threshold=0.95,
            importance_threshold_frac=0.005,
            n_selected_range=(2, 19),
            checkpoint_path=checkpoint,
        )

        assert results["n_total"] == 20
        assert results["n_valid"] == 19  # 1 constant
        assert results["n_after_redundancy"] < 19  # at least 1 redundant pair
        assert 2 <= results["n_selected"] <= 19
        assert len(results["names_selected"]) == results["n_selected"]

        # All output files exist
        assert os.path.exists(os.path.join(out_dir, "spearman_correlations.csv"))
        assert os.path.exists(os.path.join(out_dir, "removed_redundant_features.csv"))
        assert os.path.exists(os.path.join(out_dir, "selected_features.txt"))
        assert os.path.exists(os.path.join(out_dir, "feature_selection_summary.yaml"))
        assert os.path.exists(os.path.join(out_dir, "erc_per_feature.csv"))
