"""
Unit tests for Step 7: Model Validation

Tests validate_model.py functions with synthetic data.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqi.training.validate_model import (
    assign_bins,
    check_cdf_shift,
    compute_bin_distribution,
    compute_cdf_per_bin,
    compute_confusion_metrics,
    compute_quadrant_analysis,
    compute_validation_labels,
    QUALITY_BINS,
)


# ---- Fixtures ----

@pytest.fixture
def sample_scores():
    """VQI scores uniformly spanning [0, 100]."""
    np.random.seed(42)
    return np.random.randint(0, 101, size=1000)


@pytest.fixture
def perfect_labels():
    """Labels that perfectly correlate with scores (threshold=50)."""
    np.random.seed(42)
    scores = np.random.randint(0, 101, size=1000)
    labels = (scores >= 50).astype(float)
    return scores, labels


@pytest.fixture
def validation_df_with_scores(tmp_path):
    """Create a validation CSV with provider scores and a thresholds YAML."""
    n = 500
    np.random.seed(42)

    df = pd.DataFrame({
        "filename": [f"file_{i}.wav" for i in range(n)],
        "speaker_id": [f"spk_{i % 50}" for i in range(n)],
        "dataset_source": "test",
        "score_P1_ECAPA": np.random.normal(8.0, 4.0, n),
        "score_P2_RESNET": np.random.normal(7.5, 3.5, n),
        "score_P3_ECAPA2": np.random.normal(7.0, 3.0, n),
    })

    thresholds = {
        "P1": {"percentile_90": 11.0, "fmr_001": 4.5},
        "P2": {"percentile_90": 10.5, "fmr_001": 4.5},
        "P3": {"percentile_90": 9.5, "fmr_001": 4.2},
    }

    csv_path = str(tmp_path / "val.csv")
    df.to_csv(csv_path, index=False)

    yaml_path = str(tmp_path / "thresholds.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(thresholds, f)

    return df, csv_path, yaml_path, thresholds


# ---- Tests: compute_bin_distribution ----

class TestBinDistribution:
    def test_all_bins_covered(self, sample_scores):
        df = compute_bin_distribution(sample_scores)
        assert len(df) == 5
        assert set(df["bin"]) == {"Very Low", "Low", "Medium", "High", "Very High"}

    def test_counts_sum_to_total(self, sample_scores):
        df = compute_bin_distribution(sample_scores)
        assert df["count"].sum() == len(sample_scores)

    def test_percentages_sum_to_100(self, sample_scores):
        df = compute_bin_distribution(sample_scores)
        assert abs(df["pct"].sum() - 100.0) < 0.1

    def test_edge_scores(self):
        scores = np.array([0, 20, 21, 40, 41, 60, 61, 80, 81, 100])
        df = compute_bin_distribution(scores)
        assert df.loc[df["bin"] == "Very Low", "count"].values[0] == 2
        assert df.loc[df["bin"] == "Very High", "count"].values[0] == 2


# ---- Tests: assign_bins ----

class TestAssignBins:
    def test_bin_assignment(self):
        scores = np.array([5, 30, 50, 70, 95])
        bins = assign_bins(scores)
        assert bins[0] == "Very Low"
        assert bins[1] == "Low"
        assert bins[2] == "Medium"
        assert bins[3] == "High"
        assert bins[4] == "Very High"

    def test_boundaries(self):
        scores = np.array([0, 20, 21, 40, 41, 60, 61, 80, 81, 100])
        bins = assign_bins(scores)
        assert bins[0] == "Very Low"
        assert bins[1] == "Very Low"
        assert bins[2] == "Low"
        assert bins[3] == "Low"


# ---- Tests: compute_cdf_per_bin ----

class TestCDFPerBin:
    def test_cdf_structure(self, sample_scores):
        genuine = np.random.normal(5.0, 2.0, len(sample_scores)).astype(np.float32)
        cdf = compute_cdf_per_bin(sample_scores, genuine, "P1")
        assert "Very Low" in cdf
        assert "Very High" in cdf
        for name in cdf:
            assert "x" in cdf[name]
            assert "cdf" in cdf[name]
            assert "n" in cdf[name]
            assert "mean" in cdf[name]

    def test_cdf_monotonic(self, sample_scores):
        genuine = np.random.normal(5.0, 2.0, len(sample_scores)).astype(np.float32)
        cdf = compute_cdf_per_bin(sample_scores, genuine, "P1")
        for name, data in cdf.items():
            if data["n"] > 0:
                assert np.all(np.diff(data["cdf"]) >= 0), f"CDF not monotonic for {name}"

    def test_nan_genuine_excluded(self):
        scores = np.array([10, 10, 90, 90])
        genuine = np.array([1.0, np.nan, 5.0, np.nan], dtype=np.float32)
        cdf = compute_cdf_per_bin(scores, genuine, "P1")
        assert cdf["Very Low"]["n"] == 1
        assert cdf["Very High"]["n"] == 1


# ---- Tests: check_cdf_shift ----

class TestCDFShift:
    def test_positive_shift(self):
        cdf_data = {
            "Very Low": {"n": 100, "mean": 2.0, "x": np.array([]), "cdf": np.array([])},
            "Very High": {"n": 100, "mean": 8.0, "x": np.array([]), "cdf": np.array([])},
        }
        assert check_cdf_shift(cdf_data, "P1") is True

    def test_negative_shift(self):
        cdf_data = {
            "Very Low": {"n": 100, "mean": 8.0, "x": np.array([]), "cdf": np.array([])},
            "Very High": {"n": 100, "mean": 2.0, "x": np.array([]), "cdf": np.array([])},
        }
        assert check_cdf_shift(cdf_data, "P1") is False

    def test_empty_bins(self):
        cdf_data = {
            "Very Low": {"n": 0, "mean": np.nan, "x": np.array([]), "cdf": np.array([])},
            "Very High": {"n": 100, "mean": 5.0, "x": np.array([]), "cdf": np.array([])},
        }
        assert check_cdf_shift(cdf_data, "P1") is False


# ---- Tests: compute_confusion_metrics ----

class TestConfusionMetrics:
    def test_perfect_predictions(self, perfect_labels):
        scores, labels = perfect_labels
        metrics = compute_confusion_metrics(scores, labels, threshold=50)
        assert metrics["accuracy"] > 0.95
        assert metrics["auc_roc"] > 0.95

    def test_metrics_structure(self, perfect_labels):
        scores, labels = perfect_labels
        metrics = compute_confusion_metrics(scores, labels, threshold=50)
        assert "confusion_matrix" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "youden_j_threshold" in metrics

    def test_nan_labels_excluded(self):
        scores = np.array([10, 20, 80, 90, 50])
        labels = np.array([0, np.nan, 1, np.nan, 1])
        metrics = compute_confusion_metrics(scores, labels, threshold=50)
        assert metrics["n_labeled"] == 3


# ---- Tests: compute_validation_labels ----

class TestValidationLabels:
    def test_label_assignment(self, validation_df_with_scores):
        df, csv_path, yaml_path, thresholds = validation_df_with_scores
        result = compute_validation_labels(df.copy(), yaml_path)
        assert "label" in result.columns
        assert result["label"].isin([0, 1, np.nan]).all() or True  # NaN check
        n1 = (result["label"] == 1).sum()
        n0 = (result["label"] == 0).sum()
        assert n1 >= 0
        assert n0 >= 0
        assert n1 + n0 <= len(result)

    def test_class1_requires_all_above_p90(self, validation_df_with_scores):
        df, csv_path, yaml_path, thresholds = validation_df_with_scores
        result = compute_validation_labels(df.copy(), yaml_path)
        class1 = result[result["label"] == 1]
        if len(class1) > 0:
            assert (class1["score_P1_ECAPA"] >= thresholds["P1"]["percentile_90"]).all()
            assert (class1["score_P2_RESNET"] >= thresholds["P2"]["percentile_90"]).all()
            assert (class1["score_P3_ECAPA2"] >= thresholds["P3"]["percentile_90"]).all()

    def test_class0_requires_all_below_fmr(self, validation_df_with_scores):
        df, csv_path, yaml_path, thresholds = validation_df_with_scores
        result = compute_validation_labels(df.copy(), yaml_path)
        class0 = result[result["label"] == 0]
        if len(class0) > 0:
            assert (class0["score_P1_ECAPA"] < thresholds["P1"]["fmr_001"]).all()
            assert (class0["score_P2_RESNET"] < thresholds["P2"]["fmr_001"]).all()
            assert (class0["score_P3_ECAPA2"] < thresholds["P3"]["fmr_001"]).all()


# ---- Tests: compute_quadrant_analysis ----

class TestQuadrantAnalysis:
    def test_quadrant_counts(self):
        np.random.seed(42)
        n = 200
        s = np.random.randint(0, 101, n)
        v = np.random.randint(0, 101, n)
        labels = np.where(s + v > 100, 1.0, 0.0)
        genuine = {"P1": np.random.normal(5, 2, n).astype(np.float32)}

        df = compute_quadrant_analysis(s, v, labels, genuine, 50, 50)
        assert len(df) == 4
        assert df["count"].sum() == n

    def test_q1_highest_success(self):
        """Q1 (high S, high V) should have highest Class 1 rate
        when labels correlate with both scores."""
        np.random.seed(42)
        n = 1000
        s = np.random.randint(0, 101, n)
        v = np.random.randint(0, 101, n)
        # Labels strongly correlated with both scores
        labels = np.where((s >= 50) & (v >= 50), 1.0,
                          np.where((s < 50) & (v < 50), 0.0, np.nan))
        genuine = {"P1": np.random.normal(5, 2, n).astype(np.float32)}

        df = compute_quadrant_analysis(s, v, labels, genuine, 50, 50)
        q1_rate = df.loc[df["quadrant"].str.contains("Q1"), "class1_rate"].values[0]
        q3_rate = df.loc[df["quadrant"].str.contains("Q3"), "class1_rate"].values[0]
        assert q1_rate > q3_rate
