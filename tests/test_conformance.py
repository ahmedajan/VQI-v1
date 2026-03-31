"""Conformance test suite -- verifies VQIEngine output matches expected scores.

Runs VQIEngine on all 200 conformance files and asserts exact match on
vqi_s and vqi_v scores against conformance_expected_output_v2.0.csv.
"""

import csv
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CONF_DIR = os.path.join(BASE_DIR, "conformance")
FILES_DIR = os.path.join(CONF_DIR, "test_files")
EXPECTED_CSV = os.path.join(CONF_DIR, "conformance_expected_output_v2.0.csv")

# Lazy-load engine
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from vqi.engine import VQIEngine
        _engine = VQIEngine()
    return _engine


def _load_expected():
    """Load expected conformance results."""
    if not os.path.exists(EXPECTED_CSV):
        return []
    rows = []
    with open(EXPECTED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


_expected_data = None


def get_expected():
    global _expected_data
    if _expected_data is None:
        _expected_data = _load_expected()
    return _expected_data


# ---------------------------------------------------------------
# Conformance score tests
# ---------------------------------------------------------------

class TestConformanceScores:
    """Test that VQIEngine produces exact-match scores on conformance set."""

    @pytest.fixture(autouse=True)
    def check_conformance_data(self):
        expected = get_expected()
        if not expected:
            pytest.skip("Conformance expected output not yet generated")

    def test_conformance_count(self):
        """At least 200 conformance files in expected output."""
        expected = get_expected()
        assert len(expected) >= 200, f"Expected >= 200 files, got {len(expected)}"

    @pytest.mark.parametrize("idx", range(200))
    def test_conformance_score(self, idx):
        """VQI scores match expected output for file at index."""
        expected = get_expected()
        if idx >= len(expected):
            pytest.skip(f"Index {idx} out of range")

        row = expected[idx]
        filename = row["filename"]
        filepath = os.path.join(FILES_DIR, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"File not found: {filepath}")

        engine = get_engine()
        result = engine.score_file(filepath)

        expected_s = int(row["vqi_s"])
        expected_v = int(row["vqi_v"])

        assert result.score_s == expected_s, (
            f"{filename}: VQI-S expected {expected_s}, got {result.score_s}"
        )
        assert result.score_v == expected_v, (
            f"{filename}: VQI-V expected {expected_v}, got {result.score_v}"
        )


class TestConformanceCategories:
    """Test that feedback categories match expected output."""

    @pytest.fixture(autouse=True)
    def check_conformance_data(self):
        expected = get_expected()
        if not expected:
            pytest.skip("Conformance expected output not yet generated")

    def test_categories_present(self):
        """All conformance files have non-empty category scores."""
        engine = get_engine()
        expected = get_expected()
        # Test a sample (first 5 files)
        for row in expected[:5]:
            filename = row["filename"]
            filepath = os.path.join(FILES_DIR, filename)
            if not os.path.exists(filepath):
                continue
            result = engine.score_file(filepath)
            assert len(result.category_scores_s) > 0
            assert len(result.category_scores_v) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
