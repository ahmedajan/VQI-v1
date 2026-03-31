"""Unit tests for VQI GUI components and engine backend.

Tests the engine, feedback system, gauge widget, and export functionality
without requiring an actual GUI display (headless-safe where possible).
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from vqi.engine import VQIEngine, VQIResult
from vqi.feedback import (
    FEEDBACK_TEMPLATES_S,
    FEEDBACK_TEMPLATES_V,
    find_limiting_factors,
    compute_category_scores,
    render_plain_feedback,
    render_expert_feedback,
    generate_overall_assessment,
    generate_export_report,
)

# Use a single engine instance for all tests (expensive to load)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = VQIEngine()
    return _engine


# Find a real test file
TEST_WAV = os.path.join(
    BASE_DIR, "conformance", "test_files", "conf_000.wav"
)


# ---------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------

class TestEngine:
    def test_engine_score_file(self):
        """VQIEngine produces valid VQIResult from WAV file."""
        if not os.path.exists(TEST_WAV):
            pytest.skip("No conformance test files yet")
        engine = get_engine()
        result = engine.score_file(TEST_WAV)
        assert isinstance(result, VQIResult)
        assert 0 <= result.score_s <= 100
        assert 0 <= result.score_v <= 100
        assert result.processing_time_ms > 0
        assert "duration_s" in result.audio_info
        assert len(result.features_s) > 0
        assert len(result.features_v) > 0

    def test_engine_score_waveform(self):
        """VQIEngine scores a raw waveform directly."""
        engine = get_engine()
        # Generate a short sine wave
        sr = 16000
        t = np.linspace(0, 3, sr * 3, dtype=np.float32)
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = engine.score_waveform(waveform, sr)
        assert isinstance(result, VQIResult)
        assert 0 <= result.score_s <= 100
        assert 0 <= result.score_v <= 100

    def test_engine_invalid_file(self):
        """VQIEngine raises error for non-existent file."""
        engine = get_engine()
        with pytest.raises(Exception):
            engine.score_file("nonexistent_file_12345.wav")


# ---------------------------------------------------------------
# Feedback tests
# ---------------------------------------------------------------

class TestFeedback:
    def test_feedback_templates_complete(self):
        """All S and V categories have required template fields."""
        required = {"issue", "explain", "fix", "expert"}
        for cat, tmpl in FEEDBACK_TEMPLATES_S.items():
            for field in required:
                assert field in tmpl, f"S template '{cat}' missing '{field}'"
                assert len(tmpl[field]) > 0
        for cat, tmpl in FEEDBACK_TEMPLATES_V.items():
            for field in required:
                assert field in tmpl, f"V template '{cat}' missing '{field}'"
                assert len(tmpl[field]) > 0

    def test_feedback_plain_language(self):
        """Plain-language feedback is non-empty and has improvement section for low scores."""
        # Create mock limiting factors
        factors = [
            {"feature_name": "test", "value": 0.5, "percentile": 20,
             "category": "noise", "importance": 0.1, "score": 0.08},
        ]
        text = render_plain_feedback(factors, FEEDBACK_TEMPLATES_S, 25, "S")
        assert len(text) > 0
        assert "What to Improve" in text
        assert "Poor" in text

    def test_feedback_expert_details(self):
        """Expert feedback contains percentile references."""
        # Load real data for a realistic test
        engine = get_engine()
        pct_names = engine.percentile_names_s
        # Create mock features dict
        mock_feats = {n: 0.5 for n in pct_names[:10]}
        mock_cats = {n: "spectral" for n in pct_names[:10]}
        cat_scores = compute_category_scores(
            mock_feats, engine.percentiles_s, pct_names, mock_cats
        )
        factors = find_limiting_factors(
            mock_feats, engine.importances_s,
            engine.percentiles_s, pct_names, mock_cats
        )
        text = render_expert_feedback(
            mock_feats, engine.percentiles_s, pct_names,
            mock_cats, cat_scores, factors, 50, "S"
        )
        assert len(text) > 0
        assert "Expert Diagnostics" in text

    def test_feedback_overall_assessment(self):
        """All 9 score combinations produce text."""
        # Test all tier combinations
        test_cases = [
            (90, 90), (90, 55), (90, 20),
            (55, 90), (55, 55), (55, 20),
            (20, 90), (20, 55), (20, 20),
        ]
        seen = set()
        for s, v in test_cases:
            text = generate_overall_assessment(s, v)
            assert len(text) > 10, f"Empty assessment for ({s}, {v})"
            seen.add(text)
        assert len(seen) == 9, "Should produce 9 distinct assessments"

    def test_export_report(self):
        """Export report is non-empty and contains scores."""
        result = VQIResult(
            score_s=65, score_v=70,
            overall_assessment="Test assessment",
            audio_info={"duration_s": 5.0, "speech_duration_s": 4.0,
                        "speech_ratio": 0.8, "sample_rate": 16000, "warnings": []},
            plain_feedback_s="S feedback", plain_feedback_v="V feedback",
            expert_feedback_s="S expert", expert_feedback_v="V expert",
            category_scores_s={"noise": 70, "spectral": 60},
            category_scores_v={"cepstral": 65},
            processing_time_ms=5000.0,
        )
        report = generate_export_report(result)
        assert len(report) > 100
        assert "65/100" in report
        assert "70/100" in report
        assert "Test assessment" in report

    def test_category_scores(self):
        """Category scores are computed for all categories, values in [0, 100]."""
        engine = get_engine()
        if not os.path.exists(TEST_WAV):
            pytest.skip("No conformance test files yet")
        result = engine.score_file(TEST_WAV)
        for cat, sc in result.category_scores_s.items():
            assert 0 <= sc <= 100, f"S category '{cat}' score {sc} out of range"
        for cat, sc in result.category_scores_v.items():
            assert 0 <= sc <= 100, f"V category '{cat}' score {sc} out of range"

    def test_percentile_lookup(self):
        """Percentile tables load correctly and have expected shape."""
        engine = get_engine()
        assert engine.percentiles_s.shape == (430, 101)
        assert engine.percentiles_v.shape == (133, 101)
        assert len(engine.percentile_names_s) == 430
        assert len(engine.percentile_names_v) == 133


# ---------------------------------------------------------------
# Gauge widget tests (basic logic, no display needed)
# ---------------------------------------------------------------

class TestGaugeWidget:
    def test_gauge_widget_range(self):
        """GaugeWidget accepts 0, 50, 100 without error."""
        from vqi.gui.gauge_widget import GaugeWidget, _color_for_score, _label_for_score
        # Test color/label functions directly (no QApplication needed)
        for score in [0, 50, 100]:
            color = _color_for_score(score)
            assert color is not None
            label = _label_for_score(score)
            assert len(label) > 0

    def test_gauge_widget_colors(self):
        """Color matches expected score band."""
        from vqi.gui.gauge_widget import _color_for_score
        # Red band (0-30)
        c = _color_for_score(15)
        assert c.red() > c.green()
        # Green band (71-85)
        c = _color_for_score(80)
        assert c.green() > c.red()
        # Dark green (86-100)
        c = _color_for_score(95)
        assert c.green() > c.red()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
