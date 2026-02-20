"""Unit tests for VQI-S feature extraction (Step 4).

Tests:
  - Histogram aggregation with known arrays
  - Each frame feature module with synthetic signals
  - Orchestrator returns exactly 544 features in correct order
  - Edge cases: short speech, pure noise, silence
"""

import sys
import os
import warnings
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")


@pytest.fixture
def sine_wave():
    """3-second 200Hz sine wave with harmonics + light noise."""
    sr = 16000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 200 * t) +
                0.3 * np.sin(2 * np.pi * 400 * t) +
                0.1 * np.sin(2 * np.pi * 600 * t) +
                0.02 * np.random.randn(len(t)))
    n_fft, hop = 512, 160
    n_frames = 1 + (len(waveform) - n_fft) // hop
    vad_mask = np.ones(n_frames, dtype=bool)
    return waveform, sr, vad_mask


@pytest.fixture
def intermediates(sine_wave):
    """Pre-computed shared intermediates."""
    from vqi.features.shared_intermediates import compute_shared_intermediates
    waveform, sr, vad_mask = sine_wave
    return compute_shared_intermediates(waveform, sr, vad_mask)


# ===== Histogram Tests =====

class TestHistogram:
    def test_basic_aggregation(self):
        from vqi.features.histogram import aggregate_frame_features
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0] * 10)
        bins = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        feats = aggregate_frame_features(data, bins, "Test")
        assert len(feats) == 19
        assert abs(feats["Test_Mean"] - 5.0) < 0.01
        assert feats["Test_Range"] == 8.0

    def test_degenerate_short(self):
        from vqi.features.histogram import aggregate_frame_features
        data = np.array([1.0, 2.0])  # < 10 frames
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        feats = aggregate_frame_features(data, bins, "Short")
        assert len(feats) == 19
        assert all(v == 0.0 for v in feats.values())

    def test_constant_data(self):
        from vqi.features.histogram import aggregate_frame_features
        data = np.ones(100) * 5.0
        bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        feats = aggregate_frame_features(data, bins, "Const")
        assert feats["Const_Std"] == 0.0
        assert feats["Const_Range"] == 0.0
        assert feats["Const_Skew"] == 0.0  # Not NaN
        assert feats["Const_Kurt"] == 0.0  # Not NaN

    def test_histogram_sums_to_one(self):
        from vqi.features.histogram import aggregate_frame_features
        data = np.random.randn(1000)
        bins = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        feats = aggregate_frame_features(data, bins, "Norm")
        hist_sum = sum(feats[f"Norm_Hist{i}"] for i in range(10))
        assert abs(hist_sum - 1.0) < 1e-10


# ===== Frame Feature Tests =====

class TestFrameFeatures:
    def test_snr(self, sine_wave, intermediates):
        from vqi.features.frame_level.snr import compute_snr_features
        w, sr, vad = sine_wave
        feats = compute_snr_features(w, sr, vad, intermediates)
        assert len(feats) == 19
        assert feats["FrameSNR_Mean"] > 0  # sine has positive SNR

    def test_spectral_flatness(self, sine_wave, intermediates):
        from vqi.features.frame_level.spectral_flatness import compute_spectral_flatness_features
        feats = compute_spectral_flatness_features(*sine_wave, intermediates)
        assert len(feats) == 19
        assert 0 <= feats["FrameSF_Mean"] <= 1

    def test_pitch_confidence(self, sine_wave, intermediates):
        from vqi.features.frame_level.pitch_confidence import compute_pitch_confidence_features
        feats = compute_pitch_confidence_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_hnr(self, sine_wave, intermediates):
        from vqi.features.frame_level.hnr import compute_hnr_features
        feats = compute_hnr_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_mfcc_variance(self, sine_wave, intermediates):
        from vqi.features.frame_level.mfcc_variance import compute_mfcc_variance_features
        feats = compute_mfcc_variance_features(*sine_wave, intermediates)
        assert len(feats) == 19
        assert feats["FrameMFCCVar_Mean"] >= 0

    def test_cpp(self, sine_wave, intermediates):
        from vqi.features.frame_level.cpp import compute_cpp_features
        feats = compute_cpp_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_spectral_entropy(self, sine_wave, intermediates):
        from vqi.features.frame_level.spectral_entropy import compute_spectral_entropy_features
        feats = compute_spectral_entropy_features(*sine_wave, intermediates)
        assert len(feats) == 19
        assert 0 <= feats["FrameSE_Mean"] <= 1

    def test_spectral_rolloff(self, sine_wave, intermediates):
        from vqi.features.frame_level.spectral_rolloff import compute_spectral_rolloff_features
        feats = compute_spectral_rolloff_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_frame_energy(self, sine_wave, intermediates):
        from vqi.features.frame_level.frame_energy import compute_frame_energy_features
        feats = compute_frame_energy_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_zcr(self, sine_wave, intermediates):
        from vqi.features.frame_level.zcr_frame import compute_zcr_frame_features
        feats = compute_zcr_frame_features(*sine_wave, intermediates)
        assert len(feats) == 19

    def test_all_frame_modules_return_19(self, sine_wave, intermediates):
        from vqi.features.frame_level import FRAME_FEATURE_MODULES
        w, sr, vad = sine_wave
        for func, prefix in FRAME_FEATURE_MODULES:
            feats = func(w, sr, vad, intermediates)
            assert len(feats) == 19, f"{prefix} returned {len(feats)} features"


# ===== Orchestrator Tests =====

class TestOrchestrator:
    def test_returns_544_features(self, sine_wave):
        from vqi.core.feature_orchestrator import compute_all_features
        w, sr, vad = sine_wave
        feat_dict, feat_arr, inter = compute_all_features(w, sr, vad)
        assert len(feat_dict) == 544
        assert feat_arr.shape == (544,)

    def test_no_nan_inf(self, sine_wave):
        from vqi.core.feature_orchestrator import compute_all_features
        w, sr, vad = sine_wave
        _, feat_arr, _ = compute_all_features(w, sr, vad)
        assert np.all(np.isfinite(feat_arr))

    def test_feature_names_match(self, sine_wave):
        from vqi.core.feature_orchestrator import compute_all_features, get_feature_names_s
        w, sr, vad = sine_wave
        feat_dict, _, _ = compute_all_features(w, sr, vad)
        names = get_feature_names_s()
        assert len(names) == 544
        # All names should be in the dict
        for name in names:
            assert name in feat_dict, f"Missing feature: {name}"


# ===== Edge Cases =====

class TestEdgeCases:
    def test_white_noise(self):
        from vqi.core.feature_orchestrator import compute_all_features
        sr = 16000
        waveform = 0.1 * np.random.randn(3 * sr)
        n_frames = 1 + (len(waveform) - 512) // 160
        vad_mask = np.ones(n_frames, dtype=bool)
        feat_dict, feat_arr, _ = compute_all_features(waveform, sr, vad_mask)
        assert feat_arr.shape == (544,)
        assert np.all(np.isfinite(feat_arr))

    def test_short_signal(self):
        from vqi.core.feature_orchestrator import compute_all_features
        sr = 16000
        # 1 second signal
        waveform = 0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, sr))
        n_frames = 1 + (len(waveform) - 512) // 160
        vad_mask = np.ones(n_frames, dtype=bool)
        feat_dict, feat_arr, _ = compute_all_features(waveform, sr, vad_mask)
        assert feat_arr.shape == (544,)
        assert np.all(np.isfinite(feat_arr))
