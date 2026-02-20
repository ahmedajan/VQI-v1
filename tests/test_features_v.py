"""Unit tests for VQI-V (Voice Distinctiveness) feature extraction.

Tests:
  - Each VQI-V module with synthetic signals
  - Orchestrator returns exactly 161 features
  - Shared intermediates reuse (VQI-S -> VQI-V)
  - Edge cases
"""

import sys
import os
import warnings
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")


@pytest.fixture
def speech_like():
    """3-second signal with harmonics mimicking speech."""
    sr = 16000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    # Harmonic signal with F0=200Hz + formant-like resonances
    waveform = (0.4 * np.sin(2 * np.pi * 200 * t) +
                0.3 * np.sin(2 * np.pi * 400 * t) +
                0.15 * np.sin(2 * np.pi * 600 * t) +
                0.1 * np.sin(2 * np.pi * 800 * t) +
                0.03 * np.random.randn(len(t)))
    n_fft, hop = 512, 160
    n_frames = 1 + (len(waveform) - n_fft) // hop
    vad_mask = np.ones(n_frames, dtype=bool)
    return waveform, sr, vad_mask


@pytest.fixture
def intermediates(speech_like):
    from vqi.features.shared_intermediates import compute_shared_intermediates
    w, sr, vad = speech_like
    return compute_shared_intermediates(w, sr, vad)


# ===== Module Tests =====

class TestVQIVModules:
    def test_cepstral(self, speech_like, intermediates):
        from vqi.features_v.cepstral import compute_cepstral_features
        w, sr, vad = speech_like
        feats = compute_cepstral_features(w, sr, vad, intermediates)
        assert len(feats) == 65

    def test_lp_derived(self, speech_like, intermediates):
        from vqi.features_v.lp_derived import compute_lp_derived_features
        w, sr, vad = speech_like
        feats = compute_lp_derived_features(w, sr, vad, intermediates)
        assert len(feats) == 33

    def test_formant_identity(self, speech_like, intermediates):
        from vqi.features_v.formant_identity import compute_formant_identity_features
        w, sr, vad = speech_like
        feats = compute_formant_identity_features(w, sr, vad, intermediates)
        assert len(feats) == 11

    def test_prosodic_voice(self, speech_like, intermediates):
        from vqi.features_v.prosodic_voice import compute_prosodic_voice_features
        w, sr, vad = speech_like
        feats = compute_prosodic_voice_features(w, sr, vad, intermediates)
        assert len(feats) == 17

    def test_distributional_phase(self, speech_like, intermediates):
        from vqi.features_v.distributional_phase import compute_distributional_phase_features
        w, sr, vad = speech_like
        feats = compute_distributional_phase_features(w, sr, vad, intermediates)
        assert len(feats) == 35


# ===== Orchestrator Tests =====

class TestVQIVOrchestrator:
    def test_returns_161_features(self, speech_like):
        from vqi.core.feature_orchestrator_v import compute_all_features_v
        w, sr, vad = speech_like
        feat_dict, feat_arr = compute_all_features_v(w, sr, vad)
        assert len(feat_dict) == 161
        assert feat_arr.shape == (161,)

    def test_no_nan_inf(self, speech_like):
        from vqi.core.feature_orchestrator_v import compute_all_features_v
        w, sr, vad = speech_like
        _, feat_arr = compute_all_features_v(w, sr, vad)
        assert np.all(np.isfinite(feat_arr))

    def test_intermediates_reuse(self, speech_like):
        """VQI-V should work with intermediates from VQI-S."""
        from vqi.core.feature_orchestrator import compute_all_features
        from vqi.core.feature_orchestrator_v import compute_all_features_v
        w, sr, vad = speech_like
        # Compute VQI-S first
        _, _, intermediates = compute_all_features(w, sr, vad)
        # Then VQI-V with shared intermediates
        feat_dict, feat_arr = compute_all_features_v(w, sr, vad, intermediates)
        assert feat_arr.shape == (161,)
        assert np.all(np.isfinite(feat_arr))

    def test_feature_names_match(self, speech_like):
        from vqi.core.feature_orchestrator_v import compute_all_features_v, get_feature_names_v
        w, sr, vad = speech_like
        feat_dict, _ = compute_all_features_v(w, sr, vad)
        names = get_feature_names_v()
        assert len(names) == 161
        for name in names:
            assert name in feat_dict, f"Missing VQI-V feature: {name}"

    def test_combined_total(self, speech_like):
        """VQI-S + VQI-V = 705 features."""
        from vqi.core.feature_orchestrator import compute_all_features
        from vqi.core.feature_orchestrator_v import compute_all_features_v
        w, sr, vad = speech_like
        _, s_arr, intermediates = compute_all_features(w, sr, vad)
        _, v_arr = compute_all_features_v(w, sr, vad, intermediates)
        assert s_arr.shape[0] + v_arr.shape[0] == 705
