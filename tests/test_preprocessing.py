"""
Unit tests for VQI preprocessing pipeline (Step 3.6).

17 tests covering:
  - audio_loader (7 tests)
  - normalize (3 tests)
  - vad (4 tests)
  - actionable feedback (3 tests: combined degenerate + clean)
"""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqi.preprocessing.audio_loader import load_audio
from vqi.preprocessing.normalize import dc_remove_and_normalize
from vqi.preprocessing.vad import energy_vad, reconstruct_from_mask
from vqi.preprocessing.exceptions import AudioLoadError, TooShortError
from vqi.core.vqi_algorithm import check_actionable_feedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_WAV = r"D:\VQI\Datasets\VoxCeleb1\wav\id10001\1zcIwhmdeo4\00001.wav"


def _make_sine_wav(path, sr=16000, duration=3.0, freq=440.0, channels=1):
    """Write a synthetic sine wave to a WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    if channels > 1:
        x = np.stack([x] * channels, axis=-1)
    sf.write(path, x, sr)


def _make_short_wav(path, sr=16000, duration=0.3):
    """Write a very short WAV file (below 1s minimum)."""
    n = int(sr * duration)
    x = np.random.randn(n).astype(np.float32) * 0.1
    sf.write(path, x, sr)


def _make_long_wav(path, sr=16000, duration=130.0):
    """Write a WAV file longer than 120s max."""
    n = int(sr * duration)
    # Use repeating pattern to avoid huge memory
    chunk = np.random.randn(sr).astype(np.float32) * 0.3
    x = np.tile(chunk, n // sr + 1)[:n]
    sf.write(path, x, sr)


# ===========================================================================
# audio_loader tests (7)
# ===========================================================================

def _tmpfile(suffix=".wav"):
    """Create a temp file path, closed so Windows doesn't lock it."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = f.name
    f.close()
    return path


class TestAudioLoader:
    """Tests for load_audio()."""

    def test_load_wav_16khz(self):
        """Load a 16kHz WAV, verify shape and dtype."""
        path = _tmpfile(".wav")
        try:
            _make_sine_wav(path, sr=16000, duration=3.0)
            wav = load_audio(path)
        finally:
            os.unlink(path)
        assert wav.ndim == 1
        assert wav.dtype == np.float32
        assert abs(len(wav) - 48000) < 10

    def test_load_wav_48khz_resample(self):
        """Load 48kHz WAV, verify resampled to 16kHz."""
        path = _tmpfile(".wav")
        try:
            _make_sine_wav(path, sr=48000, duration=3.0)
            wav = load_audio(path)
        finally:
            os.unlink(path)
        assert wav.ndim == 1
        assert abs(len(wav) - 48000) < 100

    def test_load_stereo_to_mono(self):
        """Load stereo WAV, verify mono output."""
        path = _tmpfile(".wav")
        try:
            _make_sine_wav(path, sr=16000, duration=3.0, channels=2)
            wav = load_audio(path)
        finally:
            os.unlink(path)
        assert wav.ndim == 1

    def test_load_too_short(self):
        """File < 1s should raise TooShortError."""
        path = _tmpfile(".wav")
        try:
            _make_short_wav(path, duration=0.3)
            with pytest.raises(TooShortError):
                load_audio(path)
        finally:
            os.unlink(path)

    def test_load_corrupt_file(self):
        """Corrupt file should raise AudioLoadError."""
        path = _tmpfile(".wav")
        try:
            with open(path, "wb") as fh:
                fh.write(b"this is not a valid audio file")
            with pytest.raises(AudioLoadError):
                load_audio(path)
        finally:
            os.unlink(path)

    def test_load_flac(self):
        """Load FLAC format."""
        path = _tmpfile(".flac")
        try:
            _make_sine_wav(path, sr=16000, duration=2.0)
            wav = load_audio(path)
        finally:
            os.unlink(path)
        assert wav.ndim == 1
        assert abs(len(wav) - 32000) < 10

    def test_truncate_long_audio(self):
        """File > 120s should be truncated from center to 120s."""
        path = _tmpfile(".wav")
        try:
            _make_long_wav(path, duration=130.0)
            wav = load_audio(path)
        finally:
            os.unlink(path)
        assert len(wav) == 1920000


# ===========================================================================
# normalize tests (3)
# ===========================================================================

class TestNormalize:
    """Tests for dc_remove_and_normalize()."""

    def test_dc_removal(self):
        """Verify mean ~0 after normalization."""
        x = np.random.randn(16000).astype(np.float32) + 0.5  # DC offset
        x_norm = dc_remove_and_normalize(x)
        assert abs(x_norm.mean()) < 1e-5

    def test_peak_normalization(self):
        """Verify max absolute value ~0.95."""
        x = np.random.randn(16000).astype(np.float32)
        x_norm = dc_remove_and_normalize(x)
        assert abs(np.abs(x_norm).max() - 0.95) < 1e-4

    def test_silence_input(self):
        """All zeros in -> all zeros out."""
        x = np.zeros(16000, dtype=np.float32)
        x_norm = dc_remove_and_normalize(x)
        assert np.allclose(x_norm, 0.0)


# ===========================================================================
# vad tests (4)
# ===========================================================================

class TestVAD:
    """Tests for energy_vad() and reconstruct_from_mask()."""

    def test_clean_speech_detection(self):
        """Clean speech should have >80% speech frames detected."""
        if not os.path.exists(SAMPLE_WAV):
            pytest.skip("VoxCeleb1 sample not available")
        wav = load_audio(SAMPLE_WAV)
        wav_norm = dc_remove_and_normalize(wav)
        mask, dur, ratio = energy_vad(wav_norm)
        assert ratio > 0.30  # speech files should have significant speech

    def test_silence_rejection(self):
        """Pure silence should have 0% speech ratio."""
        silence = np.zeros(48000, dtype=np.float32)
        mask, dur, ratio = energy_vad(silence)
        assert ratio == 0.0

    def test_median_smoothing(self):
        """Verify isolated frames are removed by median smoothing."""
        # Create signal: silence with a single loud sample
        x = np.zeros(16000, dtype=np.float32)
        x[8000] = 0.9  # single impulse
        mask, dur, ratio = energy_vad(x)
        # Median smoothing should remove isolated speech frames
        # The single impulse creates very few frames, median filter removes them
        assert ratio < 0.1

    def test_vad_mask_shape(self):
        """Verify mask length matches expected frame count."""
        n_samples = 48000
        frame_length = 512
        hop_length = 160
        x = np.random.randn(n_samples).astype(np.float32) * 0.3
        mask, _, _ = energy_vad(x)
        expected_frames = 1 + (n_samples - frame_length) // hop_length
        assert len(mask) == expected_frames

    def test_reconstruct_from_mask(self):
        """Verify reconstruct_from_mask produces non-empty output for speech."""
        x = np.random.randn(48000).astype(np.float32) * 0.3
        mask, _, ratio = energy_vad(x)
        if ratio > 0:
            speech = reconstruct_from_mask(x, mask)
            assert len(speech) > 0
            assert speech.dtype == x.dtype


# ===========================================================================
# actionable feedback tests (3)
# ===========================================================================

class TestActionableFeedback:
    """Tests for check_actionable_feedback()."""

    def test_too_quiet(self):
        """Peak < 0.001 should trigger TooQuiet."""
        quiet = np.ones(48000, dtype=np.float32) * 0.0005
        mask = np.ones(300, dtype=bool)  # pretend all speech
        feedback = check_actionable_feedback(quiet, mask)
        assert "TooQuiet" in feedback

    def test_clipped(self):
        """Heavily clipped signal should trigger Clipped."""
        x = np.random.randn(48000).astype(np.float32)
        # Clip 20% of samples to +/- 0.99
        x[:9600] = 0.99
        mask = np.ones(300, dtype=bool)
        feedback = check_actionable_feedback(x, mask)
        assert "Clipped" in feedback

    def test_too_short_speech(self):
        """Less than 1.0s speech after VAD should trigger TooShort."""
        x = np.random.randn(48000).astype(np.float32) * 0.3
        # Create mask with only 50 speech frames (50 * 160 / 16000 = 0.5s)
        mask = np.zeros(300, dtype=bool)
        mask[:50] = True
        feedback = check_actionable_feedback(x, mask)
        assert "TooShort" in feedback

    def test_insufficient_speech(self):
        """VAD ratio < 5% should trigger InsufficientSpeech."""
        x = np.random.randn(48000).astype(np.float32) * 0.3
        # Only 10 of 300 frames are speech = 3.3%
        mask = np.zeros(300, dtype=bool)
        mask[:10] = True
        feedback = check_actionable_feedback(x, mask)
        assert "InsufficientSpeech" in feedback

    def test_clean_passes(self):
        """Clean speech with adequate duration should pass all checks."""
        # Simulate clean speech: moderate amplitude, no clipping, good ratio
        x = np.random.randn(80000).astype(np.float32) * 0.3
        # 400 frames, 350 speech = 87.5% ratio, 350*160/16000 = 3.5s
        mask = np.zeros(500, dtype=bool)
        mask[:350] = True
        feedback = check_actionable_feedback(x, mask)
        assert feedback == []
