"""F20: Frame-level Zero-Crossing Rate. Features 361-379."""

import numpy as np
import librosa
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
PREFIX = "FrameZCR"


def compute_zcr_frame_features(waveform, sr, vad_mask, intermediates):
    """Zero-crossing rate per frame via librosa."""
    n_fft = intermediates["n_fft"]
    hop_length = intermediates["hop_length"]

    zcr = librosa.feature.zero_crossing_rate(
        waveform, frame_length=n_fft, hop_length=hop_length,
    )[0]  # shape (n_frames,)
    n_frames = len(zcr)
    vad = _align_mask(vad_mask, n_frames)

    speech_zcr = zcr[vad]
    return aggregate_frame_features(speech_zcr, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
