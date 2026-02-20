"""F11: Frame-level Spectral Bandwidth. Features 190-208."""

import numpy as np
import librosa
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [300, 500, 800, 1100, 1400, 1700, 2100, 2600, 3200]
PREFIX = "FrameSBW"


def compute_spectral_bandwidth_features(waveform, sr, vad_mask, intermediates):
    """Standard deviation of magnitude-weighted frequency distribution."""
    S = intermediates["stft_mag"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    bw = librosa.feature.spectral_bandwidth(
        S=S, sr=sr, n_fft=intermediates["n_fft"],
        hop_length=intermediates["hop_length"],
    )[0]  # shape (n_frames,)

    speech_bw = bw[vad]
    return aggregate_frame_features(speech_bw, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
