"""F16: Frame-level Spectral Kurtosis. Features 285-303."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [-1.0, 0.0, 1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0]
PREFIX = "FrameSKurt"


def compute_spectral_kurtosis_features(waveform, sr, vad_mask, intermediates):
    """4th moment (excess kurtosis) of spectral distribution per frame."""
    S = intermediates["stft_power"]
    n_frames = S.shape[1]
    n_bins = S.shape[0]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    kurtosis = np.zeros(n_frames)
    for j in range(n_frames):
        p = S[:, j]
        total = p.sum() + eps
        p_norm = p / total
        mu = np.sum(freqs * p_norm)
        sigma = np.sqrt(np.sum(p_norm * (freqs - mu) ** 2) + eps)
        kurtosis[j] = np.sum(p_norm * ((freqs - mu) / sigma) ** 4) - 3.0

    speech_kurt = kurtosis[vad]
    return aggregate_frame_features(speech_kurt, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
