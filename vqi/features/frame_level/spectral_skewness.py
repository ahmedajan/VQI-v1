"""F15: Frame-level Spectral Skewness. Features 266-284."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 4.0]
PREFIX = "FrameSSkew"


def compute_spectral_skewness_features(waveform, sr, vad_mask, intermediates):
    """3rd moment of spectral distribution per frame."""
    S = intermediates["stft_power"]
    n_frames = S.shape[1]
    n_bins = S.shape[0]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    skewness = np.zeros(n_frames)
    for j in range(n_frames):
        p = S[:, j]
        total = p.sum() + eps
        p_norm = p / total
        mu = np.sum(freqs * p_norm)
        sigma = np.sqrt(np.sum(p_norm * (freqs - mu) ** 2) + eps)
        skewness[j] = np.sum(p_norm * ((freqs - mu) / sigma) ** 3)

    speech_skew = skewness[vad]
    return aggregate_frame_features(speech_skew, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
