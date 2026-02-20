"""F9: Frame-level Spectral Slope. Features 152-170."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [-0.08, -0.06, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0, 0.01]
PREFIX = "FrameSS"


def compute_spectral_slope_features(waveform, sr, vad_mask, intermediates):
    """Linear regression slope of log-power spectrum per frame."""
    S = intermediates["stft_power"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    log_S = np.log(S + eps)  # (n_bins, n_frames)
    n_bins = log_S.shape[0]
    x = np.arange(n_bins, dtype=np.float64)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)

    slopes = np.zeros(n_frames)
    for j in range(n_frames):
        y = log_S[:, j]
        slopes[j] = np.sum((x - x_mean) * (y - y.mean())) / (x_var + eps)

    speech_ss = slopes[vad]
    return aggregate_frame_features(speech_ss, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
