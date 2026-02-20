"""F8: Frame-level Spectral Rolloff (85th percentile). Features 133-151."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5500, 7000]
PREFIX = "FrameSR"


def compute_spectral_rolloff_features(waveform, sr, vad_mask, intermediates):
    """85th-percentile frequency of cumulative spectral energy."""
    S = intermediates["stft_power"]
    n_fft = intermediates["n_fft"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    freqs = np.linspace(0, sr / 2, S.shape[0])
    cumsum = np.cumsum(S, axis=0)
    total = cumsum[-1, :] + 1e-12
    # 85th percentile
    threshold = 0.85 * total

    rolloff = np.zeros(n_frames)
    for j in range(n_frames):
        idx = np.searchsorted(cumsum[:, j], threshold[j])
        idx = min(idx, len(freqs) - 1)
        rolloff[j] = freqs[idx]

    speech_sr = rolloff[vad]
    return aggregate_frame_features(speech_sr, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
