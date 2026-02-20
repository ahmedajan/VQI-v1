"""G5: Global Bandwidth (95th percentile cumulative energy frequency)."""

import numpy as np


def compute_bandwidth_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    S = intermediates["stft_power"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    # Mean spectrum over speech frames
    if np.sum(vad) == 0:
        return {"GlobalBandwidth": 0.0}
    mean_spec = np.mean(S[:, vad], axis=1)
    freqs = np.linspace(0, sr / 2, len(mean_spec))

    # 95th percentile of cumulative energy
    cumsum = np.cumsum(mean_spec)
    total = cumsum[-1] + 1e-12
    idx = np.searchsorted(cumsum, 0.95 * total)
    idx = min(idx, len(freqs) - 1)
    return {"GlobalBandwidth": float(freqs[idx])}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
