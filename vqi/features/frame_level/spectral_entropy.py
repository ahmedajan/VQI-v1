"""F7: Frame-level Spectral Entropy. Features 114-132."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85]
PREFIX = "FrameSE"


def compute_spectral_entropy_features(waveform, sr, vad_mask, intermediates):
    """Shannon entropy of the normalized power spectrum per frame."""
    S = intermediates["stft_power"]  # (n_fft//2+1, n_frames)
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    # Normalize each frame's spectrum to a probability distribution
    S_norm = S / (S.sum(axis=0, keepdims=True) + eps)
    # Shannon entropy (normalized by log of number of bins)
    n_bins = S.shape[0]
    entropy = -np.sum(S_norm * np.log(S_norm + eps), axis=0) / np.log(n_bins)

    speech_se = entropy[vad]
    return aggregate_frame_features(speech_se, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
