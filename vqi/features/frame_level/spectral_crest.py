"""F17: Frame-level Spectral Crest Factor. Features 304-322.

Spectral crest = max / mean of power spectrum per frame.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [2, 4, 6, 10, 15, 22, 35, 50, 75]
PREFIX = "FrameSCF"


def compute_spectral_crest_features(waveform, sr, vad_mask, intermediates):
    """Peak-to-average ratio of power spectrum per frame."""
    S = intermediates["stft_power"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    crest = np.max(S, axis=0) / (np.mean(S, axis=0) + eps)

    speech_crest = crest[vad]
    return aggregate_frame_features(speech_crest, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
