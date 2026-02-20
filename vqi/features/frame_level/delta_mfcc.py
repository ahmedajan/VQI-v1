"""F22: Frame-level Delta-MFCC Variance. Features 399-417.

Variance of delta-MFCC coefficients (C1-C13) per frame.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [10, 30, 60, 100, 160, 240, 350, 500, 750]
PREFIX = "FrameDeltaMFCC"


def compute_delta_mfcc_features(waveform, sr, vad_mask, intermediates):
    """Variance of delta-MFCC coefficients 1-13 per frame."""
    delta_mfcc = intermediates["delta_mfcc"]  # shape (14, n_frames)
    n_frames = delta_mfcc.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    frame_var = np.var(delta_mfcc[1:14, :], axis=0)
    speech_var = frame_var[vad]
    return aggregate_frame_features(speech_var, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
