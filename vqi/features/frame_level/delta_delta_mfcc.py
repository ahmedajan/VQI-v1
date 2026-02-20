"""F23: Frame-level Delta-Delta-MFCC Variance. Features 418-436.

Variance of delta-delta-MFCC coefficients (C1-C13) per frame.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [5, 15, 35, 70, 120, 200, 320, 500, 800]
PREFIX = "FrameDeltaDeltaMFCC"


def compute_delta_delta_mfcc_features(waveform, sr, vad_mask, intermediates):
    """Variance of delta-delta-MFCC coefficients 1-13 per frame."""
    delta2_mfcc = intermediates["delta2_mfcc"]  # shape (14, n_frames)
    n_frames = delta2_mfcc.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    frame_var = np.var(delta2_mfcc[1:14, :], axis=0)
    speech_var = frame_var[vad]
    return aggregate_frame_features(speech_var, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
