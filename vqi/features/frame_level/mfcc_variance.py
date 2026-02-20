"""F5: Frame-level MFCC Variance. Features 76-94.

Measures spectral shape diversity per frame: variance across MFCC
coefficients C1-C13 (excluding C0 energy).
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [20, 50, 80, 120, 170, 230, 300, 400, 550]
PREFIX = "FrameMFCCVar"


def compute_mfcc_variance_features(waveform, sr, vad_mask, intermediates):
    """Variance of MFCC coefficients 1-13 per frame."""
    mfccs = intermediates["mfccs"]  # shape (14, n_frames)
    n_frames = mfccs.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    # Variance across coefficients C1-C13 per frame
    frame_var = np.var(mfccs[1:14, :], axis=0)  # shape (n_frames,)

    speech_var = frame_var[vad]
    return aggregate_frame_features(speech_var, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
