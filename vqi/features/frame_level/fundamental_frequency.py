"""F19: Frame-level Fundamental Frequency (F0). Features 342-360.

pYIN F0 in Hz, NaN replaced with 0 (unvoiced frames).
Uses frame_length=2048, so frame count differs from VAD mask.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [50, 80, 100, 120, 150, 180, 220, 280, 380]
PREFIX = "FrameF0"


def compute_fundamental_frequency_features(waveform, sr, vad_mask, intermediates):
    """Per-frame F0 from pYIN."""
    f0 = intermediates["f0"]  # NaN already replaced with 0 in shared_intermediates
    n_frames = len(f0)
    vad = _align_mask(vad_mask, n_frames)

    speech_f0 = f0[vad]
    return aggregate_frame_features(speech_f0, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
