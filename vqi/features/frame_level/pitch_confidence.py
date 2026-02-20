"""F3: Frame-level Pitch Confidence (pYIN voiced probability). Features 38-56."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREFIX = "FramePC"


def compute_pitch_confidence_features(waveform, sr, vad_mask, intermediates):
    """Pitch confidence from pYIN voiced_prob (frame_length=2048).

    Note: pYIN uses frame_length=2048, so n_frames differs from the 512-based
    VAD mask.  We align the mask to pYIN's frame count.
    """
    voiced_prob = intermediates["voiced_prob"]
    n_frames = len(voiced_prob)
    vad = _align_mask(vad_mask, n_frames)

    speech_pc = voiced_prob[vad]
    return aggregate_frame_features(speech_pc, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
