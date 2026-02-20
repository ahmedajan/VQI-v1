"""F13: Frame-level Energy (dB). Features 228-246."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [-60, -50, -40, -35, -30, -25, -20, -15, -10]
PREFIX = "FrameE"


def compute_frame_energy_features(waveform, sr, vad_mask, intermediates):
    """RMS energy per frame in dB: 20*log10(RMS)."""
    frame_energy = intermediates["frame_energy"]  # RMS per frame
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    energy_db = 20.0 * np.log10(frame_energy + eps)

    speech_e = energy_db[vad]
    return aggregate_frame_features(speech_e, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
