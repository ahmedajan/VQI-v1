"""F4: Frame-level Harmonics-to-Noise Ratio (Praat). Features 57-75."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [-5, 0, 5, 10, 15, 20, 25, 30, 35]
PREFIX = "FrameHNR"


def compute_hnr_features(waveform, sr, vad_mask, intermediates):
    """HNR from Praat harmonicity_ac, clipped to [-10, 40] dB.

    Praat uses time_step=0.01 (100 fps), so frame count differs from VAD.
    """
    harmonicity = intermediates.get("praat_harmonicity")
    if harmonicity is None:
        return _zeros()

    # Extract HNR values from Praat harmonicity object
    # Praat Matrix: values shape is (1, n_frames); index [0, i] for frame i
    n_harm_frames = harmonicity.n_frames
    hnr_values = np.array([
        harmonicity.values[0, i] for i in range(n_harm_frames)
    ])
    # Praat returns -200 for unvoiced; replace with -10
    hnr_values = np.where(hnr_values < -10, -10.0, hnr_values)
    hnr_values = np.clip(hnr_values, -10, 40)

    vad = _align_mask(vad_mask, n_harm_frames)
    speech_hnr = hnr_values[vad]
    return aggregate_frame_features(speech_hnr, BIN_BOUNDARIES, PREFIX)


def _zeros():
    from ..histogram import aggregate_frame_features as agg
    return agg(np.array([]), BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
