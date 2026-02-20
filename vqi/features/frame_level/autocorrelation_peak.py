"""F14: Frame-level Autocorrelation Peak. Features 247-265.

Maximum normalized autocorrelation in the pitch range (60-500 Hz).
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREFIX = "FrameAC"


def compute_autocorrelation_peak_features(waveform, sr, vad_mask, intermediates):
    """Max normalized ACF in pitch lag range per frame."""
    frames = intermediates["frames"]  # (n_fft, n_frames)
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    # Pitch range: 60-500 Hz -> lag range in samples
    min_lag = int(sr / 500)   # ~32
    max_lag = int(sr / 60)    # ~267
    max_lag = min(max_lag, frames.shape[0] - 1)

    ac_peaks = np.zeros(n_frames)
    for i in range(n_frames):
        frame = frames[:, i]
        energy = np.sum(frame ** 2)
        if energy < 1e-12:
            continue
        # Normalized autocorrelation
        acf = np.correlate(frame, frame, mode="full")
        acf = acf[len(frame) - 1:]  # positive lags only
        acf = acf / (energy + 1e-12)
        # Search in pitch range
        lo = min(min_lag, len(acf) - 1)
        hi = min(max_lag + 1, len(acf))
        if hi > lo:
            ac_peaks[i] = np.max(acf[lo:hi])

    speech_ac = ac_peaks[vad]
    return aggregate_frame_features(speech_ac, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
