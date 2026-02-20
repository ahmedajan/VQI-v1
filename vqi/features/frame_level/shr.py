"""F18: Frame-level Subharmonic-to-Harmonic Ratio (SHR). Features 323-341.

SHR detects subharmonic energy (period doubling / vocal fry) via ACF.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
PREFIX = "FrameSHR"


def compute_shr_features(waveform, sr, vad_mask, intermediates):
    """Subharmonic-to-harmonic ratio via autocorrelation analysis."""
    frames = intermediates["frames"]
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    # Pitch lag range
    min_lag = int(sr / 500)
    max_lag = int(sr / 60)
    max_lag = min(max_lag, frames.shape[0] - 1)

    shr_values = np.zeros(n_frames)
    for i in range(n_frames):
        frame = frames[:, i]
        energy = np.sum(frame ** 2)
        if energy < 1e-12:
            continue

        acf = np.correlate(frame, frame, mode="full")
        acf = acf[len(frame) - 1:]
        acf_norm = acf / (energy + 1e-12)

        lo = min(min_lag, len(acf_norm) - 1)
        hi = min(max_lag + 1, len(acf_norm))
        if hi <= lo:
            continue

        # Find harmonic peak (fundamental period)
        harmonic_peak_idx = lo + np.argmax(acf_norm[lo:hi])
        harmonic_peak = acf_norm[harmonic_peak_idx]

        # Subharmonic peak at ~2x the fundamental period
        sub_lo = harmonic_peak_idx * 2 - max_lag // 10
        sub_hi = harmonic_peak_idx * 2 + max_lag // 10
        sub_lo = max(sub_lo, 0)
        sub_hi = min(sub_hi, len(acf_norm))
        if sub_hi <= sub_lo or sub_lo >= len(acf_norm):
            continue

        subharmonic_peak = np.max(acf_norm[sub_lo:sub_hi])
        shr_values[i] = subharmonic_peak / (harmonic_peak + 1e-12)

    shr_values = np.clip(shr_values, 0.0, 2.0)
    speech_shr = shr_values[vad]
    return aggregate_frame_features(speech_shr, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
