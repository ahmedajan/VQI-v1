"""F1: Frame-level SNR (Signal-to-Noise Ratio). Features 0-18."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0, 5, 10, 15, 20, 25, 30, 35, 40]
PREFIX = "FrameSNR"


def compute_snr_features(waveform, sr, vad_mask, intermediates):
    """Compute per-frame SNR and aggregate into 19 features.

    SNR estimated as frame power vs global noise floor (P10 of frame energies).
    """
    frame_energy = intermediates["frame_energy"]
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)

    # Noise floor = 10th percentile of all frame energies (robust)
    frame_power = frame_energy ** 2
    noise_power = np.percentile(frame_power[frame_power > 0], 10) if np.any(frame_power > 0) else 1e-12
    noise_power = max(noise_power, 1e-12)

    # Per-frame SNR in dB
    snr_values = 10.0 * np.log10(frame_power / noise_power + 1e-12)
    snr_values = np.clip(snr_values, -10, 60)

    # Select speech frames only
    speech_snr = snr_values[vad]
    return aggregate_frame_features(speech_snr, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    """Align VAD mask to feature frame count."""
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    # Resample by nearest-neighbor
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
