"""F21: Frame-level Glottal-to-Noise Excitation (GNE). Features 380-398.

GNE estimates the ratio of glottal excitation energy to noise energy
by inverse filtering and comparing the prediction residual.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREFIX = "FrameGNE"


def compute_gne_features(waveform, sr, vad_mask, intermediates):
    """GNE via LPC inverse filtering residual analysis."""
    frames = intermediates["frames"]  # (n_fft, n_frames)
    lpc_list = intermediates["lpc_per_frame"]
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    gne_values = np.zeros(n_frames)
    for i in range(n_frames):
        frame = frames[:, i]
        lpc_coeffs = lpc_list[i]
        energy = np.sum(frame ** 2)
        if energy < 1e-12 or np.all(lpc_coeffs == 0):
            continue

        # Inverse filter: residual = signal filtered by LPC polynomial
        from scipy.signal import lfilter
        residual = lfilter(lpc_coeffs, [1.0], frame)
        residual_energy = np.sum(residual ** 2)

        # GNE ~ 1 - (residual_energy / signal_energy)
        ratio = residual_energy / (energy + 1e-12)
        gne_values[i] = max(0.0, 1.0 - ratio)

    gne_values = np.clip(gne_values, 0.0, 1.0)
    speech_gne = gne_values[vad]
    return aggregate_frame_features(speech_gne, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
