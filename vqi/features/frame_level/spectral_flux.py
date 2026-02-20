"""F12: Frame-level Spectral Flux. Features 209-227.

L2 norm of consecutive frame spectral difference.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.50, 0.80]
PREFIX = "FrameSFlux"


def compute_spectral_flux_features(waveform, sr, vad_mask, intermediates):
    """L2 spectral flux between consecutive frames."""
    S = intermediates["stft_mag"]  # magnitude spec
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    if n_frames < 2:
        return aggregate_frame_features(np.array([]), BIN_BOUNDARIES, PREFIX)

    # Normalize each frame
    eps = 1e-12
    S_norm = S / (np.linalg.norm(S, axis=0, keepdims=True) + eps)

    # Frame-to-frame L2 difference
    diff = S_norm[:, 1:] - S_norm[:, :-1]
    flux = np.sqrt(np.sum(diff ** 2, axis=0))  # (n_frames-1,)

    # Pad first frame with 0
    flux = np.concatenate([[0.0], flux])

    speech_flux = flux[vad]
    return aggregate_frame_features(speech_flux, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
