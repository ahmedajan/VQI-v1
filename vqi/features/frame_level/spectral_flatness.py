"""F2: Frame-level Spectral Flatness. Features 19-37."""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [0.01, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80]
PREFIX = "FrameSF"


def compute_spectral_flatness_features(waveform, sr, vad_mask, intermediates):
    """Spectral flatness = geometric_mean / arithmetic_mean of power spectrum."""
    S = intermediates["stft_power"]  # (n_fft//2+1, n_frames)
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    eps = 1e-12
    # Geometric mean via log domain
    log_mean = np.mean(np.log(S + eps), axis=0)
    arith_mean = np.mean(S, axis=0) + eps
    sf = np.exp(log_mean) / arith_mean
    sf = np.clip(sf, 0.0, 1.0)

    speech_sf = sf[vad]
    return aggregate_frame_features(speech_sf, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
