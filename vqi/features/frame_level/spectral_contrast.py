"""F10: Frame-level Spectral Contrast. Features 171-189.

Peak-to-valley difference across sub-bands (mean across bands per frame).
"""

import numpy as np
import librosa
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [5, 10, 15, 18, 21, 24, 28, 33, 40]
PREFIX = "FrameSC"


def compute_spectral_contrast_features(waveform, sr, vad_mask, intermediates):
    """Spectral contrast: mean peak-valley difference across 6 sub-bands."""
    S = intermediates["stft_mag"]  # magnitude spectrogram
    n_fft = intermediates["n_fft"]
    hop_length = intermediates["hop_length"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    # librosa spectral_contrast returns (n_bands+1, n_frames)
    contrast = librosa.feature.spectral_contrast(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=6,
    )
    # Mean contrast across bands per frame
    mean_contrast = np.mean(contrast, axis=0)  # (n_frames,)

    speech_sc = mean_contrast[vad]
    return aggregate_frame_features(speech_sc, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
