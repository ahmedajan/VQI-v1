"""G7: Global Spectral Centroid (mean across speech frames, Hz)."""

import numpy as np
import librosa


def compute_spectral_centroid_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    S = intermediates["stft_mag"]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    centroid = librosa.feature.spectral_centroid(
        S=S, sr=sr, n_fft=intermediates["n_fft"],
        hop_length=intermediates["hop_length"],
    )[0]
    speech_centroid = centroid[vad] if np.any(vad) else centroid
    return {"SpectralCentroid": float(np.mean(speech_centroid))}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
