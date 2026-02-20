"""G90-G92: eGeMAPS spectral features.

G90: AlphaRatio (energy 50-1000 Hz / 1000-5000 Hz)
G91: HammarbergIndex (max energy 0-2kHz / max energy 2-5kHz)
G92: SpectralSlope0500_1500
"""

import numpy as np


def compute_egemaps_spectral_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    # Mean spectrum over speech frames
    if np.sum(vad) > 0:
        mean_spec = np.mean(S[:, vad], axis=1)
    else:
        mean_spec = np.mean(S, axis=1)

    # G90: Alpha ratio
    band_50_1000 = (freqs >= 50) & (freqs <= 1000)
    band_1000_5000 = (freqs >= 1000) & (freqs <= 5000)
    e_low = np.sum(mean_spec[band_50_1000]) + eps
    e_high = np.sum(mean_spec[band_1000_5000]) + eps
    features["AlphaRatio"] = float(10.0 * np.log10(e_low / e_high))

    # G91: Hammarberg index
    band_0_2k = (freqs >= 0) & (freqs <= 2000)
    band_2k_5k = (freqs >= 2000) & (freqs <= 5000)
    max_low = np.max(mean_spec[band_0_2k]) + eps if np.any(band_0_2k) else eps
    max_high = np.max(mean_spec[band_2k_5k]) + eps if np.any(band_2k_5k) else eps
    features["HammarbergIndex"] = float(10.0 * np.log10(max_low / max_high))

    # G92: Spectral slope 500-1500 Hz
    band_500_1500 = (freqs >= 500) & (freqs <= 1500)
    if np.sum(band_500_1500) > 1:
        f_band = freqs[band_500_1500]
        s_band = 10.0 * np.log10(mean_spec[band_500_1500] + eps)
        coeffs = np.polyfit(f_band, s_band, 1)
        features["SpectralSlope0500_1500"] = float(coeffs[0])
    else:
        features["SpectralSlope0500_1500"] = 0.0

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
