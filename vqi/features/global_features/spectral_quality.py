"""G21-G29: Spectral quality features.

G21: LTAS_Slope
G22: LTAS_Tilt
G23: SpectralFluxMean
G24: SpectralFluxStd
G25: SpectralRolloff
G26: SpectralEntropy
G27: SpectralSkewness
G28: SpectralKurtosis
G29: SpectralCrest
"""

import numpy as np


def compute_spectral_quality_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    S = intermediates["stft_power"]
    S_mag = intermediates["stft_mag"]
    n_frames = S.shape[1]
    n_bins = S.shape[0]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12

    # LTAS (Long-Term Average Spectrum) over speech frames
    if np.sum(vad) > 0:
        ltas = np.mean(S[:, vad], axis=1)
    else:
        ltas = np.mean(S, axis=1)
    ltas_db = 10.0 * np.log10(ltas + eps)
    freqs = np.linspace(0, sr / 2, n_bins)

    # G21: LTAS Slope (linear regression in dB)
    if len(freqs) > 1:
        coeffs = np.polyfit(freqs, ltas_db, 1)
        features["LTAS_Slope"] = float(coeffs[0])
    else:
        features["LTAS_Slope"] = 0.0

    # G22: LTAS Tilt (ratio of low to high energy, split at 1kHz)
    split_idx = np.searchsorted(freqs, 1000)
    e_low = np.sum(ltas[:split_idx]) + eps
    e_high = np.sum(ltas[split_idx:]) + eps
    features["LTAS_Tilt"] = float(10.0 * np.log10(e_low / e_high))

    # G23-G24: Spectral flux mean/std (from magnitude spectrogam)
    if n_frames > 1:
        S_norm = S_mag / (np.linalg.norm(S_mag, axis=0, keepdims=True) + eps)
        diff = S_norm[:, 1:] - S_norm[:, :-1]
        flux = np.sqrt(np.sum(diff ** 2, axis=0))
        speech_flux = flux[vad[1:]] if np.any(vad[1:]) else flux
        features["SpectralFluxMean"] = float(np.mean(speech_flux))
        features["SpectralFluxStd"] = float(np.std(speech_flux))
    else:
        features["SpectralFluxMean"] = 0.0
        features["SpectralFluxStd"] = 0.0

    # G25: Spectral rolloff (mean 85th pctl frequency across speech frames)
    cumsum = np.cumsum(S, axis=0)
    total = cumsum[-1, :] + eps
    rolloff = np.zeros(n_frames)
    for j in range(n_frames):
        idx = np.searchsorted(cumsum[:, j], 0.85 * total[j])
        idx = min(idx, len(freqs) - 1)
        rolloff[j] = freqs[idx]
    speech_rolloff = rolloff[vad] if np.any(vad) else rolloff
    features["SpectralRolloff"] = float(np.mean(speech_rolloff))

    # G26: Spectral entropy (mean normalized entropy)
    S_norm2 = S / (S.sum(axis=0, keepdims=True) + eps)
    entropy = -np.sum(S_norm2 * np.log(S_norm2 + eps), axis=0) / np.log(n_bins)
    speech_entropy = entropy[vad] if np.any(vad) else entropy
    features["SpectralEntropy"] = float(np.mean(speech_entropy))

    # G27-G29: Spectral moments from LTAS
    ltas_norm = ltas / (np.sum(ltas) + eps)
    mu = np.sum(freqs * ltas_norm)
    sigma = np.sqrt(np.sum(ltas_norm * (freqs - mu) ** 2) + eps)

    # G27: Skewness
    features["SpectralSkewness"] = float(np.sum(ltas_norm * ((freqs - mu) / sigma) ** 3))

    # G28: Kurtosis (excess)
    features["SpectralKurtosis"] = float(np.sum(ltas_norm * ((freqs - mu) / sigma) ** 4) - 3.0)

    # G29: Crest factor
    features["SpectralCrest"] = float(np.max(ltas) / (np.mean(ltas) + eps))

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
