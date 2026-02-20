"""G95: Frequency-Weighted SNR."""

import numpy as np


def compute_fw_snr_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    noise_mask = ~vad

    # A-weighting approximation (simplified)
    # Emphasize 1-4 kHz, de-emphasize low and very high frequencies
    f_squared = freqs ** 2
    a_weight = f_squared / (f_squared + 20.6 ** 2 + eps) * f_squared / (f_squared + 12194 ** 2 + eps)
    a_weight = a_weight / (np.max(a_weight) + eps)

    if np.any(vad) and np.any(noise_mask):
        speech_spec = np.mean(S[:, vad], axis=1)
        noise_spec = np.mean(S[:, noise_mask], axis=1)
        # Weighted SNR
        weighted_signal = np.sum(a_weight * speech_spec)
        weighted_noise = np.sum(a_weight * noise_spec) + eps
        fw_snr = 10.0 * np.log10(weighted_signal / weighted_noise + eps)
    else:
        fw_snr = 30.0

    return {"FrequencyWeightedSNR": float(np.clip(fw_snr, -10, 60))}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
