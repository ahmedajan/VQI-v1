"""G64-G65: Intelligibility features.

G64: SII_Estimate (Speech Intelligibility Index estimate)
G65: ModulationSpectrumArea
"""

import numpy as np


def compute_intelligibility_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    # G64: SII estimate (simplified band importance weighted SNR)
    # SII bands and importance weights (simplified to 4 bands)
    sii_bands = [(250, 500, 0.12), (500, 1000, 0.20), (1000, 2000, 0.30), (2000, 4000, 0.38)]
    sii = 0.0
    noise_mask = ~vad

    for lo, hi, weight in sii_bands:
        band = (freqs >= lo) & (freqs < hi)
        if not np.any(band):
            continue
        speech_e = np.mean(S[band][:, vad]) if np.any(vad) else eps
        noise_e = np.mean(S[band][:, noise_mask]) if np.any(noise_mask) else eps
        snr = 10.0 * np.log10(speech_e / (noise_e + eps) + eps)
        # SII: clip SNR to [-15, 15] dB, then normalize to [0, 1]
        snr_clipped = np.clip(snr, -15, 15)
        sii += weight * (snr_clipped + 15) / 30.0
    features["SII_Estimate"] = float(np.clip(sii, 0, 1))

    # G65: Modulation spectrum area (energy in 2-8 Hz modulation band)
    envelope = intermediates["hilbert_envelope"]
    n = len(envelope)
    if n > sr // 4:
        mod_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope))) ** 2
        mod_freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        band = (mod_freqs >= 2.0) & (mod_freqs <= 8.0)
        if np.any(band):
            features["ModulationSpectrumArea"] = float(np.sum(mod_fft[band]) / (np.sum(mod_fft) + eps))
        else:
            features["ModulationSpectrumArea"] = 0.0
    else:
        features["ModulationSpectrumArea"] = 0.0

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
