"""G57-G60: Sub-band analysis features.

G57: SubbandSNR_Low  (0-300 Hz)
G58: SubbandSNR_Mid  (300-3000 Hz)
G59: SubbandSNR_High (3000-8000 Hz)
G60: LowToHighEnergyRatio
"""

import numpy as np


def compute_subband_analysis_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    n_frames = S.shape[1]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    bands = {
        "Low": (0, 300),
        "Mid": (300, 3000),
        "High": (3000, 8000),
    }

    band_energy = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            band_energy[name] = {"speech": eps, "noise": eps}
            continue
        band_power = S[mask, :]
        speech_power = np.mean(band_power[:, vad]) if np.any(vad) else eps
        noise_mask = ~vad
        noise_power = np.mean(band_power[:, noise_mask]) if np.any(noise_mask) else eps
        band_energy[name] = {"speech": float(speech_power), "noise": float(max(noise_power, eps))}

    # G57-G59: Sub-band SNR
    for name in ["Low", "Mid", "High"]:
        snr = 10.0 * np.log10(band_energy[name]["speech"] / band_energy[name]["noise"] + eps)
        features[f"SubbandSNR_{name}"] = float(np.clip(snr, -20, 60))

    # G60: Low to high energy ratio
    e_low = band_energy["Low"]["speech"]
    e_high = band_energy["High"]["speech"]
    features["LowToHighEnergyRatio"] = float(10.0 * np.log10(e_low / (e_high + eps) + eps))

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
