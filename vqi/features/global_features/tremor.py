"""G79-G82: Tremor features.

G79: Tremor_Freq (dominant tremor frequency)
G80: Tremor_Intensity (tremor amplitude)
G81: Tremor_CycleVariation (cycle-to-cycle variation)
G82: Tremor_Regularity
"""

import numpy as np


def compute_tremor_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    f0 = intermediates["f0"]
    voiced = f0 > 0
    hop = intermediates["hop_length"]
    eps = 1e-12

    if np.sum(voiced) < 20:
        return {
            "Tremor_Freq": 0.0, "Tremor_Intensity": 0.0,
            "Tremor_CycleVariation": 0.0, "Tremor_Regularity": 0.0,
        }

    # Extract F0 contour for voiced frames
    f0_voiced = f0[voiced]
    f0_rate = sr / hop  # frame rate

    # Remove mean (detrend)
    f0_detrended = f0_voiced - np.mean(f0_voiced)

    # FFT of F0 contour to find tremor frequency
    n = len(f0_detrended)
    fft_vals = np.abs(np.fft.rfft(f0_detrended))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / f0_rate)

    # Tremor range: 3-12 Hz
    tremor_mask = (fft_freqs >= 3.0) & (fft_freqs <= 12.0)
    if not np.any(tremor_mask):
        return {
            "Tremor_Freq": 0.0, "Tremor_Intensity": 0.0,
            "Tremor_CycleVariation": 0.0, "Tremor_Regularity": 0.0,
        }

    tremor_fft = fft_vals[tremor_mask]
    tremor_freqs = fft_freqs[tremor_mask]

    # G79: Dominant tremor frequency
    peak_idx = np.argmax(tremor_fft)
    features["Tremor_Freq"] = float(tremor_freqs[peak_idx])

    # G80: Tremor intensity (peak tremor power / total)
    total_power = np.sum(fft_vals ** 2) + eps
    tremor_power = np.max(tremor_fft) ** 2
    features["Tremor_Intensity"] = float(tremor_power / total_power)

    # G81: Cycle-to-cycle variation
    # Variation in successive pitch periods
    periods = 1.0 / (f0_voiced + eps)
    if len(periods) > 1:
        period_diffs = np.abs(np.diff(periods))
        features["Tremor_CycleVariation"] = float(np.mean(period_diffs) / (np.mean(periods) + eps))
    else:
        features["Tremor_CycleVariation"] = 0.0

    # G82: Tremor regularity (how periodic is the tremor)
    # Autocorrelation of F0 contour at tremor frequency
    if len(f0_detrended) > 10:
        acf = np.correlate(f0_detrended, f0_detrended, mode="full")
        acf = acf[len(f0_detrended) - 1:]
        acf = acf / (acf[0] + eps)
        # Check for peak at expected tremor period
        tremor_period = int(f0_rate / features["Tremor_Freq"]) if features["Tremor_Freq"] > 0 else 0
        if tremor_period > 0 and tremor_period < len(acf):
            features["Tremor_Regularity"] = float(max(0, acf[tremor_period]))
        else:
            features["Tremor_Regularity"] = 0.0
    else:
        features["Tremor_Regularity"] = 0.0

    return features
