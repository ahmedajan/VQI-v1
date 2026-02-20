"""G72-G78: MDVP-style voice quality features.

G72: MDVP_Fo (fundamental frequency)
G73: MDVP_Jitter (jitter %)
G74: MDVP_Shimmer (shimmer %)
G75: MDVP_NHR
G76: MDVP_VTI (Voice Turbulence Index)
G77: MDVP_SPI (Soft Phonation Index)
G78: MDVP_DVB (Degree of Voice Breaks)
"""

import numpy as np
from parselmouth.praat import call


def compute_voice_quality_mdvp_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    f0 = intermediates["f0"]
    voiced = f0 > 0
    snd = intermediates.get("praat_sound")
    pp = intermediates.get("praat_point_process")
    harmonicity = intermediates.get("praat_harmonicity")

    # G72: MDVP Fo
    features["MDVP_Fo"] = float(np.mean(f0[voiced])) if np.any(voiced) else 0.0

    # G73: MDVP Jitter (%)
    if pp is not None:
        try:
            features["MDVP_Jitter"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100)
        except Exception:
            features["MDVP_Jitter"] = 0.0
    else:
        features["MDVP_Jitter"] = 0.0

    # G74: MDVP Shimmer (%)
    if pp is not None and snd is not None:
        try:
            features["MDVP_Shimmer"] = float(
                call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
            )
        except Exception:
            features["MDVP_Shimmer"] = 0.0
    else:
        features["MDVP_Shimmer"] = 0.0

    # G75: MDVP NHR
    if harmonicity is not None:
        n_h = harmonicity.n_frames
        hnr_vals = np.array([harmonicity.values[0, i] for i in range(n_h)])
        valid = hnr_vals[hnr_vals > -100]
        if len(valid) > 0:
            features["MDVP_NHR"] = float(10.0 ** (-np.mean(valid) / 10.0))
        else:
            features["MDVP_NHR"] = 1.0
    else:
        features["MDVP_NHR"] = 1.0

    # G76: MDVP VTI (high-frequency noise / total energy ratio)
    S = intermediates["stft_power"]
    freqs = np.linspace(0, sr / 2, S.shape[0])
    hf_mask = freqs > 4000
    total_energy = np.sum(S) + 1e-12
    hf_energy = np.sum(S[hf_mask, :])
    features["MDVP_VTI"] = float(hf_energy / total_energy)

    # G77: MDVP SPI (low-frequency harmonic energy / high-frequency)
    lf_mask = (freqs >= 70) & (freqs <= 1600)
    hf_mask2 = (freqs >= 1600) & (freqs <= 4500)
    lf_e = np.sum(S[lf_mask, :]) + 1e-12
    hf_e = np.sum(S[hf_mask2, :]) + 1e-12
    features["MDVP_SPI"] = float(lf_e / hf_e)

    # G78: MDVP DVB (degree of voice breaks)
    # Fraction of time with voice breaks (unvoiced gaps in otherwise voiced regions)
    hop = intermediates["hop_length"]
    if np.any(voiced):
        total_voiced_dur = np.sum(voiced) * hop / sr
        # Count transitions from voiced to unvoiced within speech
        transitions = np.diff(voiced.astype(int))
        n_breaks = int(np.sum(transitions == -1))
        # Simple DVB = number of breaks / total voiced duration
        features["MDVP_DVB"] = float(n_breaks / (total_voiced_dur + 1e-12))
    else:
        features["MDVP_DVB"] = 0.0

    # Replace any NaN
    for k in features:
        if not np.isfinite(features[k]):
            features[k] = 0.0

    return features
