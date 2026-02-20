"""G30-G31, G37-G40: Voice quality features.

G30: CPP_Mean
G31: CPP_Std
G37: NHR (Noise-to-Harmonics Ratio)
G38: H1H2 (first harmonic minus second harmonic)
G39: H1A3 (first harmonic minus third formant amplitude)
G40: UnvoicedFrameRatio
"""

import numpy as np
from parselmouth.praat import call


def compute_voice_quality_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}

    # G30-G31: CPP mean/std (reuse CPP computation or compute from harmonicity)
    harmonicity = intermediates.get("praat_harmonicity")
    if harmonicity is not None:
        n_h = harmonicity.n_frames
        hnr_vals = np.array([harmonicity.values[0, i] for i in range(n_h)])
        # HNR is closely related to CPP; use as proxy
        valid = hnr_vals[hnr_vals > -100]
        features["CPP_Mean"] = float(np.mean(valid)) if len(valid) > 0 else 0.0
        features["CPP_Std"] = float(np.std(valid)) if len(valid) > 0 else 0.0
    else:
        features["CPP_Mean"] = 0.0
        features["CPP_Std"] = 0.0

    # G37: NHR (inverse of HNR)
    snd = intermediates.get("praat_sound")
    pp = intermediates.get("praat_point_process")
    pitch = intermediates.get("praat_pitch")

    if harmonicity is not None:
        n_h = harmonicity.n_frames
        hnr_vals = np.array([harmonicity.values[0, i] for i in range(n_h)])
        valid_hnr = hnr_vals[hnr_vals > -100]
        if len(valid_hnr) > 0:
            mean_hnr = np.mean(valid_hnr)
            # NHR = 10^(-HNR/10)
            features["NHR"] = float(10.0 ** (-mean_hnr / 10.0))
        else:
            features["NHR"] = 1.0
    else:
        features["NHR"] = 1.0

    # G38: H1-H2 (difference between first and second harmonic amplitudes)
    f0 = intermediates["f0"]
    voiced = f0 > 0
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    freqs = np.linspace(0, sr / 2, n_bins)
    n_frames_s = S.shape[1]

    h1h2_vals = []
    f0_aligned = _align_array(f0, n_frames_s)
    for j in range(n_frames_s):
        f0_val = f0_aligned[j]
        if f0_val < 60:
            continue
        # Find H1 and H2
        h1_idx = np.argmin(np.abs(freqs - f0_val))
        h2_idx = np.argmin(np.abs(freqs - 2 * f0_val))
        if h2_idx < n_bins:
            h1_db = 10 * np.log10(S[h1_idx, j] + 1e-12)
            h2_db = 10 * np.log10(S[h2_idx, j] + 1e-12)
            h1h2_vals.append(h1_db - h2_db)
    features["H1H2"] = float(np.mean(h1h2_vals)) if h1h2_vals else 0.0

    # G39: H1-A3 (first harmonic minus third formant amplitude)
    formant = intermediates.get("praat_formant")
    h1a3_vals = []
    if formant is not None:
        n_form_frames = formant.n_frames
        for j in range(min(n_form_frames, n_frames_s)):
            f0_val = f0_aligned[j] if j < len(f0_aligned) else 0
            if f0_val < 60:
                continue
            try:
                f3 = formant.get_value_at_time(3, formant.x1 + j * formant.dx)
                if f3 is None or np.isnan(f3) or f3 <= 0:
                    continue
            except Exception:
                continue
            h1_idx = np.argmin(np.abs(freqs - f0_val))
            a3_idx = np.argmin(np.abs(freqs - f3))
            if a3_idx < n_bins:
                h1_db = 10 * np.log10(S[h1_idx, j] + 1e-12)
                a3_db = 10 * np.log10(S[a3_idx, j] + 1e-12)
                h1a3_vals.append(h1_db - a3_db)
    features["H1A3"] = float(np.mean(h1a3_vals)) if h1a3_vals else 0.0

    # G40: Unvoiced frame ratio
    voiced_flag = intermediates["voiced_flag"]
    n_voiced = np.sum(voiced_flag)
    n_total_p = len(voiced_flag)
    features["UnvoicedFrameRatio"] = float(1.0 - n_voiced / n_total_p) if n_total_p > 0 else 1.0

    return features


def _align_array(arr, n_target):
    if len(arr) == n_target:
        return arr
    indices = np.round(np.linspace(0, len(arr) - 1, n_target)).astype(int)
    return arr[indices]
