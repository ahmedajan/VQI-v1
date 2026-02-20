"""G66-G71: Glottal source features.

G66: NAQ (Normalized Amplitude Quotient)
G67: QOQ (Quasi-Open Quotient)
G68: HRF (Harmonic Richness Factor)
G69: PSP (Parabolic Spectral Parameter)
G70: GCI_Rate (Glottal Closure Instant rate)
G71: GOI_Regularity (regularity of glottal opening instants)
"""

import numpy as np
from scipy.signal import lfilter


def compute_glottal_source_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    f0 = intermediates["f0"]
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    freqs = np.linspace(0, sr / 2, n_bins)
    eps = 1e-12

    voiced = f0 > 0
    mean_f0 = np.mean(f0[voiced]) if np.any(voiced) else 150.0

    # G66: NAQ (from LPC inverse filtering)
    # Simplified: ratio of peak glottal flow to peak derivative
    lpc_list = intermediates["lpc_per_frame"]
    frames = intermediates["frames"]
    n_frames = frames.shape[1]
    vad_a = _align_mask(vad_mask, n_frames)

    naq_vals = []
    for i in range(n_frames):
        if not vad_a[i]:
            continue
        frame = frames[:, i]
        lpc_coeffs = lpc_list[i]
        if np.all(lpc_coeffs == 0):
            continue
        residual = lfilter(lpc_coeffs, [1.0], frame)
        # Integrate residual to get glottal flow
        flow = np.cumsum(residual)
        peak_flow = np.max(np.abs(flow)) + eps
        peak_deriv = np.max(np.abs(residual)) + eps
        naq_vals.append(peak_flow / (peak_deriv * (sr / mean_f0 + eps)))

    features["NAQ"] = float(np.mean(naq_vals)) if naq_vals else 0.0

    # G67: QOQ (quasi-open quotient from glottal flow)
    # Simplified: fraction of period where flow is above 50% of peak
    qoq_vals = []
    period_samples = int(sr / mean_f0) if mean_f0 > 0 else 160
    for i in range(n_frames):
        if not vad_a[i]:
            continue
        frame = frames[:, i]
        lpc_coeffs = lpc_list[i]
        if np.all(lpc_coeffs == 0):
            continue
        residual = lfilter(lpc_coeffs, [1.0], frame)
        flow = np.cumsum(residual)
        peak = np.max(np.abs(flow)) + eps
        above_half = np.sum(np.abs(flow) > 0.5 * peak)
        qoq_vals.append(above_half / (period_samples + eps))

    features["QOQ"] = float(np.clip(np.mean(qoq_vals), 0, 1)) if qoq_vals else 0.5

    # G68: HRF (harmonic richness factor = sum of harmonics / fundamental)
    n_frames_s = S.shape[1]
    f0_aligned = _align_array(f0, n_frames_s)
    hrf_vals = []
    for j in range(n_frames_s):
        f0_val = f0_aligned[j]
        if f0_val < 60:
            continue
        h1_idx = np.argmin(np.abs(freqs - f0_val))
        h1_power = S[h1_idx, j] + eps
        harmonic_power = 0.0
        for h in range(2, 10):
            h_freq = f0_val * h
            if h_freq > sr / 2:
                break
            h_idx = np.argmin(np.abs(freqs - h_freq))
            harmonic_power += S[h_idx, j]
        hrf_vals.append(np.clip(harmonic_power / h1_power, 0, 100))
    features["HRF"] = float(np.mean(hrf_vals)) if hrf_vals else 0.0

    # G69: PSP (parabolic spectral parameter)
    # Ratio of even to odd harmonics
    psp_vals = []
    for j in range(n_frames_s):
        f0_val = f0_aligned[j]
        if f0_val < 60:
            continue
        even_power = 0.0
        odd_power = eps
        for h in range(1, 10):
            h_freq = f0_val * h
            if h_freq > sr / 2:
                break
            h_idx = np.argmin(np.abs(freqs - h_freq))
            if h % 2 == 0:
                even_power += S[h_idx, j]
            else:
                odd_power += S[h_idx, j]
        psp_vals.append(np.clip(even_power / odd_power, 0, 100))
    features["PSP"] = float(np.mean(psp_vals)) if psp_vals else 0.0

    # G70: GCI rate (approximate from pitch)
    # GCI occurs once per pitch period
    n_voiced_frames = np.sum(voiced)
    total_voiced_dur = n_voiced_frames * intermediates["hop_length"] / sr
    features["GCI_Rate"] = float(mean_f0) if total_voiced_dur > 0.1 else 0.0

    # G71: GOI regularity (std of inter-GCI intervals, normalized)
    if np.any(voiced) and len(f0[voiced]) > 2:
        periods = 1.0 / (f0[voiced] + eps)
        features["GOI_Regularity"] = float(np.std(periods) / (np.mean(periods) + eps))
    else:
        features["GOI_Regularity"] = 0.0

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)


def _align_array(arr, n_target):
    if len(arr) == n_target:
        return arr
    indices = np.round(np.linspace(0, len(arr) - 1, n_target)).astype(int)
    return arr[indices]
