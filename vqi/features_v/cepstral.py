"""V1-V65: Cepstral identity features for VQI-V (Voice Distinctiveness).

65 features capturing vocal tract shape via cepstral representations:
  V1-V13:   MFCC means (C1-C13)
  V14-V26:  MFCC temporal stds (C1-C13)
  V27-V39:  Delta-MFCC means (C1-C13)
  V40-V52:  LPCC means (13 coefficients)
  V53-V65:  LFCC means (13 coefficients)
"""

import numpy as np
import librosa


def compute_cepstral_features(waveform, sr, vad_mask, intermediates):
    features = {}
    mfccs = intermediates["mfccs"]  # (14, n_frames)
    delta_mfcc = intermediates["delta_mfcc"]
    n_frames = mfccs.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    speech_mfccs = mfccs[:, vad] if np.any(vad) else mfccs

    # V1-V13: MFCC means (C1-C13, excluding C0)
    for i in range(13):
        features[f"V_MFCC_Mean_{i+1}"] = float(np.mean(speech_mfccs[i + 1, :]))

    # V14-V26: MFCC temporal stds
    for i in range(13):
        features[f"V_MFCC_Std_{i+1}"] = float(np.std(speech_mfccs[i + 1, :]))

    # V27-V39: Delta-MFCC means
    speech_delta = delta_mfcc[:, vad] if np.any(vad) else delta_mfcc
    for i in range(13):
        features[f"V_DeltaMFCC_Mean_{i+1}"] = float(np.mean(speech_delta[i + 1, :]))

    # V40-V52: LPCC means (Linear Prediction Cepstral Coefficients)
    lpc_list = intermediates["lpc_per_frame"]
    lpcc_all = []
    n_f = intermediates["frames"].shape[1]
    vad_f = _align_mask(vad_mask, n_f)
    for i in range(n_f):
        if not vad_f[i]:
            continue
        lpc_coeffs = lpc_list[i]
        if np.all(lpc_coeffs == 0):
            continue
        # Convert LPC to LPCC
        lpcc = _lpc_to_lpcc(lpc_coeffs, 13)
        lpcc_all.append(lpcc)

    if lpcc_all:
        lpcc_arr = np.array(lpcc_all)
        for i in range(13):
            features[f"V_LPCC_Mean_{i+1}"] = float(np.mean(lpcc_arr[:, i]))
    else:
        for i in range(13):
            features[f"V_LPCC_Mean_{i+1}"] = 0.0

    # V53-V65: LFCC means (Linear Frequency Cepstral Coefficients)
    # LFCC: like MFCC but with linear filterbank instead of mel
    n_fft = intermediates["n_fft"]
    hop_length = intermediates["hop_length"]
    S = intermediates["stft_power"]
    # Linear filterbank
    n_filters = 40
    n_bins = S.shape[0]
    filterbank = _linear_filterbank(n_filters, n_bins, sr)
    mel_S = filterbank @ S  # (n_filters, n_frames)
    mel_S = np.log(mel_S + 1e-12)
    from scipy.fft import dct
    lfcc = dct(mel_S, type=2, axis=0, norm="ortho")[:14, :]  # (14, n_frames)
    speech_lfcc = lfcc[:, vad] if np.any(vad) else lfcc
    for i in range(13):
        features[f"V_LFCC_Mean_{i+1}"] = float(np.mean(speech_lfcc[i + 1, :]))

    return features


def _lpc_to_lpcc(lpc_coeffs, n_lpcc):
    """Convert LPC coefficients to LPCC."""
    p = len(lpc_coeffs) - 1
    lpcc = np.zeros(n_lpcc)
    a = lpc_coeffs  # a[0]=1, a[1..p] = LPC coefficients

    for n in range(n_lpcc):
        m = n + 1
        if m <= p:
            lpcc[n] = -a[m]
            for k in range(1, m):
                lpcc[n] -= (k / m) * lpcc[k - 1] * a[m - k]
        else:
            for k in range(1, p + 1):
                lpcc[n] -= (k / m) * lpcc[n - k] * a[k] if n - k >= 0 else 0
    return lpcc


def _linear_filterbank(n_filters, n_bins, sr):
    """Create a linearly-spaced filterbank."""
    freqs = np.linspace(0, sr / 2, n_bins)
    center_freqs = np.linspace(0, sr / 2, n_filters + 2)
    filterbank = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        lo, center, hi = center_freqs[i], center_freqs[i + 1], center_freqs[i + 2]
        for j, f in enumerate(freqs):
            if lo <= f < center:
                filterbank[i, j] = (f - lo) / (center - lo + 1e-12)
            elif center <= f < hi:
                filterbank[i, j] = (hi - f) / (hi - center + 1e-12)
    return filterbank


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
