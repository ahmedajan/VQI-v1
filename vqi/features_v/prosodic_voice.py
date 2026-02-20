"""V110-V126: Prosodic and voice source features for VQI-V.

17 features:
  V110: Articulation rate (syllables/sec of speech)
  V111: F0 mean
  V112: F0 std
  V113: F0 range
  V114: F0 slope
  V115: F0 median
  V116: OQ (Open Quotient)
  V117: CQ (Closed Quotient)
  V118: SQ (Speed Quotient)
  V119: Rd parameter
  V120: Ra parameter
  V121: Rk parameter
  V122: Modal proportion
  V123: Breathy proportion
  V124: Alpha ratio (speaker-characteristic)
  V125: Harmonic richness
  V126: Spectral tilt (voice source)
"""

import numpy as np
from scipy.signal import lfilter


def compute_prosodic_voice_features(waveform, sr, vad_mask, intermediates):
    features = {}
    f0 = intermediates["f0"]
    voiced = f0 > 0
    hop = intermediates["hop_length"]
    eps = 1e-12

    # V110: Articulation rate
    transitions = np.diff(voiced.astype(int))
    n_syllables = int(np.sum(transitions == 1))
    n_speech = int(np.sum(vad_mask))
    speech_dur = n_speech * hop / sr
    features["V_ArticulationRate"] = float(n_syllables / (speech_dur + eps))

    # V111-V115: F0 statistics
    if np.any(voiced):
        f0_v = f0[voiced]
        features["V_F0_Mean"] = float(np.mean(f0_v))
        features["V_F0_Std"] = float(np.std(f0_v))
        features["V_F0_Range"] = float(np.max(f0_v) - np.min(f0_v))
        # F0 slope: linear trend
        x = np.arange(len(f0_v))
        if len(x) > 1:
            coeffs = np.polyfit(x, f0_v, 1)
            features["V_F0_Slope"] = float(coeffs[0])
        else:
            features["V_F0_Slope"] = 0.0
        features["V_F0_Median"] = float(np.median(f0_v))
    else:
        for k in ["V_F0_Mean", "V_F0_Std", "V_F0_Range", "V_F0_Slope", "V_F0_Median"]:
            features[k] = 0.0

    # V116-V121: Voice source parameters from inverse filtering
    frames = intermediates["frames"]
    lpc_list = intermediates["lpc_per_frame"]
    n_frames = frames.shape[1]
    vad_f = _align_mask(vad_mask, n_frames)

    oq_vals, cq_vals, sq_vals = [], [], []
    mean_f0 = features.get("V_F0_Mean", 150.0)
    if mean_f0 < 60:
        mean_f0 = 150.0
    period = int(sr / mean_f0)

    for i in range(n_frames):
        if not vad_f[i]:
            continue
        lpc_coeffs = lpc_list[i]
        if np.all(lpc_coeffs == 0):
            continue
        frame = frames[:, i]
        residual = lfilter(lpc_coeffs, [1.0], frame)
        flow = np.cumsum(residual)

        # Find glottal pulse boundaries in one period
        if len(flow) < period:
            continue
        segment = flow[:period]
        peak_idx = np.argmax(np.abs(segment))
        # Simple OQ: fraction of period where flow is positive
        positive_frac = np.sum(segment > 0) / (len(segment) + eps)
        oq_vals.append(positive_frac)
        cq_vals.append(1.0 - positive_frac)
        # SQ: rising time / falling time
        if peak_idx > 0 and peak_idx < len(segment) - 1:
            sq_vals.append(peak_idx / (len(segment) - peak_idx + eps))

    features["V_OQ"] = float(np.mean(oq_vals)) if oq_vals else 0.5
    features["V_CQ"] = float(np.mean(cq_vals)) if cq_vals else 0.5
    features["V_SQ"] = float(np.clip(np.mean(sq_vals), 0, 5)) if sq_vals else 1.0

    # V119-V121: Rd, Ra, Rk (simplified Liljencrants-Fant parameters)
    # Rd ~ OQ-based; Ra ~ return phase; Rk ~ symmetry
    features["V_Rd"] = float(0.5 + features["V_OQ"])
    features["V_Ra"] = float(1.0 - features["V_CQ"])
    features["V_Rk"] = float(features["V_SQ"])

    # V122-V123: Modal vs breathy proportion
    # Modal: high HNR (>10); Breathy: low HNR (<5)
    harmonicity = intermediates.get("praat_harmonicity")
    if harmonicity is not None:
        n_h = harmonicity.n_frames
        hnr_vals = np.array([harmonicity.values[0, i] for i in range(n_h)])
        valid = hnr_vals[hnr_vals > -100]
        if len(valid) > 0:
            features["V_ModalProportion"] = float(np.mean(valid > 10))
            features["V_BreathyProportion"] = float(np.mean(valid < 5))
        else:
            features["V_ModalProportion"] = 0.5
            features["V_BreathyProportion"] = 0.5
    else:
        features["V_ModalProportion"] = 0.5
        features["V_BreathyProportion"] = 0.5

    # V124: Alpha ratio (speaker-characteristic version)
    S = intermediates["stft_power"]
    n_bins = S.shape[0]
    freqs = np.linspace(0, sr / 2, n_bins)
    band_lo = (freqs >= 50) & (freqs <= 1000)
    band_hi = (freqs >= 1000) & (freqs <= 5000)
    mean_spec = np.mean(S, axis=1)
    e_lo = np.sum(mean_spec[band_lo]) + eps
    e_hi = np.sum(mean_spec[band_hi]) + eps
    features["V_AlphaRatio"] = float(10.0 * np.log10(e_lo / e_hi))

    # V125: Harmonic richness
    if np.any(voiced):
        f0_mean = np.mean(f0[voiced])
        h1_idx = np.argmin(np.abs(freqs - f0_mean))
        h1_power = mean_spec[h1_idx] + eps
        harmonic_power = 0.0
        for h in range(2, 10):
            h_freq = f0_mean * h
            if h_freq > sr / 2:
                break
            h_idx = np.argmin(np.abs(freqs - h_freq))
            harmonic_power += mean_spec[h_idx]
        features["V_HarmonicRichness"] = float(np.clip(harmonic_power / h1_power, 0, 100))
    else:
        features["V_HarmonicRichness"] = 0.0

    # V126: Spectral tilt
    if len(freqs) > 1:
        mean_spec_db = 10.0 * np.log10(mean_spec + eps)
        coeffs = np.polyfit(freqs, mean_spec_db, 1)
        features["V_SpectralTilt"] = float(coeffs[0])
    else:
        features["V_SpectralTilt"] = 0.0

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
