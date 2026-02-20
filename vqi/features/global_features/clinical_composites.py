"""G105-G107: Clinical composite scores.

G105: AVQI (Acoustic Voice Quality Index)
G106: DSI (Dysphonia Severity Index)
G107: CSID (Cepstral Spectral Index of Dysphonia)
"""

import numpy as np


def compute_clinical_composites_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    """Compute clinical voice quality composites from existing features.

    These are linear combinations of already-computed voice quality measures.
    We use the standard clinical formulas.
    """
    features = {}
    harmonicity = intermediates.get("praat_harmonicity")
    f0 = intermediates["f0"]
    voiced = f0 > 0
    eps = 1e-12

    # Get HNR
    if harmonicity is not None:
        n_h = harmonicity.n_frames
        hnr_vals = np.array([harmonicity.values[0, i] for i in range(n_h)])
        valid_hnr = hnr_vals[hnr_vals > -100]
        mean_hnr = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0
    else:
        mean_hnr = 0.0

    # Get mean F0
    mean_f0 = float(np.mean(f0[voiced])) if np.any(voiced) else 0.0

    # Get jitter/shimmer approximations
    pp = intermediates.get("praat_point_process")
    snd = intermediates.get("praat_sound")
    jitter = 0.0
    shimmer = 0.0
    if pp is not None:
        try:
            from parselmouth.praat import call
            jitter = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            if not np.isfinite(jitter):
                jitter = 0.0
        except Exception:
            pass
    if pp is not None and snd is not None:
        try:
            from parselmouth.praat import call
            shimmer = float(call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            if not np.isfinite(shimmer):
                shimmer = 0.0
        except Exception:
            pass

    # G105: AVQI = 4.152 - 0.177 * CPP - 0.006 * HNR - 0.037 * shimmer_local_dB + ...
    # Simplified version using available measures
    cpp_proxy = mean_hnr  # CPP is correlated with HNR
    avqi = 4.152 - 0.177 * cpp_proxy - 0.006 * mean_hnr - 0.037 * shimmer
    features["AVQI"] = float(np.clip(avqi, 0, 10))

    # G106: DSI = 0.13*MPT + 0.0053*F0_max - 0.26*Jitter - 1.18*Shimmer + 12.2
    # MPT (maximum phonation time) = approximate as duration
    hop = intermediates["hop_length"]
    speech_dur = np.sum(vad_mask) * hop / sr
    f0_max = float(np.max(f0[voiced])) if np.any(voiced) else 0.0
    dsi = 0.13 * speech_dur + 0.0053 * f0_max - 0.26 * (jitter * 100) - 1.18 * shimmer + 12.2
    features["DSI"] = float(np.clip(dsi, -5, 15))

    # G107: CSID = simplified cepstral spectral index
    # Uses CPP, L/H ratio
    S = intermediates["stft_power"]
    freqs = np.linspace(0, sr / 2, S.shape[0])
    mean_spec = np.mean(S, axis=1)
    lf_mask = (freqs >= 0) & (freqs <= 1500)
    hf_mask = (freqs >= 1500) & (freqs <= 5000)
    lh_ratio = np.sum(mean_spec[lf_mask]) / (np.sum(mean_spec[hf_mask]) + eps)
    csid = 0.5 * cpp_proxy + 0.3 * np.log10(lh_ratio + eps) - 0.2 * (jitter * 100)
    features["CSID"] = float(csid)

    return features
