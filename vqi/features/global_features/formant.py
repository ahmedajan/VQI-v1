"""G83-G89: Formant features.

G83: F1_Mean
G84: F2_Mean
G85: F3_Mean
G86: FormantDispersion (mean difference between consecutive formants)
G87: F1_BW (F1 bandwidth)
G88: F2_BW
G89: F3_BW
"""

import numpy as np


def compute_formant_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    formant = intermediates.get("praat_formant")

    if formant is None:
        return {
            "F1_Mean": 500.0, "F2_Mean": 1500.0, "F3_Mean": 2500.0,
            "FormantDispersion": 1000.0,
            "F1_BW": 100.0, "F2_BW": 100.0, "F3_BW": 100.0,
        }

    n_form_frames = formant.n_frames
    f1_vals, f2_vals, f3_vals = [], [], []
    f1_bw, f2_bw, f3_bw = [], [], []

    for i in range(n_form_frames):
        t = formant.x1 + i * formant.dx
        try:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            b1 = formant.get_bandwidth_at_time(1, t)
            b2 = formant.get_bandwidth_at_time(2, t)
            b3 = formant.get_bandwidth_at_time(3, t)
        except Exception:
            continue

        if f1 is not None and not np.isnan(f1) and f1 > 0:
            f1_vals.append(f1)
            if b1 is not None and not np.isnan(b1):
                f1_bw.append(b1)
        if f2 is not None and not np.isnan(f2) and f2 > 0:
            f2_vals.append(f2)
            if b2 is not None and not np.isnan(b2):
                f2_bw.append(b2)
        if f3 is not None and not np.isnan(f3) and f3 > 0:
            f3_vals.append(f3)
            if b3 is not None and not np.isnan(b3):
                f3_bw.append(b3)

    features["F1_Mean"] = float(np.mean(f1_vals)) if f1_vals else 500.0
    features["F2_Mean"] = float(np.mean(f2_vals)) if f2_vals else 1500.0
    features["F3_Mean"] = float(np.mean(f3_vals)) if f3_vals else 2500.0

    # Formant dispersion
    features["FormantDispersion"] = float(
        (features["F3_Mean"] - features["F1_Mean"]) / 2.0
    )

    features["F1_BW"] = float(np.mean(f1_bw)) if f1_bw else 100.0
    features["F2_BW"] = float(np.mean(f2_bw)) if f2_bw else 100.0
    features["F3_BW"] = float(np.mean(f3_bw)) if f3_bw else 100.0

    return features
