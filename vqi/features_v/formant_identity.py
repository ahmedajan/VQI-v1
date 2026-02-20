"""V99-V109: Formant identity features for VQI-V.

11 features capturing speaker anatomy:
  V99:  F2/F1 ratio (vowel space indicator)
  V100: F3/F2 ratio
  V101: F3/F1 ratio
  V102: VTL estimate (vocal tract length from formants)
  V103: F1 dynamics (std over time)
  V104: F2 dynamics
  V105: F3 dynamics
  V106: Formant centralization (distance from neutral vowel)
  V107: F1-F2 correlation
  V108: F2-F3 correlation
  V109: F1-F3 correlation
"""

import numpy as np


def compute_formant_identity_features(waveform, sr, vad_mask, intermediates):
    features = {}
    formant = intermediates.get("praat_formant")

    if formant is None:
        return _defaults()

    n_form_frames = formant.n_frames
    f1_vals, f2_vals, f3_vals = [], [], []

    for i in range(n_form_frames):
        t = formant.x1 + i * formant.dx
        try:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
        except Exception:
            continue
        if (f1 is not None and not np.isnan(f1) and f1 > 0 and
                f2 is not None and not np.isnan(f2) and f2 > 0 and
                f3 is not None and not np.isnan(f3) and f3 > 0):
            f1_vals.append(f1)
            f2_vals.append(f2)
            f3_vals.append(f3)

    if len(f1_vals) < 5:
        return _defaults()

    f1 = np.array(f1_vals)
    f2 = np.array(f2_vals)
    f3 = np.array(f3_vals)
    eps = 1e-12

    # V99-V101: Formant ratios
    features["V_F2F1_Ratio"] = float(np.mean(f2) / (np.mean(f1) + eps))
    features["V_F3F2_Ratio"] = float(np.mean(f3) / (np.mean(f2) + eps))
    features["V_F3F1_Ratio"] = float(np.mean(f3) / (np.mean(f1) + eps))

    # V102: VTL estimate = c / (4 * mean_formant_spacing)
    # Using formant dispersion method
    formant_spacing = (np.mean(f3) - np.mean(f1)) / 2.0
    speed_of_sound = 35000  # cm/s
    vtl = speed_of_sound / (4 * formant_spacing + eps)
    features["V_VTL"] = float(np.clip(vtl, 5, 30))

    # V103-V105: Formant dynamics (temporal std)
    features["V_F1_Dynamics"] = float(np.std(f1))
    features["V_F2_Dynamics"] = float(np.std(f2))
    features["V_F3_Dynamics"] = float(np.std(f3))

    # V106: Formant centralization (distance from neutral vowel ~500/1500/2500)
    neutral = np.array([500, 1500, 2500])
    actual = np.array([np.mean(f1), np.mean(f2), np.mean(f3)])
    features["V_FormantCentralization"] = float(np.sqrt(np.sum((actual - neutral) ** 2)))

    # V107-V109: Formant correlations
    features["V_F1F2_Corr"] = float(np.corrcoef(f1, f2)[0, 1]) if len(f1) > 2 else 0.0
    features["V_F2F3_Corr"] = float(np.corrcoef(f2, f3)[0, 1]) if len(f2) > 2 else 0.0
    features["V_F1F3_Corr"] = float(np.corrcoef(f1, f3)[0, 1]) if len(f1) > 2 else 0.0

    # Fix NaN correlations
    for k in ["V_F1F2_Corr", "V_F2F3_Corr", "V_F1F3_Corr"]:
        if not np.isfinite(features[k]):
            features[k] = 0.0

    return features


def _defaults():
    return {
        "V_F2F1_Ratio": 3.0, "V_F3F2_Ratio": 1.67, "V_F3F1_Ratio": 5.0,
        "V_VTL": 17.0,
        "V_F1_Dynamics": 0.0, "V_F2_Dynamics": 0.0, "V_F3_Dynamics": 0.0,
        "V_FormantCentralization": 0.0,
        "V_F1F2_Corr": 0.0, "V_F2F3_Corr": 0.0, "V_F1F3_Corr": 0.0,
    }
