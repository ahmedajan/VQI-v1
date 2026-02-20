"""G9-G10, G32-G36: Jitter/Shimmer features via Praat.

G9:  Jitter (local, %)
G10: Shimmer (local, dB)
G32: JitterPPQ5
G33: JitterRAP
G34: ShimmerAPQ3
G35: ShimmerAPQ5
G36: ShimmerAPQ11
"""

import numpy as np
import parselmouth
from parselmouth.praat import call


def compute_jitter_shimmer_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    pp = intermediates.get("praat_point_process")
    snd = intermediates.get("praat_sound")

    if pp is None or snd is None:
        return _defaults()

    try:
        features["Jitter"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        features["Shimmer"] = float(call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        features["JitterPPQ5"] = float(call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
        features["JitterRAP"] = float(call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        features["ShimmerAPQ3"] = float(call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        features["ShimmerAPQ5"] = float(call([snd, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        features["ShimmerAPQ11"] = float(call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        return _defaults()

    # Replace NaN with 0
    for k in features:
        if not np.isfinite(features[k]):
            features[k] = 0.0

    return features


def _defaults():
    return {
        "Jitter": 0.0, "Shimmer": 0.0,
        "JitterPPQ5": 0.0, "JitterRAP": 0.0,
        "ShimmerAPQ3": 0.0, "ShimmerAPQ5": 0.0, "ShimmerAPQ11": 0.0,
    }
