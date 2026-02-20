"""G93-G94: Prosody/rhythm features.

G93: MeanF0 (mean fundamental frequency)
G94: F0_StdDev (standard deviation of F0)
"""

import numpy as np


def compute_prosody_rhythm_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    f0 = intermediates["f0"]
    voiced = f0 > 0

    if np.any(voiced):
        f0_voiced = f0[voiced]
        return {
            "MeanF0": float(np.mean(f0_voiced)),
            "F0_StdDev": float(np.std(f0_voiced)),
        }
    return {"MeanF0": 0.0, "F0_StdDev": 0.0}
