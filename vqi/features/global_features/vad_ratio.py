"""G2: Global VAD Ratio (speech frames / total frames)."""

import numpy as np


def compute_vad_ratio_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    n_total = len(vad_mask)
    n_speech = int(np.sum(vad_mask))
    ratio = n_speech / n_total if n_total > 0 else 0.0
    return {"GlobalVADRatio": float(ratio)}
