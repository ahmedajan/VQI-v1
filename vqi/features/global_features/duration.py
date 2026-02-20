"""G1: Global Duration (speech duration in seconds, VAD-based)."""

import numpy as np


def compute_duration_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    hop_length = intermediates["hop_length"]
    n_speech = int(np.sum(vad_mask))
    duration = n_speech * hop_length / sr
    return {"GlobalDuration": float(duration)}
