"""G11: Speech Rate (syllable-like rate from pitch voicing transitions)."""

import numpy as np


def compute_speech_rate_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    voiced_flag = intermediates["voiced_flag"]
    hop_length = intermediates["hop_length"]

    # Count voiced-to-unvoiced transitions (approximates syllable count)
    transitions = np.diff(voiced_flag.astype(int))
    n_syllables = int(np.sum(transitions == 1))  # onsets

    # Duration of speech
    n_speech = int(np.sum(vad_mask))
    duration = n_speech * hop_length / sr
    if duration < 0.1:
        return {"SpeechRate": 0.0}

    rate = n_syllables / duration  # syllables per second
    return {"SpeechRate": float(rate)}
