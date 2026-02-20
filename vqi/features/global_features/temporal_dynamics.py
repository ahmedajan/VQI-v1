"""G43-G48: Temporal dynamics features.

G43: PauseDurationMean
G44: PauseRate (pauses per second)
G45: LongestPause
G46: SpeechContinuity (mean speech segment length)
G47: OnsetStrengthMean
G48: OnsetStrengthStd
"""

import numpy as np
import librosa


def compute_temporal_dynamics_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    hop_length = intermediates["hop_length"]
    frame_dur = hop_length / sr

    # Find pause segments (contiguous non-speech)
    pauses = []
    speech_segs = []
    in_pause = False
    pause_len = 0
    speech_len = 0

    for v in vad_mask:
        if not v:
            if not in_pause:
                if speech_len > 0:
                    speech_segs.append(speech_len)
                speech_len = 0
                in_pause = True
            pause_len += 1
        else:
            if in_pause:
                pauses.append(pause_len)
                pause_len = 0
                in_pause = False
            speech_len += 1
    if in_pause:
        pauses.append(pause_len)
    if speech_len > 0:
        speech_segs.append(speech_len)

    pause_durations = [p * frame_dur for p in pauses]

    # G43: Mean pause duration
    features["PauseDurationMean"] = float(np.mean(pause_durations)) if pause_durations else 0.0

    # G44: Pause rate
    total_dur = len(vad_mask) * frame_dur
    features["PauseRate"] = float(len(pauses) / total_dur) if total_dur > 0 else 0.0

    # G45: Longest pause
    features["LongestPause"] = float(max(pause_durations)) if pause_durations else 0.0

    # G46: Speech continuity (mean speech segment length in seconds)
    speech_dur_segs = [s * frame_dur for s in speech_segs]
    features["SpeechContinuity"] = float(np.mean(speech_dur_segs)) if speech_dur_segs else 0.0

    # G47-G48: Onset strength
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=sr, hop_length=hop_length,
    )
    features["OnsetStrengthMean"] = float(np.mean(onset_env))
    features["OnsetStrengthStd"] = float(np.std(onset_env))

    return features
