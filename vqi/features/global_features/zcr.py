"""G8: Global Zero-Crossing Rate (mean across speech frames)."""

import numpy as np
import librosa


def compute_zcr_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    zcr = librosa.feature.zero_crossing_rate(
        waveform, frame_length=intermediates["n_fft"],
        hop_length=intermediates["hop_length"],
    )[0]
    n_frames = len(zcr)
    vad = _align_mask(vad_mask, n_frames)
    speech_zcr = zcr[vad] if np.any(vad) else zcr
    return {"GlobalZCR": float(np.mean(speech_zcr))}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
