"""G63: Signal Integrity.

G63: InterruptionCount (sudden silence gaps within speech)
"""

import numpy as np


def compute_signal_integrity_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    frame_energy = intermediates["frame_energy"]
    hop_length = intermediates["hop_length"]
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12

    # Count short (<50ms) silence gaps surrounded by speech
    min_gap = max(1, int(0.05 * sr / hop_length))  # 50ms in frames
    max_gap = int(0.5 * sr / hop_length)  # 500ms max to be an interruption

    interruptions = 0
    gap_len = 0
    in_gap = False

    for i in range(n_frames):
        if vad[i]:
            if in_gap and min_gap <= gap_len <= max_gap:
                interruptions += 1
            gap_len = 0
            in_gap = False
        else:
            if not in_gap and i > 0 and vad[i - 1]:
                in_gap = True
                gap_len = 0
            if in_gap:
                gap_len += 1

    return {"InterruptionCount": float(interruptions)}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
