"""G41-G42: Energy dynamics features.

G41: EnergyRange (max - min energy in dB)
G42: EnergyContourVariance (variance of energy contour)
"""

import numpy as np


def compute_energy_dynamics_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    frame_energy = intermediates["frame_energy"]
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12

    speech_energy = frame_energy[vad]
    if len(speech_energy) < 2:
        return {"EnergyRange": 0.0, "EnergyContourVariance": 0.0}

    energy_db = 20.0 * np.log10(speech_energy + eps)
    features = {
        "EnergyRange": float(np.max(energy_db) - np.min(energy_db)),
        "EnergyContourVariance": float(np.var(energy_db)),
    }
    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
