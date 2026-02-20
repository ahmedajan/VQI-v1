"""G3: Global Energy (mean RMS in dB, speech frames only)."""

import numpy as np


def compute_energy_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    frame_energy = intermediates["frame_energy"]
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)
    speech_energy = frame_energy[vad]
    if len(speech_energy) == 0:
        return {"GlobalEnergy": -60.0}
    mean_rms = np.mean(speech_energy)
    energy_db = 20.0 * np.log10(mean_rms + 1e-12)
    return {"GlobalEnergy": float(energy_db)}


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
