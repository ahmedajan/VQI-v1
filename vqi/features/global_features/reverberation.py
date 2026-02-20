"""G6, G18-G20: Reverberation features.

G6:  GlobalReverb (modulation spectrum 4Hz energy ratio)
G18: RT60_Est (estimated RT60 from energy decay)
G19: C50_Est (clarity index estimate)
G20: ModulationDepth
"""

import numpy as np
from scipy.signal import hilbert


def compute_reverberation_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    envelope = intermediates["hilbert_envelope"]
    features = {}

    # G6: Modulation spectrum 4Hz ratio (STI-based)
    # Compute modulation spectrum of the envelope
    n = len(envelope)
    if n < sr // 2:
        features["GlobalReverb"] = 0.0
        features["RT60_Est"] = 0.0
        features["C50_Est"] = 0.0
        features["ModulationDepth"] = 0.0
        return features

    mod_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
    mod_freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Energy around 4Hz (3-5Hz band) vs total
    band_mask = (mod_freqs >= 3.0) & (mod_freqs <= 5.0)
    total_mask = mod_freqs > 0.5  # above DC
    e_4hz = np.sum(mod_fft[band_mask] ** 2) if np.any(band_mask) else 0.0
    e_total = np.sum(mod_fft[total_mask] ** 2) + 1e-12
    features["GlobalReverb"] = float(e_4hz / e_total)

    # G18: RT60 estimate from energy decay curve
    # Simple: find time for energy to decay 60dB after a speech offset
    frame_energy = intermediates["frame_energy"]
    hop = intermediates["hop_length"]
    energy_db = 20.0 * np.log10(frame_energy + 1e-12)
    peak_db = np.max(energy_db)

    # Find decay from peak
    peak_idx = np.argmax(energy_db)
    decay_region = energy_db[peak_idx:]
    target = peak_db - 60.0
    below = np.where(decay_region <= target)[0]
    if len(below) > 0:
        decay_frames = below[0]
        rt60 = decay_frames * hop / sr
    else:
        rt60 = len(decay_region) * hop / sr
    features["RT60_Est"] = float(min(rt60, 5.0))

    # G19: C50 estimate (energy in first 50ms vs rest after direct sound)
    samples_50ms = int(0.05 * sr)
    e_early = np.sum(waveform[:samples_50ms] ** 2)
    e_late = np.sum(waveform[samples_50ms:] ** 2) + 1e-12
    c50 = 10.0 * np.log10(e_early / e_late + 1e-12)
    features["C50_Est"] = float(np.clip(c50, -30, 30))

    # G20: Modulation depth (peak-to-trough ratio of envelope)
    if np.mean(envelope) > 1e-12:
        mod_depth = (np.max(envelope) - np.min(envelope)) / (np.mean(envelope) + 1e-12)
    else:
        mod_depth = 0.0
    features["ModulationDepth"] = float(min(mod_depth, 10.0))

    return features
