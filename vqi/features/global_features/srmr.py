"""G96: SRMR (Speech-to-Reverberation Modulation energy Ratio)."""

import numpy as np


def compute_srmr_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    """Simplified SRMR: ratio of modulation energy in speech band vs reverb band."""
    envelope = intermediates["hilbert_envelope"]
    n = len(envelope)
    eps = 1e-12

    if n < sr // 2:
        return {"SRMR": 1.0}

    # Modulation spectrum
    mod_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope))) ** 2
    mod_freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Speech modulation: 2-8 Hz; Reverberation modulation: 0.5-2 Hz
    speech_band = (mod_freqs >= 2.0) & (mod_freqs <= 8.0)
    reverb_band = (mod_freqs >= 0.5) & (mod_freqs < 2.0)

    e_speech = np.sum(mod_fft[speech_band]) + eps if np.any(speech_band) else eps
    e_reverb = np.sum(mod_fft[reverb_band]) + eps if np.any(reverb_band) else eps

    srmr = e_speech / e_reverb
    return {"SRMR": float(srmr)}
