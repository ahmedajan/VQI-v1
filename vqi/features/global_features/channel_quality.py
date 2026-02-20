"""G54-G56: Channel quality features.

G54: DCOffset (residual DC component)
G55: PowerLineHum (50/60 Hz hum detection)
G56: AGC_Activity (automatic gain control activity)
"""

import numpy as np


def compute_channel_quality_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    eps = 1e-12

    # G54: DC offset (mean of waveform, should be ~0 after normalization)
    wav = raw_waveform if raw_waveform is not None else waveform
    features["DCOffset"] = float(np.abs(np.mean(wav)))

    # G55: Power line hum (energy at 50Hz and 60Hz relative to neighbors)
    S = intermediates["stft_power"]
    freqs = np.linspace(0, sr / 2, S.shape[0])
    mean_spec = np.mean(S, axis=1)

    hum_energy = 0.0
    for hum_freq in [50, 60, 100, 120]:  # fundamentals + 2nd harmonics
        idx = np.argmin(np.abs(freqs - hum_freq))
        # Compare to neighbors
        lo = max(0, idx - 3)
        hi = min(len(mean_spec), idx + 4)
        neighbor_mean = np.mean(np.concatenate([mean_spec[lo:idx], mean_spec[idx + 1:hi]])) + eps
        hum_energy += mean_spec[idx] / neighbor_mean

    features["PowerLineHum"] = float(hum_energy / 4.0)  # average ratio

    # G56: AGC activity (variance of slowly-varying energy envelope)
    frame_energy = intermediates["frame_energy"]
    if len(frame_energy) > 10:
        # Smooth energy with large window (500ms ~ 50 frames at 100fps)
        kernel_size = min(50, len(frame_energy) // 2)
        if kernel_size > 1:
            smoothed = np.convolve(frame_energy, np.ones(kernel_size) / kernel_size, mode="valid")
            agc = np.std(smoothed) / (np.mean(smoothed) + eps)
        else:
            agc = 0.0
    else:
        agc = 0.0
    features["AGC_Activity"] = float(agc)

    return features
