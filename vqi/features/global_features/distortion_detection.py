"""G49-G53: Distortion detection features.

G49: ClickRate (transient spike rate)
G50: DropoutRate (energy dropouts)
G51: SaturationRatio (near-clipping frames)
G52: MusicalNoiseLevel (tonal artifacts)
G53: QuantizationNoise (bit-depth-related noise)
"""

import numpy as np


def compute_distortion_detection_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    hop_length = intermediates["hop_length"]
    frame_energy = intermediates["frame_energy"]
    n_frames = len(frame_energy)
    eps = 1e-12

    # G49: Click rate - detect short transient spikes
    # Clicks = frames where energy jumps > 3x neighbors
    click_count = 0
    for i in range(1, n_frames - 1):
        if frame_energy[i] > 3 * frame_energy[i - 1] and frame_energy[i] > 3 * frame_energy[i + 1]:
            click_count += 1
    duration = len(waveform) / sr
    features["ClickRate"] = float(click_count / duration) if duration > 0 else 0.0

    # G50: Dropout rate - frames with sudden energy drop to near silence
    median_energy = np.median(frame_energy[frame_energy > eps]) if np.any(frame_energy > eps) else eps
    dropout_threshold = median_energy * 0.01
    vad_aligned = _align_mask(vad_mask, n_frames)
    dropout_count = int(np.sum((frame_energy < dropout_threshold) & vad_aligned))
    features["DropoutRate"] = float(dropout_count / duration) if duration > 0 else 0.0

    # G51: Saturation ratio (samples near +-1.0)
    wav = raw_waveform if raw_waveform is not None else waveform
    near_clip = np.sum(np.abs(wav) > 0.95)
    features["SaturationRatio"] = float(near_clip / len(wav)) if len(wav) > 0 else 0.0

    # G52: Musical noise level (spectral peaks that persist across frames)
    S = intermediates["stft_power"]
    # Compute temporal variance of each frequency bin, normalized
    if S.shape[1] > 1:
        temporal_var = np.var(S, axis=1)
        temporal_mean = np.mean(S, axis=1) + eps
        # Musical noise = low-variance, high-energy bins (tonal artifacts)
        ratio = temporal_var / temporal_mean
        # Fraction of bins with very low variation (< median/10)
        med_ratio = np.median(ratio)
        musical = np.sum(ratio < med_ratio * 0.1) / len(ratio) if med_ratio > 0 else 0.0
    else:
        musical = 0.0
    features["MusicalNoiseLevel"] = float(musical)

    # G53: Quantization noise (estimate from LSB patterns)
    if len(waveform) > 0:
        # Residual after removing smooth signal -> proxy for quantization
        diff = np.diff(waveform)
        unique_diffs = len(np.unique(np.round(diff, 6)))
        max_possible = min(len(diff), 65536)
        quant_noise = 1.0 - unique_diffs / max_possible if max_possible > 0 else 0.0
    else:
        quant_noise = 0.0
    features["QuantizationNoise"] = float(max(0.0, quant_noise))

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
