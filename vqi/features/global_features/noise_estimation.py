"""G13-G17: Noise estimation features.

G13: SegmentalSNR
G14: WADA_SNR
G15: NoiseFloorLevel
G16: NoiseBandwidth
G17: NoiseStationarity
"""

import numpy as np


def compute_noise_estimation_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    frame_energy = intermediates["frame_energy"]
    n_frames = len(frame_energy)
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12

    speech_energy = frame_energy[vad]
    noise_mask = ~vad
    noise_energy = frame_energy[noise_mask] if np.any(noise_mask) else np.array([eps])

    # G13: Segmental SNR (mean of per-frame SNR)
    noise_floor = np.mean(noise_energy ** 2) + eps
    if len(speech_energy) > 0:
        frame_snr = 10.0 * np.log10(speech_energy ** 2 / noise_floor + eps)
        features["SegmentalSNR"] = float(np.mean(frame_snr))
    else:
        features["SegmentalSNR"] = 0.0

    # G14: WADA SNR (Waveform Amplitude Distribution Analysis)
    # Simplified: ratio of speech amplitude std to noise amplitude std
    hop = intermediates["hop_length"]
    n_fft = intermediates["n_fft"]
    speech_samples = _get_speech_samples(waveform, vad_mask, n_fft, hop)
    noise_samples = _get_noise_samples(waveform, vad_mask, n_fft, hop)

    if len(noise_samples) > 0 and np.std(noise_samples) > eps:
        wada = 20.0 * np.log10(np.std(speech_samples) / (np.std(noise_samples) + eps) + eps)
    else:
        wada = 40.0
    features["WADA_SNR"] = float(np.clip(wada, -10, 60))

    # G15: Noise floor level (dB)
    noise_floor_db = 20.0 * np.log10(np.mean(noise_energy) + eps)
    features["NoiseFloorLevel"] = float(noise_floor_db)

    # G16: Noise bandwidth
    S = intermediates["stft_power"]
    stft_noise_mask = _align_mask(vad_mask, S.shape[1])
    stft_noise_mask = ~stft_noise_mask
    if np.any(stft_noise_mask):
        noise_spec = np.mean(S[:, stft_noise_mask], axis=1)
        freqs = np.linspace(0, sr / 2, len(noise_spec))
        cumsum = np.cumsum(noise_spec)
        total = cumsum[-1] + eps
        idx_95 = np.searchsorted(cumsum, 0.95 * total)
        idx_95 = min(idx_95, len(freqs) - 1)
        features["NoiseBandwidth"] = float(freqs[idx_95])
    else:
        features["NoiseBandwidth"] = float(sr / 2)

    # G17: Noise stationarity (std of noise frame energies / mean)
    if len(noise_energy) > 1:
        stationarity = np.std(noise_energy) / (np.mean(noise_energy) + eps)
    else:
        stationarity = 0.0
    features["NoiseStationarity"] = float(stationarity)

    return features


def _get_speech_samples(waveform, vad_mask, n_fft, hop):
    samples = []
    for i, is_speech in enumerate(vad_mask):
        if is_speech:
            start = i * hop
            end = min(start + n_fft, len(waveform))
            samples.append(waveform[start:end])
    return np.concatenate(samples) if samples else np.array([0.0])


def _get_noise_samples(waveform, vad_mask, n_fft, hop):
    samples = []
    for i, is_speech in enumerate(vad_mask):
        if not is_speech:
            start = i * hop
            end = min(start + n_fft, len(waveform))
            samples.append(waveform[start:end])
    return np.concatenate(samples) if samples else np.array([0.0])


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
