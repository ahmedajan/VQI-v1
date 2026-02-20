"""V127-V161: Distributional and phase features for VQI-V.

35 features:
  V127-V130: LTFD (Long-Term Feature Distribution, 4 features)
  V131-V137: LTAS levels (5 band levels + 2 ratios)
  V138-V150: MGDCC (Modified Group Delay Cepstral Coefficients, 13)
  V151-V156: Rhythm features (6)
  V157-V161: Spectral identity (5)
"""

import numpy as np
from scipy.fft import dct


def compute_distributional_phase_features(waveform, sr, vad_mask, intermediates):
    features = {}
    S = intermediates["stft_power"]
    mfccs = intermediates["mfccs"]
    n_frames = S.shape[1]
    n_bins = S.shape[0]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12
    freqs = np.linspace(0, sr / 2, n_bins)

    speech_mfccs = mfccs[:, vad] if np.any(vad) else mfccs

    # V127-V130: LTFD (Long-Term Feature Distribution)
    # Statistics of the MFCC distribution over time
    if speech_mfccs.shape[1] > 2:
        # Flatness of MFCC distribution
        mfcc_means = np.mean(speech_mfccs[1:14, :], axis=1)
        features["V_LTFD_Flatness"] = float(np.exp(np.mean(np.log(np.abs(mfcc_means) + eps))) /
                                             (np.mean(np.abs(mfcc_means)) + eps))
        # Entropy of MFCC frame distribution
        mfcc_energy = np.sum(speech_mfccs[1:14, :] ** 2, axis=0)
        mfcc_energy_norm = mfcc_energy / (np.sum(mfcc_energy) + eps)
        features["V_LTFD_Entropy"] = float(-np.sum(mfcc_energy_norm * np.log(mfcc_energy_norm + eps)))
        # Kurtosis of MFCC temporal distribution
        from scipy.stats import kurtosis
        features["V_LTFD_Kurtosis"] = float(kurtosis(mfcc_energy, fisher=True))
        if not np.isfinite(features["V_LTFD_Kurtosis"]):
            features["V_LTFD_Kurtosis"] = 0.0
        # Range of MFCC energy
        features["V_LTFD_Range"] = float(np.max(mfcc_energy) - np.min(mfcc_energy))
    else:
        features["V_LTFD_Flatness"] = 0.0
        features["V_LTFD_Entropy"] = 0.0
        features["V_LTFD_Kurtosis"] = 0.0
        features["V_LTFD_Range"] = 0.0

    # V131-V137: LTAS levels
    mean_spec = np.mean(S[:, vad], axis=1) if np.any(vad) else np.mean(S, axis=1)
    mean_spec_db = 10.0 * np.log10(mean_spec + eps)

    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_levels = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            band_levels.append(float(np.mean(mean_spec_db[mask])))
        else:
            band_levels.append(-60.0)

    for i, (lo, hi) in enumerate(bands):
        features[f"V_LTAS_{lo}_{hi}"] = band_levels[i]

    # Ratios
    features["V_LTAS_LowMidRatio"] = float(band_levels[0] - band_levels[2])
    features["V_LTAS_MidHighRatio"] = float(band_levels[2] - band_levels[4])

    # V138-V150: MGDCC (Modified Group Delay Cepstral Coefficients)
    # Group delay = -d(phase)/d(omega)
    stft_complex = np.sqrt(S + eps) * np.exp(1j * np.random.randn(*S.shape) * 0.01)
    # For proper MGDCC, compute from actual STFT
    try:
        import librosa
        stft_full = librosa.stft(waveform, n_fft=intermediates["n_fft"],
                                  hop_length=intermediates["hop_length"])
        phase = np.angle(stft_full)
        # Group delay: negative derivative of phase w.r.t. frequency
        if phase.shape[0] > 1:
            gd = -np.diff(phase, axis=0)
            # Modified group delay: apply cepstral smoothing
            mean_gd = np.mean(gd[:, vad], axis=1) if np.any(vad) else np.mean(gd, axis=1)
            # DCT to get cepstral coefficients
            mgdcc = dct(mean_gd, type=2, norm="ortho")
            for i in range(13):
                if i < len(mgdcc):
                    features[f"V_MGDCC_{i+1}"] = float(mgdcc[i])
                else:
                    features[f"V_MGDCC_{i+1}"] = 0.0
        else:
            for i in range(13):
                features[f"V_MGDCC_{i+1}"] = 0.0
    except Exception:
        for i in range(13):
            features[f"V_MGDCC_{i+1}"] = 0.0

    # V151-V156: Rhythm features
    hop = intermediates["hop_length"]
    frame_dur = hop / sr

    # Find speech and silence segments
    speech_lens = []
    silence_lens = []
    current_len = 0
    current_is_speech = bool(vad_mask[0]) if len(vad_mask) > 0 else False

    for v in vad_mask:
        if bool(v) == current_is_speech:
            current_len += 1
        else:
            if current_is_speech:
                speech_lens.append(current_len * frame_dur)
            else:
                silence_lens.append(current_len * frame_dur)
            current_len = 1
            current_is_speech = bool(v)
    if current_len > 0:
        if current_is_speech:
            speech_lens.append(current_len * frame_dur)
        else:
            silence_lens.append(current_len * frame_dur)

    # PVI (Pairwise Variability Index) of speech segments
    if len(speech_lens) > 1:
        diffs = [abs(speech_lens[i] - speech_lens[i + 1])
                 for i in range(len(speech_lens) - 1)]
        features["V_Rhythm_nPVI"] = float(np.mean(diffs) / (np.mean(speech_lens) + eps))
    else:
        features["V_Rhythm_nPVI"] = 0.0

    # Percentage of voicing
    features["V_Rhythm_VoicedPct"] = float(np.mean(vad_mask))

    # Speech segment variability
    features["V_Rhythm_SpeechSegVar"] = float(np.std(speech_lens)) if speech_lens else 0.0

    # Silence segment variability
    features["V_Rhythm_SilenceSegVar"] = float(np.std(silence_lens)) if silence_lens else 0.0

    # Speech-to-silence ratio (clamped to [0, 1000])
    total_speech = sum(speech_lens)
    total_silence = sum(silence_lens) + eps
    features["V_Rhythm_SpeechSilenceRatio"] = float(np.clip(total_speech / total_silence, 0, 1000))

    # Tempo variability (onset interval std)
    try:
        import librosa
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                             hop_length=hop, units="time")
        if len(onsets) > 2:
            ioi = np.diff(onsets)
            features["V_Rhythm_TempoVar"] = float(np.std(ioi) / (np.mean(ioi) + eps))
        else:
            features["V_Rhythm_TempoVar"] = 0.0
    except Exception:
        features["V_Rhythm_TempoVar"] = 0.0

    # V157-V161: Spectral identity features
    # Spectral centroid mean (speaker-characteristic)
    sc = np.sum(freqs[:, np.newaxis] * S, axis=0) / (np.sum(S, axis=0) + eps)
    speech_sc = sc[vad] if np.any(vad) else sc
    features["V_SpectralCentroid_Mean"] = float(np.mean(speech_sc))
    features["V_SpectralCentroid_Std"] = float(np.std(speech_sc))

    # Spectral bandwidth (speaker-characteristic)
    centroid = features["V_SpectralCentroid_Mean"]
    bw = np.sqrt(np.sum(S * (freqs[:, np.newaxis] - centroid) ** 2, axis=0) /
                 (np.sum(S, axis=0) + eps))
    speech_bw = bw[vad] if np.any(vad) else bw
    features["V_SpectralBW_Mean"] = float(np.mean(speech_bw))

    # Spectral entropy (speaker-characteristic variation)
    S_norm = S / (S.sum(axis=0, keepdims=True) + eps)
    entropy = -np.sum(S_norm * np.log(S_norm + eps), axis=0) / np.log(n_bins)
    speech_ent = entropy[vad] if np.any(vad) else entropy
    features["V_SpectralEntropy_Mean"] = float(np.mean(speech_ent))
    features["V_SpectralEntropy_Std"] = float(np.std(speech_ent))

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
