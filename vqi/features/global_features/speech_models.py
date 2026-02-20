"""G61-G62: Speech model features.

G61: LPCResidualEnergy (mean LPC prediction residual energy)
G62: VocalTractRegularity (consistency of LPC coefficients over time)
"""

import numpy as np


def compute_speech_models_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}
    lpc_list = intermediates["lpc_per_frame"]
    frames = intermediates["frames"]
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)
    eps = 1e-12

    # G61: LPC residual energy
    residual_energies = []
    for i in range(n_frames):
        if not vad[i]:
            continue
        frame = frames[:, i]
        lpc_coeffs = lpc_list[i]
        energy = np.sum(frame ** 2)
        if energy < eps or np.all(lpc_coeffs == 0):
            continue
        from scipy.signal import lfilter
        residual = lfilter(lpc_coeffs, [1.0], frame)
        residual_energies.append(np.sum(residual ** 2) / (energy + eps))

    features["LPCResidualEnergy"] = float(np.mean(residual_energies)) if residual_energies else 0.0

    # G62: Vocal tract regularity (std of LPC coefficient variation over time)
    speech_lpc = [lpc_list[i] for i in range(n_frames) if vad[i] and not np.all(lpc_list[i] == 0)]
    if len(speech_lpc) > 2:
        lpc_array = np.array(speech_lpc)
        # Mean variance across coefficients
        coeff_var = np.var(lpc_array, axis=0)
        features["VocalTractRegularity"] = float(np.mean(coeff_var))
    else:
        features["VocalTractRegularity"] = 0.0

    return features


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
