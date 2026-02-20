"""G12: Speaker Turns (BIC-based speaker change detection on MFCCs)."""

import numpy as np


def compute_speaker_turns_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    mfccs = intermediates["mfccs"]  # (14, n_frames)
    n_frames = mfccs.shape[1]

    if n_frames < 40:
        return {"SpeakerTurns": 0.0}

    # Simple BIC-based change detection
    window = 20  # frames
    penalty = np.log(window) * mfccs.shape[0]  # BIC penalty
    changes = 0

    for i in range(window, n_frames - window, window // 2):
        left = mfccs[:, i - window:i].T
        right = mfccs[:, i:i + window].T
        combined = mfccs[:, i - window:i + window].T

        n_l, n_r, n_c = len(left), len(right), len(combined)
        if n_l < 2 or n_r < 2:
            continue

        cov_l = np.cov(left, rowvar=False) + 1e-6 * np.eye(mfccs.shape[0])
        cov_r = np.cov(right, rowvar=False) + 1e-6 * np.eye(mfccs.shape[0])
        cov_c = np.cov(combined, rowvar=False) + 1e-6 * np.eye(mfccs.shape[0])

        bic = (n_c * _log_det(cov_c) - n_l * _log_det(cov_l) - n_r * _log_det(cov_r)) / 2.0
        bic -= penalty

        if bic > 0:
            changes += 1

    return {"SpeakerTurns": float(changes)}


def _log_det(cov):
    sign, logdet = np.linalg.slogdet(cov)
    return logdet if sign > 0 else 0.0
