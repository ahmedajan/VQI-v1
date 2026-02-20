"""
Shared intermediate computations for feature extraction (Sub-task 4.1b).

Computes expensive representations ONCE and returns a dict that frame-level
and global feature modules can pull from.  Avoids redundant STFT, pitch,
Praat object creation, etc.
"""

import logging
import numpy as np
import librosa
import parselmouth
from scipy.signal import hilbert, lfilter

logger = logging.getLogger(__name__)


def compute_shared_intermediates(waveform, sr, vad_mask):
    """Compute shared intermediate representations.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32/float64, preprocessed (normalized, 16 kHz).
    sr : int
        Sample rate (16000).
    vad_mask : np.ndarray
        Boolean array, shape (n_frames,), from energy_vad().

    Returns
    -------
    dict
        Keys described below. Each value computed once.
    """
    intermediates = {}
    n_fft = 512
    hop_length = 160

    waveform = np.asarray(waveform, dtype=np.float64)
    intermediates["waveform"] = waveform
    intermediates["sr"] = sr
    intermediates["vad_mask"] = vad_mask
    intermediates["n_fft"] = n_fft
    intermediates["hop_length"] = hop_length

    # ---- STFT power spectrogram ----
    S = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                            window="hann")) ** 2
    intermediates["stft_power"] = S  # shape (n_fft//2+1, n_frames)

    # ---- Magnitude spectrogram (for features needing magnitude) ----
    intermediates["stft_mag"] = np.sqrt(S)

    # ---- MFCCs ----
    mfccs = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop_length,
        n_mels=40, fmin=20, fmax=8000,
    )
    intermediates["mfccs"] = mfccs  # shape (14, n_frames)

    # ---- Delta MFCCs ----
    delta_mfcc = librosa.feature.delta(mfccs, order=1)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    intermediates["delta_mfcc"] = delta_mfcc
    intermediates["delta2_mfcc"] = delta2_mfcc

    # ---- Pitch (pYIN) ---- frame_length=2048 for reliable low-F0 estimation
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waveform, fmin=60, fmax=500, sr=sr,
            frame_length=2048, hop_length=hop_length,
        )
        # Replace NaN F0 with 0
        f0 = np.where(np.isnan(f0), 0.0, f0)
    except Exception:
        n_pitch_frames = 1 + (len(waveform) - 2048) // hop_length
        if n_pitch_frames < 1:
            n_pitch_frames = 1
        f0 = np.zeros(n_pitch_frames)
        voiced_flag = np.zeros(n_pitch_frames, dtype=bool)
        voiced_prob = np.zeros(n_pitch_frames)
        logger.warning("pYIN failed, using zero F0")
    intermediates["f0"] = f0
    intermediates["voiced_flag"] = voiced_flag
    intermediates["voiced_prob"] = voiced_prob

    # ---- Praat Sound object ----
    praat_sound = parselmouth.Sound(waveform, sampling_frequency=float(sr))
    intermediates["praat_sound"] = praat_sound

    # ---- Praat Harmonicity (autocorrelation method) ----
    try:
        harmonicity = praat_sound.to_harmonicity_ac(
            time_step=0.01, minimum_pitch=75.0,
            silence_threshold=0.1, periods_per_window=1.0,
        )
        intermediates["praat_harmonicity"] = harmonicity
    except Exception:
        intermediates["praat_harmonicity"] = None
        logger.warning("Praat harmonicity computation failed")

    # ---- Praat Pitch (autocorrelation) ----
    try:
        praat_pitch = praat_sound.to_pitch_ac(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0,
        )
        intermediates["praat_pitch"] = praat_pitch
    except Exception:
        intermediates["praat_pitch"] = None
        logger.warning("Praat pitch computation failed")

    # ---- Praat PointProcess (for jitter/shimmer) ----
    try:
        if intermediates["praat_pitch"] is not None:
            pp = parselmouth.praat.call(
                [praat_sound, intermediates["praat_pitch"]],
                "To PointProcess (cc)",
            )
            intermediates["praat_point_process"] = pp
        else:
            intermediates["praat_point_process"] = None
    except Exception:
        intermediates["praat_point_process"] = None
        logger.warning("Praat PointProcess computation failed")

    # ---- Praat Formant (Burg) ----
    try:
        formant = praat_sound.to_formant_burg(
            time_step=0.01, max_number_of_formants=5,
            maximum_formant=5500.0, window_length=0.025,
            pre_emphasis_from=50.0,
        )
        intermediates["praat_formant"] = formant
    except Exception:
        intermediates["praat_formant"] = None
        logger.warning("Praat formant computation failed")

    # ---- Frame energy (RMS per frame) ----
    frames = librosa.util.frame(waveform, frame_length=n_fft, hop_length=hop_length)
    frame_energy = np.sqrt(np.mean(frames ** 2, axis=0))
    intermediates["frames"] = frames  # shape (n_fft, n_frames)
    intermediates["frame_energy"] = frame_energy  # shape (n_frames,)

    # ---- LPC per frame (order=14) ----
    lpc_order = 14
    lpc_list = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        # Apply Hamming window for LPC
        windowed = frame * np.hamming(len(frame))
        try:
            a = librosa.lpc(windowed, order=lpc_order)
            lpc_list.append(a)
        except Exception:
            lpc_list.append(np.zeros(lpc_order + 1))
    intermediates["lpc_per_frame"] = lpc_list  # list of arrays, each len 15

    # ---- Hilbert envelope ----
    try:
        analytic = hilbert(waveform)
        intermediates["hilbert_envelope"] = np.abs(analytic)
    except Exception:
        intermediates["hilbert_envelope"] = np.abs(waveform)
        logger.warning("Hilbert transform failed, using abs(waveform)")

    return intermediates
