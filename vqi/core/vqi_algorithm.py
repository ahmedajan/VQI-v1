"""
Actionable quality feedback checks for VQI (Step 3.4).

Before full feature extraction, checks for degenerate audio conditions.
If any check triggers, VQI score = 0 and a feedback label is returned.

All amplitude-based checks (TooQuiet, Clipped) operate on the RAW waveform
BEFORE normalization to detect actual recording problems.

Four checks:
  - TooQuiet:            peak amplitude < 0.001
  - Clipped:             clipping ratio > 0.10
  - TooShort:            speech duration after VAD < 1.0s
  - InsufficientSpeech:  VAD speech ratio < 0.05
"""

from typing import List
import numpy as np

# Thresholds
QUIET_THRESHOLD = 0.001        # peak amplitude below this = digital silence
CLIPPING_THRESHOLD = 0.99      # samples at or above this are considered clipped
CLIPPING_RATIO_LIMIT = 0.10    # >10% clipped samples = catastrophic
MIN_SPEECH_DURATION = 1.0      # seconds
MIN_SPEECH_RATIO = 0.05        # <5% speech frames = wrong input


def check_actionable_feedback(
    raw_waveform: np.ndarray,
    vad_mask: np.ndarray,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> List[str]:
    """Check for degenerate audio conditions before feature extraction.

    Parameters
    ----------
    raw_waveform : np.ndarray
        1-D audio signal BEFORE normalization (raw amplitude values).
    vad_mask : np.ndarray
        Boolean array of shape (n_frames,) from energy_vad(). True = speech.
    hop_length : int
        Hop length used in VAD (default 160).
    sample_rate : int
        Sample rate (default 16000).

    Returns
    -------
    List[str]
        List of triggered feedback labels. Empty list = all checks passed.
        Possible labels: "TooQuiet", "Clipped", "TooShort", "InsufficientSpeech".
    """
    feedback = []

    # TooQuiet -- on raw waveform, before normalization
    peak = np.abs(raw_waveform).max()
    if peak < QUIET_THRESHOLD:
        feedback.append("TooQuiet")

    # Clipped -- on raw waveform, before normalization
    n_clipped = np.sum(np.abs(raw_waveform) >= CLIPPING_THRESHOLD)
    clipping_ratio = n_clipped / len(raw_waveform) if len(raw_waveform) > 0 else 0.0
    if clipping_ratio > CLIPPING_RATIO_LIMIT:
        feedback.append("Clipped")

    # TooShort -- speech duration after VAD
    n_speech_frames = int(np.sum(vad_mask)) if vad_mask.size > 0 else 0
    speech_duration = n_speech_frames * hop_length / sample_rate
    if speech_duration < MIN_SPEECH_DURATION:
        feedback.append("TooShort")

    # InsufficientSpeech -- VAD ratio
    n_total_frames = len(vad_mask) if vad_mask.size > 0 else 0
    vad_ratio = n_speech_frames / n_total_frames if n_total_frames > 0 else 0.0
    if vad_ratio < MIN_SPEECH_RATIO:
        feedback.append("InsufficientSpeech")

    return feedback
