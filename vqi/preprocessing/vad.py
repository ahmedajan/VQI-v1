"""
Energy-based Voice Activity Detection (VAD) for VQI.

Implements Algorithm 2 from the VQI blueprint (Phase/Step 3.3):
  1. Frame the signal with frame_length=512 (32ms), hop_length=160 (10ms)
  2. Compute RMS energy per frame -> dB
  3. Dynamic threshold: max(energy_db) - threshold_below_peak_db, floored at absolute_floor_db
  4. Binary mask: energy_db > threshold
  5. Median smoothing with kernel_size=11

This module is used by:
  - Step 2.1: compute speech duration after VAD
  - Step 3: full preprocessing pipeline (VAD + speech extraction)
  - Step 4: feature extraction (VAD mask determines speech frames)
"""

import numpy as np
from scipy.ndimage import median_filter


def energy_vad(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 512,
    hop_length: int = 160,
    threshold_below_peak_db: float = 40.0,
    absolute_floor_db: float = -60.0,
    median_kernel_size: int = 11,
) -> tuple:
    """Compute energy-based VAD mask for a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D audio signal (mono). If 2-D with shape (1, N) or (N, 1),
        it will be squeezed to 1-D automatically.
    sample_rate : int
        Sample rate of the waveform (default 16000). Only used for
        computing duration; no resampling is performed here.
    frame_length : int
        Number of samples per frame (default 512 = 32ms at 16kHz).
    hop_length : int
        Number of samples between frame starts (default 160 = 10ms at 16kHz).
    threshold_below_peak_db : float
        dB below peak energy to set as speech threshold (default 40).
    absolute_floor_db : float
        Absolute minimum threshold in dB (default -60). Prevents silence
        from being classified as speech in very quiet recordings.
    median_kernel_size : int
        Size of the median filter kernel for smoothing the VAD mask
        (default 11 = 110ms at 10ms hop = ~one syllable duration).

    Returns
    -------
    vad_mask : np.ndarray
        Boolean array of shape (n_frames,). True = speech frame.
    speech_duration_sec : float
        Total speech duration in seconds (sum of speech frames * hop / sr).
    speech_ratio : float
        Ratio of speech frames to total frames. In [0.0, 1.0].
    """
    # Ensure 1-D
    waveform = np.squeeze(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {waveform.shape}")

    n_samples = len(waveform)

    # Edge case: very short signal (fewer samples than one frame)
    if n_samples < frame_length:
        return np.array([], dtype=bool), 0.0, 0.0

    # --- Step 1: Frame the signal ---
    # Number of complete frames
    n_frames = 1 + (n_samples - frame_length) // hop_length
    # Build frame indices
    frame_starts = np.arange(n_frames) * hop_length
    frame_indices = frame_starts[:, None] + np.arange(frame_length)[None, :]
    frames = waveform[frame_indices]  # shape: (n_frames, frame_length)

    # --- Step 2: RMS energy per frame -> dB ---
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Avoid log(0) by clamping to a tiny positive value
    eps = 1e-10
    energy_db = 20.0 * np.log10(np.maximum(rms, eps))

    # --- Step 3: Dynamic threshold ---
    peak_db = np.max(energy_db)
    threshold_db = max(peak_db - threshold_below_peak_db, absolute_floor_db)

    # --- Step 4: Binary mask ---
    vad_mask = energy_db > threshold_db

    # --- Step 5: Median smoothing ---
    # Convert to float for median_filter, then back to bool
    vad_mask_smoothed = median_filter(
        vad_mask.astype(np.float32), size=median_kernel_size
    )
    vad_mask = vad_mask_smoothed > 0.5

    # --- Compute duration and ratio ---
    n_speech_frames = int(np.sum(vad_mask))
    speech_duration_sec = n_speech_frames * hop_length / sample_rate
    speech_ratio = n_speech_frames / n_frames if n_frames > 0 else 0.0

    return vad_mask, speech_duration_sec, speech_ratio


def reconstruct_from_mask(
    waveform: np.ndarray,
    vad_mask: np.ndarray,
    hop_length: int = 160,
    gap_samples: int = 160,
) -> np.ndarray:
    """Extract speech segments from waveform using a frame-level VAD mask.

    Converts the frame-level boolean mask to sample-level, extracts
    contiguous speech regions, and concatenates them with small silence
    gaps between segments.

    Parameters
    ----------
    waveform : np.ndarray
        1-D audio signal (mono).
    vad_mask : np.ndarray
        Boolean array of shape (n_frames,). True = speech frame.
    hop_length : int
        Hop length used to compute the VAD mask (default 160 = 10ms at 16kHz).
    gap_samples : int
        Number of zero samples to insert between speech segments
        (default 160 = 10ms at 16kHz).

    Returns
    -------
    np.ndarray
        1-D array containing concatenated speech segments with small
        silence gaps. Empty array if no speech frames detected.
    """
    waveform = np.squeeze(waveform)
    if vad_mask.size == 0 or not np.any(vad_mask):
        return np.array([], dtype=waveform.dtype)

    n_samples = len(waveform)

    # Build sample-level mask by expanding each speech frame
    sample_mask = np.zeros(n_samples, dtype=bool)
    for i, is_speech in enumerate(vad_mask):
        if is_speech:
            start = i * hop_length
            end = min(start + hop_length, n_samples)
            sample_mask[start:end] = True

    # Find contiguous speech segments
    diff = np.diff(sample_mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        segments.append(waveform[s:e])

    if not segments:
        return np.array([], dtype=waveform.dtype)

    # Concatenate with silence gaps between segments
    gap = np.zeros(gap_samples, dtype=waveform.dtype)
    parts = []
    for i, seg in enumerate(segments):
        if i > 0:
            parts.append(gap)
        parts.append(seg)

    return np.concatenate(parts)
