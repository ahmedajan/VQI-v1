"""
Audio loading and format normalization for VQI (Step 3.1).

Implements Algorithm 1 from the VQI blueprint:
  1. Load audio via torchaudio
  2. Mix to mono if multi-channel
  3. Resample to 16kHz using sinc_interp_kaiser
  4. Validate duration [1.0s, 120.0s]
  5. Handle edge cases (NaN/Inf, corrupt files, too short/long)
"""

import logging
import numpy as np
import torch
import torchaudio

from .exceptions import AudioLoadError, TooShortError

logger = logging.getLogger(__name__)

# Target parameters
TARGET_SR = 16000
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 120.0
MIN_SAMPLES = int(TARGET_SR * MIN_DURATION_SEC)    # 16000
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)     # 1920000


def load_audio(filepath: str) -> np.ndarray:
    """Load an audio file, resample to 16kHz mono float32.

    Parameters
    ----------
    filepath : str
        Path to audio file (WAV, FLAC, or any format supported by
        the torchaudio/soundfile backend).

    Returns
    -------
    waveform : np.ndarray
        1-D float32 array at 16kHz, shape (num_samples,).
        Duration guaranteed in [1.0s, 120.0s].

    Raises
    ------
    AudioLoadError
        If the file cannot be loaded or is empty.
    TooShortError
        If the audio is shorter than 1.0 second after resampling.
    """
    # Step 1: Load audio
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        raise AudioLoadError(f"Cannot load audio: {filepath} -> {e}") from e

    # Check for empty file
    if waveform.numel() == 0:
        raise AudioLoadError(f"Empty audio file: {filepath}")

    # Check for NaN/Inf and replace with zeros
    bad_mask = ~torch.isfinite(waveform)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.warning(
            "Replaced %d NaN/Inf samples with zeros in %s", n_bad, filepath
        )
        waveform = torch.where(bad_mask, torch.zeros_like(waveform), waveform)

    # Step 2: Mix to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 3: Resample to 16kHz if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=TARGET_SR,
            resampling_method="sinc_interp_kaiser",
            rolloff=0.99,
            lowpass_filter_width=6,
        )
        waveform = resampler(waveform)

    # Step 4: Squeeze to 1-D
    waveform = waveform.squeeze(0)  # shape: (num_samples,)

    # Step 5: Duration validation
    num_samples = waveform.shape[0]

    # Too short -> raise
    if num_samples < MIN_SAMPLES:
        duration = num_samples / TARGET_SR
        raise TooShortError(
            f"Audio too short: {duration:.2f}s < {MIN_DURATION_SEC}s ({filepath})"
        )

    # Too long -> truncate from center
    if num_samples > MAX_SAMPLES:
        center = num_samples // 2
        half = MAX_SAMPLES // 2
        waveform = waveform[center - half : center - half + MAX_SAMPLES]
        logger.info(
            "Truncated %s from %d to %d samples (center crop)",
            filepath, num_samples, MAX_SAMPLES,
        )

    # Convert to numpy float32
    return waveform.numpy().astype(np.float32)
