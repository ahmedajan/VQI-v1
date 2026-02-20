"""
DC offset removal and peak normalization for VQI (Step 3.2).

Pipeline:
  1. Remove DC offset: x = x - mean(x)
  2. Peak normalize to [-0.95, 0.95]: x = x / (max|x| + eps) * 0.95
"""

import numpy as np


def dc_remove_and_normalize(
    waveform,
    peak_target: float = 0.95,
    eps: float = 1e-8,
) -> np.ndarray:
    """Remove DC offset and peak-normalize a waveform.

    Parameters
    ----------
    waveform : np.ndarray or torch.Tensor
        1-D float32 audio signal. Torch tensors are converted to numpy.
    peak_target : float
        Target peak amplitude (default 0.95).
    eps : float
        Small constant to prevent division by zero (default 1e-8).

    Returns
    -------
    np.ndarray
        Normalized waveform with DC offset removed and peak at +-peak_target.
        All-zero input returns all zeros.
    """
    # Convert torch tensors to numpy
    if not isinstance(waveform, np.ndarray):
        waveform = np.asarray(waveform)

    # DC removal
    x = waveform - waveform.mean()

    # Peak normalization
    peak = np.abs(x).max()
    x = x / (peak + eps) * peak_target

    return x
