"""VQI-S Feature Orchestrator (Sub-task 4.19).

Orchestrates computation of all 544 VQI-S features:
  - 437 frame-level features (23 measures x 19 aggregation stats)
  - 107 global scalar features

Returns both a named dict and a numpy array in blueprint index order.
"""

import logging
import numpy as np

from ..features.shared_intermediates import compute_shared_intermediates
from ..features.frame_level import FRAME_FEATURE_MODULES
from ..features.global_features import GLOBAL_FEATURE_MODULES

logger = logging.getLogger(__name__)

# Total expected features
N_FRAME_FEATURES = 23 * 19  # 437
N_GLOBAL_FEATURES = 107
N_TOTAL_S = N_FRAME_FEATURES + N_GLOBAL_FEATURES  # 544


def compute_all_features(waveform, sr, vad_mask, raw_waveform=None):
    """Compute all 544 VQI-S features.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float, preprocessed (normalized, 16 kHz).
    sr : int
        Sample rate (16000).
    vad_mask : np.ndarray
        Boolean array from energy_vad().
    raw_waveform : np.ndarray, optional
        Raw waveform before normalization (for clipping detection).

    Returns
    -------
    features_dict : dict[str, float]
        544 named features.
    features_array : np.ndarray
        Shape (544,), ordered per blueprint index.
    intermediates : dict
        Shared intermediates for VQI-V reuse.
    """
    # Step 1: Compute shared intermediates ONCE
    intermediates = compute_shared_intermediates(waveform, sr, vad_mask)

    features_dict = {}

    # Step 2: Frame-level features (F1-F23, indices 0-436)
    for func, prefix in FRAME_FEATURE_MODULES:
        try:
            feats = func(waveform, sr, vad_mask, intermediates)
            features_dict.update(feats)
        except Exception as e:
            logger.warning(f"Frame feature {prefix} failed: {e}")
            # Fill with zeros
            for i in range(10):
                features_dict[f"{prefix}_Hist{i}"] = 0.0
            for stat in ("Mean", "Std", "Skew", "Kurt", "Median", "IQR",
                          "P5", "P95", "Range"):
                features_dict[f"{prefix}_{stat}"] = 0.0

    # Step 3: Global features (G1-G107, indices 437-543)
    for func, names in GLOBAL_FEATURE_MODULES:
        try:
            feats = func(waveform, sr, vad_mask, intermediates, raw_waveform=raw_waveform)
            features_dict.update(feats)
        except Exception as e:
            logger.warning(f"Global feature {names[0]} failed: {e}")
            for name in names:
                features_dict[name] = 0.0

    # Step 4: Build ordered array
    feature_names = get_feature_names_s()
    features_array = np.zeros(N_TOTAL_S, dtype=np.float64)
    for i, name in enumerate(feature_names):
        val = features_dict.get(name, 0.0)
        features_array[i] = val if np.isfinite(val) else 0.0

    return features_dict, features_array, intermediates


def get_feature_names_s():
    """Return ordered list of 544 VQI-S feature names."""
    names = []
    # Frame features (F1-F23)
    for _, prefix in FRAME_FEATURE_MODULES:
        for i in range(10):
            names.append(f"{prefix}_Hist{i}")
        for stat in ("Mean", "Std", "Skew", "Kurt", "Median", "IQR",
                      "P5", "P95", "Range"):
            names.append(f"{prefix}_{stat}")
    # Global features (G1-G107)
    for _, feat_names in GLOBAL_FEATURE_MODULES:
        names.extend(feat_names)
    assert len(names) == N_TOTAL_S, f"Expected {N_TOTAL_S} names, got {len(names)}"
    return names
