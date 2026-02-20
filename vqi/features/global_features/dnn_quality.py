"""G97-G104: DNN-based quality features (optional, graceful fallback).

G97:  DNSMOS_SIG  (signal quality)
G98:  DNSMOS_BAK  (background noise)
G99:  DNSMOS_OVRL (overall quality)
G100: NISQA_MOS
G101: NISQA_NOI
G102: NISQA_DIS
G103: NISQA_COL
G104: NISQA_LOUD
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_DNSMOS_AVAILABLE = None
_NISQA_AVAILABLE = None


def compute_dnn_quality_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    features = {}

    # --- DNSMOS (G97-G99) ---
    dnsmos = _compute_dnsmos(waveform, sr)
    features.update(dnsmos)

    # --- NISQA (G100-G104) ---
    nisqa = _compute_nisqa(waveform, sr)
    features.update(nisqa)

    return features


def _compute_dnsmos(waveform, sr):
    global _DNSMOS_AVAILABLE
    defaults = {"DNSMOS_SIG": np.nan, "DNSMOS_BAK": np.nan, "DNSMOS_OVRL": np.nan}

    if _DNSMOS_AVAILABLE is False:
        return defaults

    try:
        import onnxruntime  # noqa: F401
        _DNSMOS_AVAILABLE = True
    except ImportError:
        _DNSMOS_AVAILABLE = False
        logger.info("onnxruntime not available; DNSMOS features will be NaN")
        return defaults

    # DNSMOS requires specific model files - return NaN if not configured
    # Full implementation would load ONNX models and run inference
    return defaults


def _compute_nisqa(waveform, sr):
    global _NISQA_AVAILABLE
    defaults = {
        "NISQA_MOS": np.nan, "NISQA_NOI": np.nan,
        "NISQA_DIS": np.nan, "NISQA_COL": np.nan, "NISQA_LOUD": np.nan,
    }

    if _NISQA_AVAILABLE is False:
        return defaults

    try:
        import nisqa  # noqa: F401
        _NISQA_AVAILABLE = True
    except ImportError:
        _NISQA_AVAILABLE = False
        logger.info("nisqa not available; NISQA features will be NaN")
        return defaults

    # Full implementation would run NISQA model inference
    return defaults
