"""VQI-V Feature Orchestrator (Sub-task 4.26).

Orchestrates computation of all 161 VQI-V (Voice Distinctiveness) features.
Reuses intermediates from VQI-S to avoid redundant computation.
"""

import logging
import numpy as np

from ..features.shared_intermediates import compute_shared_intermediates
from ..features_v import VQIV_FEATURE_MODULES

logger = logging.getLogger(__name__)

N_TOTAL_V = 161


def compute_all_features_v(waveform, sr, vad_mask, intermediates=None):
    """Compute all 161 VQI-V features.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float, preprocessed (normalized, 16 kHz).
    sr : int
        Sample rate (16000).
    vad_mask : np.ndarray
        Boolean array from energy_vad().
    intermediates : dict, optional
        Shared intermediates from VQI-S orchestrator. If None, computes them.

    Returns
    -------
    features_v_dict : dict[str, float]
        161 named features.
    features_v_array : np.ndarray
        Shape (161,), ordered V1-V161.
    """
    if intermediates is None:
        intermediates = compute_shared_intermediates(waveform, sr, vad_mask)

    features_v_dict = {}

    for func, expected_count in VQIV_FEATURE_MODULES:
        try:
            feats = func(waveform, sr, vad_mask, intermediates)
            features_v_dict.update(feats)
        except Exception as e:
            logger.warning(f"VQI-V module {func.__name__} failed: {e}")
            # Will be filled with 0 in array construction

    # Build ordered array
    feature_names = get_feature_names_v()
    features_v_array = np.zeros(N_TOTAL_V, dtype=np.float64)
    for i, name in enumerate(feature_names):
        val = features_v_dict.get(name, 0.0)
        features_v_array[i] = val if np.isfinite(val) else 0.0

    return features_v_dict, features_v_array


def get_feature_names_v():
    """Return ordered list of 161 VQI-V feature names."""
    names = []

    # V1-V13: MFCC means
    for i in range(13):
        names.append(f"V_MFCC_Mean_{i+1}")
    # V14-V26: MFCC stds
    for i in range(13):
        names.append(f"V_MFCC_Std_{i+1}")
    # V27-V39: Delta-MFCC means
    for i in range(13):
        names.append(f"V_DeltaMFCC_Mean_{i+1}")
    # V40-V52: LPCC means
    for i in range(13):
        names.append(f"V_LPCC_Mean_{i+1}")
    # V53-V65: LFCC means
    for i in range(13):
        names.append(f"V_LFCC_Mean_{i+1}")

    # V66-V79: LSF means
    for i in range(14):
        names.append(f"V_LSF_Mean_{i+1}")
    # V80-V87: Reflection coefficients
    for i in range(8):
        names.append(f"V_RC_{i+1}")
    # V88-V95: Log Area Ratios
    for i in range(8):
        names.append(f"V_LAR_{i+1}")
    # V96-V98: LPC gain stats
    names.extend(["V_LPCGain_Mean", "V_LPCGain_Std", "V_LPCGain_Range"])

    # V99-V109: Formant identity
    names.extend([
        "V_F2F1_Ratio", "V_F3F2_Ratio", "V_F3F1_Ratio", "V_VTL",
        "V_F1_Dynamics", "V_F2_Dynamics", "V_F3_Dynamics",
        "V_FormantCentralization",
        "V_F1F2_Corr", "V_F2F3_Corr", "V_F1F3_Corr",
    ])

    # V110-V126: Prosodic voice
    names.extend([
        "V_ArticulationRate",
        "V_F0_Mean", "V_F0_Std", "V_F0_Range", "V_F0_Slope", "V_F0_Median",
        "V_OQ", "V_CQ", "V_SQ", "V_Rd", "V_Ra", "V_Rk",
        "V_ModalProportion", "V_BreathyProportion",
        "V_AlphaRatio", "V_HarmonicRichness", "V_SpectralTilt",
    ])

    # V127-V161: Distributional/phase
    names.extend([
        "V_LTFD_Flatness", "V_LTFD_Entropy", "V_LTFD_Kurtosis", "V_LTFD_Range",
    ])
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    for lo, hi in bands:
        names.append(f"V_LTAS_{lo}_{hi}")
    names.extend(["V_LTAS_LowMidRatio", "V_LTAS_MidHighRatio"])
    for i in range(13):
        names.append(f"V_MGDCC_{i+1}")
    names.extend([
        "V_Rhythm_nPVI", "V_Rhythm_VoicedPct", "V_Rhythm_SpeechSegVar",
        "V_Rhythm_SilenceSegVar", "V_Rhythm_SpeechSilenceRatio", "V_Rhythm_TempoVar",
    ])
    names.extend([
        "V_SpectralCentroid_Mean", "V_SpectralCentroid_Std",
        "V_SpectralBW_Mean",
        "V_SpectralEntropy_Mean", "V_SpectralEntropy_Std",
    ])

    assert len(names) == N_TOTAL_V, f"Expected {N_TOTAL_V} VQI-V names, got {len(names)}"
    return names
