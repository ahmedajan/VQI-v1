"""Per-quadrant performance analysis (Sub-task 8.11).

Assigns pairs to quadrants based on VQI-S and VQI-V quality scores,
then computes per-quadrant recognition performance metrics.

Quadrants (using median as default threshold):
  Q1: high S, high V  (best quality)
  Q2: low S, high V   (signal issues, voice OK)
  Q3: low S, low V    (worst quality)
  Q4: high S, low V   (signal OK, voice indistinct)

Q1 should have lowest EER, Q3 highest.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from .det import compute_det_curve, compute_fnmr_at_fmr

logger = logging.getLogger(__name__)


def assign_quadrants(
    quality_s: np.ndarray,
    quality_v: np.ndarray,
    threshold_s: Optional[float] = None,
    threshold_v: Optional[float] = None,
) -> np.ndarray:
    """Assign each sample to a quadrant.

    Args:
        quality_s: (N,) VQI-S quality scores.
        quality_v: (N,) VQI-V quality scores.
        threshold_s: S threshold (default: median of quality_s).
        threshold_v: V threshold (default: median of quality_v).

    Returns:
        (N,) array of quadrant labels (1, 2, 3, 4).
    """
    if threshold_s is None:
        threshold_s = float(np.median(quality_s))
    if threshold_v is None:
        threshold_v = float(np.median(quality_v))

    quadrants = np.zeros(len(quality_s), dtype=int)
    high_s = quality_s >= threshold_s
    high_v = quality_v >= threshold_v

    quadrants[high_s & high_v] = 1   # Q1: high S, high V
    quadrants[~high_s & high_v] = 2  # Q2: low S, high V
    quadrants[~high_s & ~high_v] = 3  # Q3: low S, low V
    quadrants[high_s & ~high_v] = 4  # Q4: high S, low V

    return quadrants


def assign_pair_quadrants(
    quality_s: np.ndarray,
    quality_v: np.ndarray,
    pair_indices: np.ndarray,
    threshold_s: Optional[float] = None,
    threshold_v: Optional[float] = None,
) -> np.ndarray:
    """Assign each pair to a quadrant based on min quality of both samples.

    Uses min(q_s_1, q_s_2) for S and min(q_v_1, q_v_2) for V.

    Args:
        quality_s: (N,) VQI-S scores per sample.
        quality_v: (N,) VQI-V scores per sample.
        pair_indices: (M, 2) index pairs.
        threshold_s: S threshold (default: median).
        threshold_v: V threshold (default: median).

    Returns:
        (M,) quadrant assignments.
    """
    min_s = np.minimum(quality_s[pair_indices[:, 0]], quality_s[pair_indices[:, 1]])
    min_v = np.minimum(quality_v[pair_indices[:, 0]], quality_v[pair_indices[:, 1]])
    return assign_quadrants(min_s, min_v, threshold_s, threshold_v)


def compute_quadrant_performance(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    quadrant_genuine: np.ndarray,
    quadrant_impostor: np.ndarray,
) -> Dict:
    """Compute recognition performance per quadrant.

    Args:
        genuine_sim: (N_gen,) genuine similarity scores.
        impostor_sim: (N_imp,) impostor similarity scores.
        quadrant_genuine: (N_gen,) quadrant assignment for genuine pairs.
        quadrant_impostor: (N_imp,) quadrant assignment for impostor pairs.

    Returns:
        Dict[quadrant_label] -> performance metrics.
    """
    results = {}

    for q_label in [1, 2, 3, 4]:
        gen_mask = quadrant_genuine == q_label
        imp_mask = quadrant_impostor == q_label

        n_gen = int(gen_mask.sum())
        n_imp = int(imp_mask.sum())

        if n_gen == 0 or n_imp == 0:
            logger.warning("Quadrant Q%d has %d genuine, %d impostor - skipping", q_label, n_gen, n_imp)
            results[f"Q{q_label}"] = {
                "n_genuine": n_gen,
                "n_impostor": n_imp,
                "eer": np.nan,
                "fnmr_at_fmr_001": np.nan,
                "fnmr_at_fmr_0001": np.nan,
                "genuine_mean": np.nan,
                "impostor_mean": np.nan,
            }
            continue

        gen_sub = genuine_sim[gen_mask]
        imp_sub = impostor_sim[imp_mask]

        det = compute_det_curve(gen_sub, imp_sub)
        fnmr_001 = compute_fnmr_at_fmr(gen_sub, imp_sub, 0.01)
        fnmr_0001 = compute_fnmr_at_fmr(gen_sub, imp_sub, 0.001)

        results[f"Q{q_label}"] = {
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "eer": det["eer"],
            "fnmr_at_fmr_001": fnmr_001,
            "fnmr_at_fmr_0001": fnmr_0001,
            "genuine_mean": float(gen_sub.mean()),
            "genuine_std": float(gen_sub.std()),
            "impostor_mean": float(imp_sub.mean()),
            "impostor_std": float(imp_sub.std()),
            "det": det,
        }

    # Verify Q1 < Q3 EER
    eer_q1 = results.get("Q1", {}).get("eer", np.nan)
    eer_q3 = results.get("Q3", {}).get("eer", np.nan)
    q1_lt_q3 = bool(eer_q1 < eer_q3) if not (np.isnan(eer_q1) or np.isnan(eer_q3)) else None

    return {
        "quadrants": results,
        "q1_eer_lt_q3_eer": q1_lt_q3,
        "eer_q1": eer_q1,
        "eer_q3": eer_q3,
    }
