"""Ranked DET (Detection Error Tradeoff) curves (Sub-tasks 8.2, 8.8).

Partitions comparison pairs by pairwise quality into 3 groups:
  - Bottom 15% quality
  - Middle 70% quality
  - Top 15% quality

Then computes DET curves for each group.

Design rationale: 15%/70%/15% splits isolate extremes symmetrically.
The 70% represents "typical" quality. Demonstrates both FNMR and FMR
predictive power.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_det_curve(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    n_thresholds: int = 500,
) -> Dict:
    """Compute DET curve (FMR vs FNMR) for a set of scores.

    Args:
        genuine_sim: (N_gen,) genuine similarity scores.
        impostor_sim: (N_imp,) impostor similarity scores.
        n_thresholds: number of threshold steps.

    Returns:
        Dict with fmr, fnmr, thresholds, eer.
    """
    all_scores = np.concatenate([genuine_sim, impostor_sim])
    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    thresholds = np.linspace(lo, hi, n_thresholds)

    fmr = np.zeros(n_thresholds)
    fnmr = np.zeros(n_thresholds)
    n_gen = len(genuine_sim)
    n_imp = len(impostor_sim)

    for i, tau in enumerate(thresholds):
        if n_gen > 0:
            fnmr[i] = np.mean(genuine_sim < tau)
        if n_imp > 0:
            fmr[i] = np.mean(impostor_sim >= tau)

    # EER: point where FMR ~= FNMR
    diff = np.abs(fmr - fnmr)
    eer_idx = np.argmin(diff)
    eer = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0)

    return {
        "fmr": fmr,
        "fnmr": fnmr,
        "thresholds": thresholds,
        "eer": eer,
        "eer_idx": int(eer_idx),
    }


def compute_fnmr_at_fmr(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    target_fmr: float,
) -> float:
    """Compute FNMR at a given FMR operating point.

    Args:
        genuine_sim: genuine similarity scores.
        impostor_sim: impostor similarity scores.
        target_fmr: target FMR (e.g., 0.01 for 1%, 0.001 for 0.1%).

    Returns:
        FNMR at the threshold achieving target_fmr.
    """
    if len(impostor_sim) == 0 or len(genuine_sim) == 0:
        return np.nan

    # Find threshold where FMR = target_fmr
    # FMR = P(impostor >= tau) = target_fmr
    # tau = percentile(impostor, 100 * (1 - target_fmr))
    tau = float(np.percentile(impostor_sim, 100 * (1.0 - target_fmr)))
    fnmr = float(np.mean(genuine_sim < tau))
    return fnmr


def compute_ranked_det(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    quality_genuine: np.ndarray,
    quality_impostor: np.ndarray,
    percentiles: Tuple[float, float] = (15.0, 85.0),
) -> Dict:
    """Compute Ranked DET curves by quality group.

    Partitions pairs into 3 quality groups based on pairwise quality:
      - bottom: quality < P15
      - middle: P15 <= quality <= P85
      - top: quality > P85

    Args:
        genuine_sim: (N_gen,) genuine similarity scores.
        impostor_sim: (N_imp,) impostor similarity scores.
        quality_genuine: (N_gen,) pairwise quality for genuine pairs.
        quality_impostor: (N_imp,) pairwise quality for impostor pairs.
        percentiles: (low, high) percentile boundaries.

    Returns:
        Dict with keys bottom, middle, top, each containing a DET dict.
        Also includes quality thresholds and group sizes.
    """
    all_quality = np.concatenate([quality_genuine, quality_impostor])
    q_low = float(np.percentile(all_quality, percentiles[0]))
    q_high = float(np.percentile(all_quality, percentiles[1]))

    groups = {}
    for group_name, lo, hi in [
        ("bottom", -np.inf, q_low),
        ("middle", q_low, q_high),
        ("top", q_high, np.inf),
    ]:
        if group_name == "bottom":
            gen_mask = quality_genuine < q_low
            imp_mask = quality_impostor < q_low
        elif group_name == "top":
            gen_mask = quality_genuine > q_high
            imp_mask = quality_impostor > q_high
        else:
            gen_mask = (quality_genuine >= q_low) & (quality_genuine <= q_high)
            imp_mask = (quality_impostor >= q_low) & (quality_impostor <= q_high)

        gen_sub = genuine_sim[gen_mask]
        imp_sub = impostor_sim[imp_mask]

        if len(gen_sub) == 0 or len(imp_sub) == 0:
            logger.warning(
                "Group '%s' has %d genuine, %d impostor pairs - skipping DET",
                group_name, len(gen_sub), len(imp_sub),
            )
            groups[group_name] = {
                "det": None,
                "n_genuine": len(gen_sub),
                "n_impostor": len(imp_sub),
                "fnmr_at_fmr_001": np.nan,
                "fnmr_at_fmr_0001": np.nan,
            }
            continue

        det = compute_det_curve(gen_sub, imp_sub)
        fnmr_001 = compute_fnmr_at_fmr(gen_sub, imp_sub, 0.01)
        fnmr_0001 = compute_fnmr_at_fmr(gen_sub, imp_sub, 0.001)

        groups[group_name] = {
            "det": det,
            "n_genuine": len(gen_sub),
            "n_impostor": len(imp_sub),
            "fnmr_at_fmr_001": fnmr_001,
            "fnmr_at_fmr_0001": fnmr_0001,
        }

    # Compute EER separation ratio
    eer_bottom = groups["bottom"]["det"]["eer"] if groups["bottom"]["det"] else np.nan
    eer_top = groups["top"]["det"]["eer"] if groups["top"]["det"] else np.nan
    if eer_top > 0 and not np.isnan(eer_bottom):
        eer_separation = eer_bottom / eer_top
    else:
        eer_separation = np.nan

    return {
        "groups": groups,
        "q_low": q_low,
        "q_high": q_high,
        "eer_separation": eer_separation,
    }
