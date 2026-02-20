"""Combined ERC comparison for dual-score analysis (Sub-task 8.10).

Compares 4 rejection strategies using VQI-S and VQI-V:
  1. S-only: reject if VQI-S quality < q
  2. V-only: reject if VQI-V quality < q
  3. Union: reject if either S < q OR V < q (more aggressive)
  4. Intersection: reject if both S < q AND V < q (more conservative)

Overlays all 4 ERCs per provider to assess dual-score value.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .erc import compute_fnmr_at_threshold, compute_fmr_at_threshold

logger = logging.getLogger(__name__)


def _compute_erc_with_strategy(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    quality_gen_s: np.ndarray,
    quality_gen_v: np.ndarray,
    quality_imp_s: np.ndarray,
    quality_imp_v: np.ndarray,
    tau: float,
    strategy: str,
    q_range: np.ndarray,
) -> Dict:
    """Compute ERC for a specific rejection strategy.

    Args:
        genuine_sim: (N_gen,) genuine similarity scores.
        impostor_sim: (N_imp,) impostor similarity scores.
        quality_gen_s: (N_gen,) VQI-S pairwise quality for genuine pairs.
        quality_gen_v: (N_gen,) VQI-V pairwise quality for genuine pairs.
        quality_imp_s: (N_imp,) VQI-S pairwise quality for impostor pairs.
        quality_imp_v: (N_imp,) VQI-V pairwise quality for impostor pairs.
        tau: decision threshold.
        strategy: one of "s_only", "v_only", "union", "intersection".
        q_range: quality thresholds to sweep.

    Returns:
        Dict with reject_fracs, fnmr_values, fmr_values, etc.
    """
    n_total = len(genuine_sim) + len(impostor_sim)
    reject_fracs = np.zeros(len(q_range))
    fnmr_values = np.zeros(len(q_range))
    fmr_values = np.zeros(len(q_range))

    for i, q in enumerate(q_range):
        # Determine which pairs to keep based on strategy
        if strategy == "s_only":
            gen_keep = quality_gen_s >= q
            imp_keep = quality_imp_s >= q
        elif strategy == "v_only":
            gen_keep = quality_gen_v >= q
            imp_keep = quality_imp_v >= q
        elif strategy == "union":
            # Reject if either < q => keep if both >= q
            gen_keep = (quality_gen_s >= q) & (quality_gen_v >= q)
            imp_keep = (quality_imp_s >= q) & (quality_imp_v >= q)
        elif strategy == "intersection":
            # Reject if both < q => keep if at least one >= q
            gen_keep = (quality_gen_s >= q) | (quality_gen_v >= q)
            imp_keep = (quality_imp_s >= q) | (quality_imp_v >= q)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        n_remaining = int(gen_keep.sum()) + int(imp_keep.sum())
        reject_fracs[i] = 1.0 - n_remaining / n_total if n_total > 0 else 0.0

        if gen_keep.sum() > 0:
            fnmr_values[i] = compute_fnmr_at_threshold(genuine_sim[gen_keep], tau)
        else:
            fnmr_values[i] = np.nan

        if imp_keep.sum() > 0:
            fmr_values[i] = compute_fmr_at_threshold(impostor_sim[imp_keep], tau)
        else:
            fmr_values[i] = np.nan

    return {
        "reject_fracs": reject_fracs,
        "fnmr_values": fnmr_values,
        "fmr_values": fmr_values,
        "q_thresholds": q_range,
        "strategy": strategy,
    }


def compute_combined_erc(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    quality_gen_s: np.ndarray,
    quality_gen_v: np.ndarray,
    quality_imp_s: np.ndarray,
    quality_imp_v: np.ndarray,
    tau: float,
    q_range: Optional[np.ndarray] = None,
) -> Dict:
    """Compute ERCs for all 4 rejection strategies.

    Args:
        genuine_sim, impostor_sim: similarity scores.
        quality_gen_s, quality_gen_v: VQI-S/V pairwise quality for genuine pairs.
        quality_imp_s, quality_imp_v: VQI-S/V pairwise quality for impostor pairs.
        tau: decision threshold.
        q_range: quality thresholds (default: 0..100).

    Returns:
        Dict mapping strategy name to ERC result.
    """
    if q_range is None:
        q_range = np.arange(0, 101, dtype=float)

    strategies = ["s_only", "v_only", "union", "intersection"]
    results = {}

    for strategy in strategies:
        results[strategy] = _compute_erc_with_strategy(
            genuine_sim, impostor_sim,
            quality_gen_s, quality_gen_v,
            quality_imp_s, quality_imp_v,
            tau, strategy, q_range,
        )

    return results


def compute_combined_fnmr_reduction_summary(
    combined_erc: Dict,
    target_reject_fracs: List[float] = None,
) -> Dict:
    """Summarize FNMR reduction across strategies at target rejection fracs.

    Args:
        combined_erc: output of compute_combined_erc().
        target_reject_fracs: (default: [0.10, 0.20, 0.30]).

    Returns:
        Dict[strategy][reject_frac] -> {fnmr, fnmr_reduction_pct}.
    """
    if target_reject_fracs is None:
        target_reject_fracs = [0.10, 0.20, 0.30]

    summary = {}
    for strategy, erc in combined_erc.items():
        reject_fracs = erc["reject_fracs"]
        fnmr_values = erc["fnmr_values"]
        baseline_fnmr = fnmr_values[0] if not np.isnan(fnmr_values[0]) else 0.0

        strategy_summary = {}
        for target_rf in target_reject_fracs:
            idx = np.argmin(np.abs(reject_fracs - target_rf))
            fnmr = fnmr_values[idx]
            if baseline_fnmr > 0 and not np.isnan(fnmr):
                reduction_pct = (1.0 - fnmr / baseline_fnmr) * 100
            else:
                reduction_pct = 0.0

            strategy_summary[target_rf] = {
                "actual_reject_frac": float(reject_fracs[idx]),
                "fnmr": float(fnmr) if not np.isnan(fnmr) else None,
                "baseline_fnmr": float(baseline_fnmr),
                "fnmr_reduction_pct": float(reduction_pct),
            }

        summary[strategy] = strategy_summary

    return summary
