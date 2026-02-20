"""ERC (Error vs. Reject Curves) computation (Sub-tasks 8.1, 8.7).

Implements NIST.IR.8382-style quality-based rejection analysis.
For each quality threshold q, reject pairs where min(q1, q2) < q,
then compute FNMR on remaining pairs.

Design rationale: ERC directly answers "If I reject low-quality samples,
does recognition accuracy improve?" -- the actionable value proposition.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_pairwise_quality(
    quality_scores: np.ndarray,
    pair_indices: np.ndarray,
) -> np.ndarray:
    """Compute pairwise quality as min(q1, q2) for each pair.

    Args:
        quality_scores: (N,) quality score per sample.
        pair_indices: (M, 2) index pairs into quality_scores.

    Returns:
        (M,) pairwise quality scores.
    """
    q1 = quality_scores[pair_indices[:, 0]]
    q2 = quality_scores[pair_indices[:, 1]]
    return np.minimum(q1, q2)


def compute_fnmr_at_threshold(
    genuine_scores: np.ndarray,
    tau: float,
) -> float:
    """Compute FNMR at decision threshold tau.

    FNMR = fraction of genuine scores below tau.
    """
    if len(genuine_scores) == 0:
        return np.nan
    return float(np.mean(genuine_scores < tau))


def compute_fmr_at_threshold(
    impostor_scores: np.ndarray,
    tau: float,
) -> float:
    """Compute FMR at decision threshold tau.

    FMR = fraction of impostor scores >= tau.
    """
    if len(impostor_scores) == 0:
        return np.nan
    return float(np.mean(impostor_scores >= tau))


def find_tau_for_fnmr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_fnmr: float,
) -> float:
    """Find decision threshold tau that yields a given initial FNMR.

    Uses the genuine score distribution to find the quantile
    corresponding to target_fnmr.
    """
    # FNMR = P(genuine < tau) = target_fnmr
    # So tau = quantile(genuine, target_fnmr)
    tau = float(np.percentile(genuine_scores, target_fnmr * 100))
    return tau


def compute_erc(
    genuine_sim: np.ndarray,
    impostor_sim: np.ndarray,
    quality_genuine: np.ndarray,
    quality_impostor: np.ndarray,
    tau: float,
    q_range: Optional[np.ndarray] = None,
) -> Dict:
    """Compute Error vs. Reject Curve.

    For each quality threshold q in q_range, reject pairs where
    pairwise quality < q, then compute FNMR on remaining genuine pairs.

    Args:
        genuine_sim: (N_gen,) cosine similarity for genuine pairs.
        impostor_sim: (N_imp,) cosine similarity for impostor pairs.
        quality_genuine: (N_gen,) pairwise quality for genuine pairs
            (i.e., min(q1, q2) for each genuine pair).
        quality_impostor: (N_imp,) pairwise quality for impostor pairs.
        tau: decision threshold for verification.
        q_range: quality thresholds to sweep. Default: 0..100.

    Returns:
        Dict with keys:
            reject_fracs: (K,) fraction of total pairs rejected at each q.
            fnmr_values: (K,) FNMR at each q (after rejection).
            fmr_values: (K,) FMR at each q (after rejection).
            q_thresholds: (K,) quality thresholds used.
            n_genuine_remaining: (K,) count of remaining genuine pairs.
            n_impostor_remaining: (K,) count of remaining impostor pairs.
    """
    if q_range is None:
        q_range = np.arange(0, 101, dtype=float)

    n_total = len(genuine_sim) + len(impostor_sim)

    reject_fracs = np.zeros(len(q_range))
    fnmr_values = np.zeros(len(q_range))
    fmr_values = np.zeros(len(q_range))
    n_gen_remaining = np.zeros(len(q_range), dtype=int)
    n_imp_remaining = np.zeros(len(q_range), dtype=int)

    for i, q in enumerate(q_range):
        # Keep pairs where pairwise quality >= q
        gen_mask = quality_genuine >= q
        imp_mask = quality_impostor >= q

        n_gen = int(gen_mask.sum())
        n_imp = int(imp_mask.sum())
        n_remaining = n_gen + n_imp
        n_rejected = n_total - n_remaining

        reject_fracs[i] = n_rejected / n_total if n_total > 0 else 0.0
        n_gen_remaining[i] = n_gen
        n_imp_remaining[i] = n_imp

        if n_gen > 0:
            fnmr_values[i] = compute_fnmr_at_threshold(genuine_sim[gen_mask], tau)
        else:
            fnmr_values[i] = np.nan

        if n_imp > 0:
            fmr_values[i] = compute_fmr_at_threshold(impostor_sim[imp_mask], tau)
        else:
            fmr_values[i] = np.nan

    return {
        "reject_fracs": reject_fracs,
        "fnmr_values": fnmr_values,
        "fmr_values": fmr_values,
        "q_thresholds": q_range,
        "n_genuine_remaining": n_gen_remaining,
        "n_impostor_remaining": n_imp_remaining,
    }


def compute_random_rejection_baseline(
    genuine_sim: np.ndarray,
    tau: float,
    reject_fracs: np.ndarray,
    n_bootstrap: int = 100,
    rng_seed: int = 42,
) -> np.ndarray:
    """Compute random rejection baseline for ERC comparison.

    At each rejection fraction, randomly remove that fraction of genuine
    pairs and compute FNMR. Averaged over n_bootstrap repetitions.

    Args:
        genuine_sim: (N_gen,) genuine similarity scores.
        tau: decision threshold.
        reject_fracs: (K,) rejection fractions to evaluate.
        n_bootstrap: number of bootstrap repetitions.
        rng_seed: random seed.

    Returns:
        (K,) FNMR values for random rejection.
    """
    rng = np.random.RandomState(rng_seed)
    n_gen = len(genuine_sim)
    fnmr_random = np.zeros(len(reject_fracs))

    for i, rf in enumerate(reject_fracs):
        n_keep = max(1, int(n_gen * (1.0 - rf)))
        fnmr_samples = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_gen, size=n_keep, replace=False)
            fnmr_samples.append(compute_fnmr_at_threshold(genuine_sim[indices], tau))
        fnmr_random[i] = np.mean(fnmr_samples)

    return fnmr_random


def compute_fnmr_reduction_at_reject(
    erc_result: Dict,
    target_reject_fracs: List[float] = None,
) -> Dict:
    """Compute FNMR reduction at specific rejection fractions.

    Args:
        erc_result: output of compute_erc().
        target_reject_fracs: rejection fractions to report (default: 10%, 20%, 30%).

    Returns:
        Dict mapping reject_frac -> {fnmr, fnmr_reduction_pct, q_threshold}.
    """
    if target_reject_fracs is None:
        target_reject_fracs = [0.10, 0.20, 0.30]

    reject_fracs = erc_result["reject_fracs"]
    fnmr_values = erc_result["fnmr_values"]
    q_thresholds = erc_result["q_thresholds"]

    # Baseline FNMR (at q=0, no rejection)
    baseline_fnmr = fnmr_values[0] if not np.isnan(fnmr_values[0]) else 0.0

    results = {}
    for target_rf in target_reject_fracs:
        # Find closest rejection fraction
        idx = np.argmin(np.abs(reject_fracs - target_rf))
        actual_rf = reject_fracs[idx]
        fnmr = fnmr_values[idx]

        if baseline_fnmr > 0 and not np.isnan(fnmr):
            reduction_pct = (1.0 - fnmr / baseline_fnmr) * 100
        else:
            reduction_pct = 0.0

        results[target_rf] = {
            "actual_reject_frac": float(actual_rf),
            "fnmr": float(fnmr) if not np.isnan(fnmr) else None,
            "baseline_fnmr": float(baseline_fnmr),
            "fnmr_reduction_pct": float(reduction_pct),
            "q_threshold": float(q_thresholds[idx]),
        }

    return results
