"""Cross-system generalization evaluation (Sub-tasks 8.3, 8.9).

VQI is trained on P1/P2/P3 scores only. This module evaluates whether
VQI quality predictions generalize to unseen providers P4 and P5.

Pass criterion: ERC curves for P4 and P5 show same monotonically
decreasing FNMR trend as P1-P3.

Design rationale: If VQI only works for training providers, it's
provider-specific, not a general quality metric. P4 tests legacy
systems (x-vector); P5 tests modern self-supervised models (WavLM).
"""

import logging
from typing import Dict, List

import numpy as np

from .erc import compute_erc, find_tau_for_fnmr, compute_fnmr_reduction_at_reject
from .det import compute_ranked_det

logger = logging.getLogger(__name__)

TRAIN_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]
TEST_PROVIDERS = ["P4_XVECTOR", "P5_WAVLM"]


def check_monotonicity(values: np.ndarray, tolerance: float = 0.02) -> bool:
    """Check if FNMR values are approximately monotonically non-increasing.

    Allows small violations up to `tolerance` fraction of the initial value.

    Args:
        values: FNMR values ordered by increasing rejection.
        tolerance: allowed fraction of initial FNMR for upward jumps.

    Returns:
        True if approximately monotonically non-increasing.
    """
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return True

    max_allowed_increase = valid[0] * tolerance
    for i in range(1, len(valid)):
        if valid[i] > valid[i - 1] + max_allowed_increase:
            return False
    return True


def evaluate_cross_system(
    provider_data: Dict[str, Dict],
    quality_genuine: np.ndarray,
    quality_impostor: np.ndarray,
    target_fnmr_levels: List[float] = None,
) -> Dict:
    """Evaluate cross-system generalization for all providers.

    Args:
        provider_data: Dict mapping provider name to:
            {genuine_sim: (N,), impostor_sim: (M,)}
        quality_genuine: (N,) pairwise quality for genuine pairs (same for all providers).
        quality_impostor: (M,) pairwise quality for impostor pairs.
        target_fnmr_levels: initial FNMR levels for ERC (default: [0.01, 0.10]).

    Returns:
        Dict with per-provider ERC results, DET results, and cross-system verdict.
    """
    if target_fnmr_levels is None:
        target_fnmr_levels = [0.01, 0.10]

    results = {}

    for provider_name, pdata in provider_data.items():
        gen_sim = pdata["genuine_sim"]
        imp_sim = pdata["impostor_sim"]

        provider_result = {"erc": {}, "det": None, "is_train": provider_name in TRAIN_PROVIDERS}

        # ERC at each FNMR level
        for target_fnmr in target_fnmr_levels:
            tau = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr)
            erc = compute_erc(gen_sim, imp_sim, quality_genuine, quality_impostor, tau)
            reductions = compute_fnmr_reduction_at_reject(erc)
            monotonic = check_monotonicity(erc["fnmr_values"])

            provider_result["erc"][f"fnmr_{int(target_fnmr * 100)}pct"] = {
                "tau": tau,
                "erc": erc,
                "reductions": reductions,
                "monotonic": monotonic,
            }

        # Ranked DET
        det_result = compute_ranked_det(gen_sim, imp_sim, quality_genuine, quality_impostor)
        provider_result["det"] = det_result

        results[provider_name] = provider_result

    # Cross-system verdict
    verdict = _compute_cross_system_verdict(results)
    results["_verdict"] = verdict

    return results


def _compute_cross_system_verdict(results: Dict) -> Dict:
    """Determine if cross-system generalization passes.

    Criteria:
      1. P4 and P5 ERC curves are monotonically decreasing.
      2. P4 and P5 show positive FNMR reduction at 20% rejection.
      3. Compare P4/P5 reduction to mean P1-P3 reduction.
    """
    train_reductions_10 = []
    test_reductions_10 = []
    test_monotonic = {}

    for pname, pres in results.items():
        if pname.startswith("_"):
            continue

        # Get FNMR=10% ERC results
        erc_key = "fnmr_10pct"
        if erc_key not in pres["erc"]:
            continue

        erc_data = pres["erc"][erc_key]
        red_20 = erc_data["reductions"].get(0.20, {}).get("fnmr_reduction_pct", 0)

        if pname in TRAIN_PROVIDERS:
            train_reductions_10.append(red_20)
        elif pname in TEST_PROVIDERS:
            test_reductions_10.append(red_20)
            test_monotonic[pname] = erc_data["monotonic"]

    # Verdict
    all_monotonic = all(test_monotonic.values()) if test_monotonic else False
    all_positive_reduction = all(r > 0 for r in test_reductions_10)

    mean_train_red = np.mean(train_reductions_10) if train_reductions_10 else 0
    mean_test_red = np.mean(test_reductions_10) if test_reductions_10 else 0

    # Pass if monotonic AND positive reduction for test providers
    passed = all_monotonic and all_positive_reduction

    return {
        "passed": passed,
        "all_monotonic": all_monotonic,
        "all_positive_reduction": all_positive_reduction,
        "test_monotonic": test_monotonic,
        "mean_train_reduction_20pct": float(mean_train_red),
        "mean_test_reduction_20pct": float(mean_test_red),
        "test_reductions_20pct": {
            pname: erc_data["reductions"].get(0.20, {}).get("fnmr_reduction_pct", 0)
            for pname, pres in results.items()
            if pname in TEST_PROVIDERS and "fnmr_10pct" in pres.get("erc", {})
            for erc_data in [pres["erc"]["fnmr_10pct"]]
        },
    }
