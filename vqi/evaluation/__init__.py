"""VQI evaluation modules for Step 8: Evaluation of Predictive Power.

Modules:
  - erc: Error vs. Reject Curves (8.1, 8.7)
  - det: Ranked DET curves (8.2, 8.8)
  - cross_system: Cross-system generalization (8.3, 8.9)
  - combined_erc: Dual-score ERC comparison (8.10)
  - quadrant_analysis: Per-quadrant performance (8.11)
"""

from .erc import (
    compute_erc,
    compute_pairwise_quality,
    compute_fnmr_reduction_at_reject,
    compute_random_rejection_baseline,
    find_tau_for_fnmr,
)
from .det import (
    compute_det_curve,
    compute_ranked_det,
    compute_fnmr_at_fmr,
)
from .cross_system import (
    evaluate_cross_system,
    check_monotonicity,
)
from .combined_erc import (
    compute_combined_erc,
    compute_combined_fnmr_reduction_summary,
)
from .quadrant_analysis import (
    assign_quadrants,
    assign_pair_quadrants,
    compute_quadrant_performance,
)

__all__ = [
    "compute_erc",
    "compute_pairwise_quality",
    "compute_fnmr_reduction_at_reject",
    "compute_random_rejection_baseline",
    "find_tau_for_fnmr",
    "compute_det_curve",
    "compute_ranked_det",
    "compute_fnmr_at_fmr",
    "evaluate_cross_system",
    "check_monotonicity",
    "compute_combined_erc",
    "compute_combined_fnmr_reduction_summary",
    "assign_quadrants",
    "assign_pair_quadrants",
    "compute_quadrant_performance",
]
