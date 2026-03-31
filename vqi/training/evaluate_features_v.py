"""
Step 5 VQI-V: Feature Evaluation and Selection for Voice Distinctiveness.

Thin wrapper around shared pipeline in evaluate_features.py with VQI-V parameters:
  - 161 candidate features
  - Spearman acceptance: |rho| > 0.2
  - N_selected_V range: [15, 120]
  - Outputs to data/step5/evaluation_v/
"""

import logging
from typing import Dict, Optional

from vqi.training.evaluate_features import (
    _load_checkpoint,
    _run_selection_pipeline,
)

logger = logging.getLogger(__name__)


def run_vqi_v_pipeline(
    features_path: str,
    names_path: str,
    training_csv: str,
    fisher_csv: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict:
    """Run the full VQI-V (Voice Distinctiveness) feature selection pipeline.

    161 candidates -> N_selected_V features.
    """
    completed = set()
    if resume and checkpoint_path:
        completed = _load_checkpoint(checkpoint_path)
        if completed:
            logger.info("Resuming VQI-V: completed stages = %s", completed)

    return _run_selection_pipeline(
        score_type="v",
        features_path=features_path,
        names_path=names_path,
        training_csv=training_csv,
        fisher_csv=fisher_csv,
        output_dir=output_dir,
        redundancy_threshold=0.95,
        importance_threshold_frac=0.005,
        n_selected_range=(15, 120),
        checkpoint_path=checkpoint_path,
        completed_stages=completed,
    )
