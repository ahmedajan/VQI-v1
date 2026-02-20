"""
Step 7: VQI-V (Voice Distinctiveness) Model Validation — Thin Wrapper

Calls the shared validation pipeline from validate_model.py with score_type='v'.
"""

import logging
import os
from typing import Dict, Optional

from vqi.training.validate_model import _run_validation_pipeline

logger = logging.getLogger(__name__)


def run_vqi_v_validation(
    validation_csv: str,
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    model_path: str,
    training_dir: str,
    provider_scores_dir: str,
    thresholds_yaml: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    split_name: str = "val_set",
    resume: bool = False,
) -> Dict:
    """Run VQI-V validation pipeline. Sub-tasks 7.8-7.15."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "_checkpoint_step7_v.yaml")

    return _run_validation_pipeline(
        score_type="v",
        validation_csv=validation_csv,
        features_npy=features_npy,
        feature_names_json=feature_names_json,
        selected_features_txt=selected_features_txt,
        model_path=model_path,
        training_dir=training_dir,
        provider_scores_dir=provider_scores_dir,
        thresholds_yaml=thresholds_yaml,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        resume=resume,
    )
