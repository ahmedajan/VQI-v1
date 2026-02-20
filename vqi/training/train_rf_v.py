"""
Step 6 VQI-V: Random Forest Training for Voice Distinctiveness.

Thin wrapper around shared pipeline in train_rf.py with VQI-V parameters.
Sub-tasks 6.7-6.10, 6.12.
"""

import logging
from typing import Dict, Optional

from vqi.training.train_rf import _run_training_pipeline

logger = logging.getLogger(__name__)


def run_vqi_v_pipeline(
    features_npy: str,
    feature_names_json: str,
    selected_features_txt: str,
    training_csv: str,
    output_dir: str,
    model_path: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict:
    """Run VQI-V (Voice Distinctiveness) training pipeline. Sub-tasks 6.7-6.10, 6.12."""
    if checkpoint_path is None:
        checkpoint_path = __import__("os").path.join(output_dir, "_checkpoint_step6_v.yaml")

    return _run_training_pipeline(
        score_type="v",
        features_npy=features_npy,
        feature_names_json=feature_names_json,
        selected_features_txt=selected_features_txt,
        training_csv=training_csv,
        output_dir=output_dir,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
