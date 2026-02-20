"""
VQI-V Score Prediction via Random Forest (Sub-task 6.11).

Loads the trained VQI-V RF model and maps a feature vector to a [0-100] score.
Score = int(round(P(Class1) * 100)), clipped to [0, 100].
"""

import logging
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> RandomForestClassifier:
    """Load a trained RF model from a joblib file."""
    clf = joblib.load(model_path)
    logger.info(
        "Loaded VQI-V RF model from %s (%d trees, %d features)",
        model_path, clf.n_estimators, clf.n_features_in_,
    )
    return clf


def get_selected_feature_names(names_path: str) -> List[str]:
    """Read selected feature names from a text file (one per line)."""
    with open(names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def predict_score(clf: RandomForestClassifier, features: np.ndarray) -> int:
    """Map a feature vector to a VQI-V score in [0, 100].

    Args:
        clf: trained RandomForestClassifier
        features: (N_selected_V,) or (1, N_selected_V) feature vector

    Returns:
        Integer score in [0, 100]. Higher = better voice distinctiveness.
    """
    x = features.reshape(1, -1).astype(np.float32)
    proba = clf.predict_proba(x)[0]
    score = int(np.round(proba[1] * 100))
    score = int(np.clip(score, 0, 100))
    return score


def predict_scores_batch(clf: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    """Map multiple feature vectors to VQI-V scores.

    Args:
        clf: trained RandomForestClassifier
        features: (N, N_selected_V) feature matrix

    Returns:
        (N,) integer array of scores in [0, 100].
    """
    x = features.astype(np.float32)
    probas = clf.predict_proba(x)[:, 1]
    scores = np.round(probas * 100).astype(int)
    scores = np.clip(scores, 0, 100)
    return scores
