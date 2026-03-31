"""VQI Engine -- single entry-point for scoring audio files and waveforms.

Loads all models and metadata once, then provides ``score_file()`` and
``score_waveform()`` methods that run the full VQI pipeline and return a
``VQIResult`` with scores, feedback, and diagnostics.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import xgboost as xgb

from .preprocessing.audio_loader import load_audio
from .preprocessing.normalize import dc_remove_and_normalize
from .preprocessing.vad import energy_vad, reconstruct_from_mask
from .core.vqi_algorithm import check_actionable_feedback
from .core.feature_orchestrator import compute_all_features
from .core.feature_orchestrator_v import compute_all_features_v
from .feedback import (
    FEEDBACK_TEMPLATES_S,
    FEEDBACK_TEMPLATES_V,
    find_limiting_factors,
    compute_category_scores,
    render_plain_feedback,
    render_expert_feedback,
    generate_overall_assessment,
)

logger = logging.getLogger(__name__)

# Default base directory (implementation/)
_DEFAULT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class VQIResult:
    """Container for all VQI scoring outputs."""
    score_s: int = 0
    score_v: int = 0
    features_s: Dict[str, float] = field(default_factory=dict)
    features_v: Dict[str, float] = field(default_factory=dict)
    limiting_factors_s: List[dict] = field(default_factory=list)
    limiting_factors_v: List[dict] = field(default_factory=list)
    overall_assessment: str = ""
    category_scores_s: Dict[str, int] = field(default_factory=dict)
    category_scores_v: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    audio_info: dict = field(default_factory=dict)
    plain_feedback_s: str = ""
    plain_feedback_v: str = ""
    expert_feedback_s: str = ""
    expert_feedback_v: str = ""
    warnings: List[str] = field(default_factory=list)
    waveform: Optional[np.ndarray] = None  # raw waveform for visualization


ProgressCallback = Optional[Callable[[int, int, str], None]]


class VQIEngine:
    """Main VQI scoring engine.

    Loads all models and metadata once in ``__init__()``, then reuses
    them across multiple ``score_file()`` / ``score_waveform()`` calls.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the engine by loading models and metadata.

        Parameters
        ----------
        base_dir : str, optional
            Path to the ``implementation/`` directory.  Defaults to the
            parent of the ``vqi/`` package directory.
        """
        self.base_dir = base_dir or _DEFAULT_BASE
        t0 = time.time()

        # --- Load v4.0 metadata ---
        meta_path = os.path.join(self.base_dir, "models", "vqi_v4_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.v4_meta = json.load(f)

        # --- Load VQI-S model: StandardScaler + Ridge Regressor (20K) ---
        self.scaler_s = joblib.load(os.path.join(self.base_dir, "models", "vqi_v4_scaler_s.joblib"))
        self.model_s = joblib.load(os.path.join(self.base_dir, "models", "vqi_v4_model_s.joblib"))

        # --- Load VQI-V model: StandardScaler + XGBoost Regressor (58K) ---
        self.scaler_v = joblib.load(os.path.join(self.base_dir, "models", "vqi_v4_scaler_v.joblib"))
        self.model_v = xgb.XGBRegressor()
        self.model_v.load_model(os.path.join(self.base_dir, "models", "vqi_v4_model_v.json"))

        # --- Load selected feature names ---
        sel_s_path = os.path.join(self.base_dir, "data", "step5", "evaluation", "selected_features.txt")
        sel_v_path = os.path.join(self.base_dir, "data", "step5", "evaluation_v", "selected_features.txt")
        with open(sel_s_path, "r", encoding="utf-8") as f:
            self.selected_s = [ln.strip() for ln in f if ln.strip()]
        with open(sel_v_path, "r", encoding="utf-8") as f:
            self.selected_v = [ln.strip() for ln in f if ln.strip()]

        # --- Load percentile tables ---
        pct_s = np.load(os.path.join(self.base_dir, "data", "step9", "feature_percentiles_s.npz"))
        self.percentiles_s = pct_s["percentiles"]          # (430, 101)
        self.percentile_names_s = list(pct_s["feature_names"])  # 430

        pct_v = np.load(os.path.join(self.base_dir, "data", "step9", "feature_percentiles_v.npz"))
        self.percentiles_v = pct_v["percentiles"]          # (133, 101)
        self.percentile_names_v = list(pct_v["feature_names"])  # 133

        # --- Load feature categories ---
        cat_path = os.path.join(self.base_dir, "data", "step9", "feature_categories.json")
        with open(cat_path, "r", encoding="utf-8") as f:
            self.categories = json.load(f)

        # --- Load feature importances ---
        self.importances_s = self._load_importances(
            os.path.join(self.base_dir, "data", "step6", "full_feature", "training", "feature_importances.csv")
        )
        self.importances_v = self._load_importances(
            os.path.join(self.base_dir, "data", "step6", "full_feature", "training_v", "feature_importances.csv")
        )

        elapsed = time.time() - t0
        logger.info("VQIEngine loaded in %.1fs (S=%d feats, V=%d feats)",
                     elapsed, len(self.selected_s), len(self.selected_v))

    @staticmethod
    def _load_importances(csv_path: str) -> Dict[str, float]:
        """Load feature importances from CSV (feature, importance, ...)."""
        imp = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imp[row["feature"]] = float(row["importance"])
        return imp

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_file(
        self,
        filepath: str,
        progress_callback: ProgressCallback = None,
    ) -> VQIResult:
        """Score an audio file through the full VQI pipeline.

        Parameters
        ----------
        filepath : str
            Path to an audio file (WAV, FLAC, etc.).
        progress_callback : callable, optional
            Called as ``progress_callback(step, total_steps, message)``
            for GUI progress updates.

        Returns
        -------
        VQIResult
            Complete scoring result with scores, feedback, and diagnostics.
        """
        t0 = time.time()

        def _progress(step, msg):
            if progress_callback:
                progress_callback(step, 4, msg)

        # Step 1: Load audio
        _progress(1, "Loading audio...")
        raw_waveform = load_audio(filepath)
        duration_s = len(raw_waveform) / 16000.0

        return self._score_waveform_internal(raw_waveform, 16000, duration_s, _progress, t0)

    def score_waveform(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        progress_callback: ProgressCallback = None,
    ) -> VQIResult:
        """Score a raw waveform (e.g. from microphone recording).

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 array.
        sr : int
            Sample rate.
        progress_callback : callable, optional
            Progress callback.

        Returns
        -------
        VQIResult
        """
        t0 = time.time()

        def _progress(step, msg):
            if progress_callback:
                progress_callback(step, 4, msg)

        _progress(1, "Preparing audio...")
        # Ensure float32 and mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        waveform = waveform.astype(np.float32)
        duration_s = len(waveform) / sr

        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio
            import torch
            t = torch.from_numpy(waveform).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, 16000)
            waveform = t.squeeze(0).numpy()

        return self._score_waveform_internal(waveform, 16000, duration_s, _progress, t0)

    def _score_waveform_internal(self, raw_waveform, sr, duration_s, _progress, t0):
        """Internal shared scoring pipeline."""
        result = VQIResult()
        result.audio_info = {
            "duration_s": duration_s,
            "sample_rate": sr,
        }

        # Store waveform for visualization
        result.waveform = raw_waveform.copy()

        # Step 2: Preprocess
        _progress(2, "Preprocessing (normalization + VAD)...")
        normalized = dc_remove_and_normalize(raw_waveform)
        vad_mask, speech_dur, speech_ratio = energy_vad(normalized, sr)
        result.audio_info["speech_duration_s"] = speech_dur
        result.audio_info["speech_ratio"] = speech_ratio

        # Actionable checks
        warnings = check_actionable_feedback(raw_waveform, vad_mask)
        result.warnings = warnings
        result.audio_info["warnings"] = warnings

        # Reconstruct speech-only waveform
        speech = reconstruct_from_mask(normalized, vad_mask)

        # Step 3: Feature extraction
        _progress(3, "Extracting features...")
        features_dict_s, features_array_s, intermediates = compute_all_features(
            speech, sr, vad_mask, raw_waveform=raw_waveform
        )
        features_dict_v, features_array_v = compute_all_features_v(
            speech, sr, vad_mask, intermediates=intermediates
        )

        # Select features for prediction
        selected_feats_s = {n: features_dict_s.get(n, 0.0) for n in self.selected_s}
        selected_feats_v = {n: features_dict_v.get(n, 0.0) for n in self.selected_v}

        # Build feature vectors in correct order
        feat_vec_s = np.array([selected_feats_s[n] for n in self.selected_s], dtype=np.float64)
        feat_vec_v = np.array([selected_feats_v[n] for n in self.selected_v], dtype=np.float64)

        # Replace NaN/Inf with 0
        feat_vec_s = np.nan_to_num(feat_vec_s, nan=0.0, posinf=0.0, neginf=0.0)
        feat_vec_v = np.nan_to_num(feat_vec_v, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Predict + feedback
        _progress(4, "Computing scores and feedback...")

        # VQI-S: StandardScaler + Ridge regressor
        feat_s_scaled = self.scaler_s.transform(feat_vec_s.reshape(1, -1))
        pred_s = self.model_s.predict(feat_s_scaled)[0]
        result.score_s = int(np.clip(np.round(pred_s * 100), 0, 100))

        # VQI-V: StandardScaler + XGBoost regressor
        feat_v_scaled = self.scaler_v.transform(feat_vec_v.reshape(1, -1))
        pred_v = self.model_v.predict(feat_v_scaled)[0]
        result.score_v = int(np.clip(np.round(pred_v * 100), 0, 100))
        result.features_s = selected_feats_s
        result.features_v = selected_feats_v

        # Limiting factors
        result.limiting_factors_s = find_limiting_factors(
            selected_feats_s, self.importances_s,
            self.percentiles_s, self.percentile_names_s,
            self.categories,
        )
        result.limiting_factors_v = find_limiting_factors(
            selected_feats_v, self.importances_v,
            self.percentiles_v, self.percentile_names_v,
            self.categories,
        )

        # Category scores
        result.category_scores_s = compute_category_scores(
            selected_feats_s, self.percentiles_s,
            self.percentile_names_s, self.categories,
        )
        result.category_scores_v = compute_category_scores(
            selected_feats_v, self.percentiles_v,
            self.percentile_names_v, self.categories,
        )

        # Overall assessment
        result.overall_assessment = generate_overall_assessment(result.score_s, result.score_v)

        # Render feedback text
        result.plain_feedback_s = render_plain_feedback(
            result.limiting_factors_s, FEEDBACK_TEMPLATES_S, result.score_s, "S"
        )
        result.plain_feedback_v = render_plain_feedback(
            result.limiting_factors_v, FEEDBACK_TEMPLATES_V, result.score_v, "V"
        )
        result.expert_feedback_s = render_expert_feedback(
            selected_feats_s, self.percentiles_s, self.percentile_names_s,
            self.categories, result.category_scores_s,
            result.limiting_factors_s, result.score_s, "S",
        )
        result.expert_feedback_v = render_expert_feedback(
            selected_feats_v, self.percentiles_v, self.percentile_names_v,
            self.categories, result.category_scores_v,
            result.limiting_factors_v, result.score_v, "V",
        )

        result.processing_time_ms = (time.time() - t0) * 1000
        return result
