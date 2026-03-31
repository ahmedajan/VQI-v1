"""X1.12: Comprehensive Head-to-Head Comparison of All 56 Models.

Evaluates 28 original + 28 expanded models (7 families × 2 paradigms × 2 data sizes
× 2 score types) on validation + 5 test datasets, generates ~100 plots and a
recommendation report.

Usage:
    python scripts/x1_comprehensive_comparison.py [--score-type s|v|both] [--skip-eval]
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import (
    load_validation_data, load_test_features, load_test_pairs,
)
from vqi.evaluation.erc import (
    compute_erc, compute_pairwise_quality, find_tau_for_fnmr,
)
from vqi.evaluation.det import compute_ranked_det

# ============================================================================
# Constants
# ============================================================================

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "x1")
OUTPUT_DIR = os.path.join(DATA_DIR, "x1_comparison")

MODEL_FAMILIES = ["rf", "xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"]
REG_FAMILIES = ["rf", "xgboost", "lightgbm", "ridge", "svm", "mlp", "tabnet"]
DATA_VARIANTS = ["20K", "58K"]
PARADIGMS = ["clf", "reg"]

TEST_DATASETS = ["voxceleb1", "vctk", "cnceleb", "vpqad", "vseadc"]
PROVIDERS = ["P1_ECAPA", "P3_ECAPA2"]  # Primary providers for ERC/DET plots
ALL_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]

FAMILY_LABELS = {
    "rf": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "logreg": "LogReg", "svm": "SVM", "mlp": "MLP", "tabnet": "TabNet",
    "ridge": "Ridge",
}

FAMILY_COLORS = {
    "rf": "#1f77b4", "xgboost": "#ff7f0e", "lightgbm": "#2ca02c",
    "logreg": "#d62728", "svm": "#9467bd", "mlp": "#8c564b",
    "tabnet": "#e377c2", "ridge": "#d62728",
}

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Loading
# ============================================================================

def _model_path(family, paradigm, data_variant, score_type):
    """Get model file path for any of the 56 models."""
    prefix = "vqi_v" if score_type == "v" else "vqi"
    exp = "_exp" if data_variant == "58K" else ""
    reg = "_reg" if paradigm == "reg" else ""

    # Determine name for the family in the regression context
    if paradigm == "reg":
        fam = family  # ridge stays as ridge
    else:
        fam = family
        if family == "ridge":
            return None  # ridge is regression-only

    base = f"{prefix}{exp}{reg}_{fam}_model"
    ext_map = {
        "rf": "joblib", "xgboost": "json", "lightgbm": "joblib",
        "logreg": "joblib", "svm": "joblib", "mlp": "pt",
        "tabnet": "zip", "ridge": "joblib",
    }
    ext = ext_map.get(fam, "joblib")
    path = os.path.join(MODELS_DIR, f"{base}.{ext}")

    # Handle TabNet double-zip bug for expanded models
    if fam == "tabnet" and not os.path.exists(path):
        alt = path + ".zip"
        if os.path.exists(alt):
            path = alt

    return path if os.path.exists(path) else None


def _scaler_path(data_variant, score_type):
    """Get feature scaler path."""
    suffix = "_v" if score_type == "v" else ""
    if data_variant == "58K":
        return os.path.join(MODELS_DIR, f"x1_exp_feature_scaler{suffix}.joblib")
    else:
        return os.path.join(MODELS_DIR, f"x1{suffix}_feature_scaler.joblib")


def _needs_scaling(family):
    """Check if model family needs scaled features."""
    return family in ("logreg", "svm", "mlp", "tabnet", "ridge")


def load_model(family, paradigm, data_variant, score_type):
    """Load any model. Returns (model, load_type) or (None, None)."""
    path = _model_path(family, paradigm, data_variant, score_type)
    if path is None:
        return None, None

    fam = family
    if fam in ("rf", "lightgbm", "logreg", "svm", "ridge"):
        return joblib.load(path), "sklearn"
    elif fam == "xgboost":
        if paradigm == "reg":
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(path)
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(path)
        return m, "xgboost"
    elif fam == "mlp":
        import torch
        config_path = path.replace(".pt", "_config.yaml")
        if not os.path.exists(config_path):
            # Original models: vqi_mlp_config.yaml (no "_model")
            config_path = path.replace("_model.pt", "_config.yaml")
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        import torch.nn as tnn

        class _MLP(tnn.Module):
            def __init__(self, input_dim, hidden_layers, dropout):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h in hidden_layers:
                    layers.extend([
                        tnn.Linear(prev_dim, h),
                        tnn.ReLU(),
                        tnn.BatchNorm1d(h),
                        tnn.Dropout(dropout),
                    ])
                    prev_dim = h
                layers.append(tnn.Linear(prev_dim, 1))
                self.network = tnn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        model = _MLP(
            cfg["input_dim"], cfg["hidden_layers"], cfg.get("dropout", 0.3)
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model, "pytorch"
    elif fam == "tabnet":
        if paradigm == "reg":
            from pytorch_tabnet.tab_model import TabNetRegressor
            m = TabNetRegressor()
        else:
            from pytorch_tabnet.tab_model import TabNetClassifier
            m = TabNetClassifier()
        # load_model expects the actual file path
        m.load_model(path)
        return m, "tabnet"

    return None, None


def predict_scores(model, load_type, paradigm, X):
    """Predict VQI scores [0-100] for any model."""
    if model is None:
        return None

    if paradigm == "reg":
        # Regression: predict continuous [0,1], scale to [0,100]
        if load_type == "sklearn":
            raw = model.predict(X)
        elif load_type == "xgboost":
            raw = model.predict(X)
        elif load_type == "pytorch":
            import torch
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                raw = model(t).squeeze().numpy()
        elif load_type == "tabnet":
            raw = model.predict(X).flatten()
        else:
            return None
        probas = np.clip(raw, 0.0, 1.0)
    else:
        # Classification: predict_proba -> P(class=1)
        if load_type == "sklearn":
            probas = model.predict_proba(X)[:, 1]
        elif load_type == "xgboost":
            probas = model.predict_proba(X)[:, 1]
        elif load_type == "pytorch":
            import torch
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                logits = model(t).squeeze()
                probas = torch.sigmoid(logits).numpy()
        elif load_type == "tabnet":
            probas = model.predict_proba(X)[:, 1]
        else:
            return None

    scores = np.clip(np.round(probas * 100), 0, 100).astype(int)
    return scores


# ============================================================================
# Model ID Helpers
# ============================================================================

def model_id(family, paradigm, data_variant):
    """Short identifier for a model configuration."""
    fam_label = FAMILY_LABELS.get(family, family)
    par = "Clf" if paradigm == "clf" else "Reg"
    return f"{fam_label} ({par}, {data_variant})"


def all_model_configs(paradigm_filter=None):
    """Yield (family, paradigm, data_variant) for all 56 models."""
    for dv in DATA_VARIANTS:
        for fam in MODEL_FAMILIES:
            if paradigm_filter is None or paradigm_filter == "clf":
                yield fam, "clf", dv
        reg_fams = REG_FAMILIES
        for fam in reg_fams:
            if paradigm_filter is None or paradigm_filter == "reg":
                yield fam, "reg", dv


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_all_models(score_type):
    """Evaluate all 56 models on validation + test sets."""
    logger.info("Evaluating all models for VQI-%s...", score_type.upper())

    X_val, y_val = load_validation_data(score_type)
    logger.info("  Validation: %d samples", len(y_val))

    results = []
    all_val_scores = {}
    all_test_scores = {}

    configs = list(all_model_configs())
    total = len(configs)

    for i, (fam, par, dv) in enumerate(configs):
        mid = model_id(fam, par, dv)
        logger.info("  [%d/%d] %s", i + 1, total, mid)

        model, load_type = load_model(fam, par, dv, score_type)
        if model is None:
            logger.warning("    SKIP (model not found)")
            continue

        # Apply scaler if needed
        scaler = None
        if _needs_scaling(fam):
            sp = _scaler_path(dv, score_type)
            if os.path.exists(sp):
                scaler = joblib.load(sp)

        X_val_use = scaler.transform(X_val) if scaler else X_val

        # Validation predictions
        val_scores = predict_scores(model, load_type, par, X_val_use)
        if val_scores is None:
            continue
        all_val_scores[mid] = val_scores
        val_probas = val_scores / 100.0

        # Classification metrics
        preds = (val_probas >= 0.5).astype(int)
        row = {
            "family": fam, "paradigm": par, "data_variant": dv,
            "model_id": mid, "score_type": score_type,
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "auc_roc": roc_auc_score(y_val, val_probas),
            "brier": brier_score_loss(y_val, val_probas),
        }

        # Inference speed
        t0 = time.perf_counter()
        for _ in range(100):
            predict_scores(model, load_type, par, X_val_use[:10])
        elapsed = (time.perf_counter() - t0) / 100 / 10 * 1000
        row["ms_per_sample"] = elapsed

        # Score spread
        row["score_min"] = int(val_scores.min())
        row["score_max"] = int(val_scores.max())
        row["score_mean"] = float(val_scores.mean())
        row["score_std"] = float(val_scores.std())

        # Test set ERC/DET
        for ds in TEST_DATASETS:
            X_test = load_test_features(score_type, ds)
            if X_test is None:
                continue
            # Replace NaN/Inf with 0 (extraction failures)
            nan_mask = ~np.isfinite(X_test)
            if nan_mask.any():
                X_test = np.where(nan_mask, 0.0, X_test)
            X_test_use = scaler.transform(X_test) if scaler else X_test
            test_scores = predict_scores(model, load_type, par, X_test_use)
            if test_scores is None:
                continue

            test_key = f"{mid}|{ds}"
            all_test_scores[test_key] = test_scores

            pairs_data = load_test_pairs(ds)
            if pairs_data is None:
                continue
            pairs, pair_labels = pairs_data[0], pairs_data[1]

            for prov in PROVIDERS:
                pair_sims = _load_pair_sims(ds, prov)
                if pair_sims is None:
                    continue

                # ERC
                q_gen = compute_pairwise_quality(test_scores, pairs[pair_labels == 1])
                q_imp = compute_pairwise_quality(test_scores, pairs[pair_labels == 0])
                gen_sim = pair_sims[pair_labels == 1]
                imp_sim = pair_sims[pair_labels == 0]

                tau10 = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr=0.10)
                erc = compute_erc(gen_sim, imp_sim, q_gen, q_imp, tau10)
                erc20 = _erc_at_reject(erc, 0.20)
                row[f"erc20_{ds}_{prov}"] = erc20

                # DET separation
                det_result = compute_ranked_det(
                    gen_sim, imp_sim, q_gen, q_imp, percentiles=(15.0, 85.0)
                )
                row[f"det_sep_{ds}_{prov}"] = det_result.get("eer_separation", np.nan)

        results.append(row)

    df = pd.DataFrame(results)
    return df, all_val_scores, all_test_scores, y_val


def _load_pair_sims(dataset, provider):
    """Load pair similarity scores for a dataset/provider."""
    csv_path = os.path.join(
        DATA_DIR, "step8", "full_feature", "test_scores",
        f"pair_scores_{dataset}_{provider}.csv"
    )
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df["cos_sim_snorm"].values


def _erc_at_reject(erc, target_reject):
    """Get FNMR reduction at a target rejection fraction."""
    rfs = np.array(erc["reject_fracs"])
    fnmrs = np.array(erc["fnmr_values"])
    if len(rfs) == 0:
        return np.nan
    idx = np.argmin(np.abs(rfs - target_reject))
    baseline = fnmrs[0] if len(fnmrs) > 0 else np.nan
    if baseline == 0:
        return 0.0
    return (baseline - fnmrs[idx]) / baseline * 100


# ============================================================================
# Plotting Helpers
# ============================================================================

def _savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("    Saved: %s", os.path.basename(path))


def _prefix(score_type):
    return "s" if score_type == "s" else "v"


def _get_line_style(paradigm, data_variant):
    """Solid for clf, dashed for reg; circle for 20K, square for 58K."""
    ls = "-" if paradigm == "clf" else "--"
    marker = "o" if data_variant == "20K" else "s"
    return ls, marker


# ============================================================================
# Plot Functions (25 types)
# ============================================================================

# --- 1. ROC Overlay ---
def plot_roc_overlay(score_type, y_val, val_scores_dict, df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 9))
    # Sort by AUC
    sorted_models = df.sort_values("auc_roc", ascending=False)

    for _, row in sorted_models.iterrows():
        mid = row["model_id"]
        if mid not in val_scores_dict:
            continue
        probas = val_scores_dict[mid] / 100.0
        fpr, tpr, _ = roc_curve(y_val, probas)
        ls, mk = _get_line_style(row["paradigm"], row["data_variant"])
        color = FAMILY_COLORS.get(row["family"], "#333333")
        lw = 2.0 if row["auc_roc"] >= sorted_models["auc_roc"].iloc[4] else 0.8
        alpha = 1.0 if lw > 1 else 0.4
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                label=f"{mid}: {row['auc_roc']:.4f}" if lw > 1 else None)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — All Models (VQI-{score_type.upper()})", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_roc_overlay_all.png"))


# --- 2. Score Distribution Histograms ---
def plot_score_distributions(score_type, y_val, val_scores_dict, df, out_dir):
    # Top 8 models by AUC
    top = df.nlargest(8, "auc_roc")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 8:
            break
        ax = axes[i]
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        ax.hist(scores[y_val == 0], bins=range(0, 101, 5), alpha=0.6, color="blue",
                label="C0", density=True)
        ax.hist(scores[y_val == 1], bins=range(0, 101, 5), alpha=0.6, color="red",
                label="C1", density=True)
        ax.set_title(mid, fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score Distributions — Top 8 Models (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_score_distributions_all.png"))


# --- 3. Confusion Matrix Grid ---
def plot_confusion_matrices(score_type, y_val, val_scores_dict, df, out_dir):
    top = df.nlargest(8, "auc_roc")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 8:
            break
        ax = axes[i]
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        preds = (scores >= 50).astype(int)
        cm = confusion_matrix(y_val, preds)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(mid, fontsize=9)
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, f"{val:,}", ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(f"Confusion Matrices — Top 8 (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_confusion_matrices_all.png"))


# --- 4. Calibration Reliability Diagrams ---
def plot_calibration(score_type, y_val, val_scores_dict, df, out_dir):
    top = df.nlargest(8, "auc_roc")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

    for _, row in top.iterrows():
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        probas = scores / 100.0
        frac_pos, mean_pred = calibration_curve(y_val, probas, n_bins=10)
        ls, marker = _get_line_style(row["paradigm"], row["data_variant"])
        color = FAMILY_COLORS.get(row["family"], "#333333")
        ax.plot(mean_pred, frac_pos, marker=marker, ls=ls, color=color, label=mid, linewidth=1.5)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Calibration — Top 8 (VQI-{score_type.upper()})", fontsize=14)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_calibration_all.png"))


# --- 5. Precision-Recall Overlay ---
def plot_precision_recall(score_type, y_val, val_scores_dict, df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 8))
    top = df.nlargest(10, "auc_roc")

    for _, row in top.iterrows():
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        probas = scores / 100.0
        prec, rec, _ = precision_recall_curve(y_val, probas)
        pr_auc = auc(rec, prec)
        ls, _ = _get_line_style(row["paradigm"], row["data_variant"])
        color = FAMILY_COLORS.get(row["family"], "#333333")
        ax.plot(rec, prec, linestyle=ls, color=color, linewidth=1.5,
                label=f"{mid}: {pr_auc:.4f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall — Top 10 (VQI-{score_type.upper()})", fontsize=14)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_precision_recall_all.png"))


# --- 6. ERC Curves (per dataset × provider) ---
def plot_erc_curves(score_type, test_scores_dict, df, out_dir):
    top5 = df.nlargest(5, "auc_roc")
    top_ids = top5["model_id"].tolist()

    for ds in TEST_DATASETS:
        pairs_data = load_test_pairs(ds)
        if pairs_data is None:
            continue
        pairs, pair_labels = pairs_data[0], pairs_data[1]

        for prov in PROVIDERS:
            pair_sims = _load_pair_sims(ds, prov)
            if pair_sims is None:
                continue

            gen_sim = pair_sims[pair_labels == 1]
            imp_sim = pair_sims[pair_labels == 0]

            fig, ax = plt.subplots(figsize=(9, 7))
            tau10 = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr=0.10)

            for _, row in top5.iterrows():
                mid = row["model_id"]
                test_key = f"{mid}|{ds}"
                test_scores = test_scores_dict.get(test_key)
                if test_scores is None:
                    continue

                q_gen = compute_pairwise_quality(test_scores, pairs[pair_labels == 1])
                q_imp = compute_pairwise_quality(test_scores, pairs[pair_labels == 0])
                erc = compute_erc(gen_sim, imp_sim, q_gen, q_imp, tau10)

                ls, _ = _get_line_style(row["paradigm"], row["data_variant"])
                color = FAMILY_COLORS.get(row["family"], "#333333")
                ax.plot(erc["reject_fracs"], erc["fnmr_values"],
                        linestyle=ls, color=color, linewidth=2, label=mid)

            ax.set_xlabel("Rejection Fraction", fontsize=12)
            ax.set_ylabel("FNMR", fontsize=12)
            ax.set_title(f"ERC — {ds} / {prov} (VQI-{score_type.upper()})", fontsize=13)
            ax.set_xlim(0, 0.8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            _savefig(fig, os.path.join(
                out_dir, f"{_prefix(score_type)}_erc_{ds}_{prov}_fnmr10.png"))


# --- 7. Ranked DET Curves ---
def plot_det_curves(score_type, test_scores_dict, df, out_dir):
    # Pick diverse models: RF baseline + best non-SVM from each paradigm/data combo
    # Avoid SVM-only top-3 which have extreme scores and empty DET groups
    pick_ids = []
    # Always include RF baselines (smooth scores -> good DET groups)
    for dv in ["20K", "58K"]:
        mid = model_id("rf", "clf", dv)
        if mid in df["model_id"].values:
            pick_ids.append(mid)
    # Add best non-RF by AUC that has score_std > 15 (reasonable spread)
    spread = df[df["score_std"] > 15].sort_values("auc_roc", ascending=False)
    for _, row in spread.iterrows():
        mid = row["model_id"]
        if mid not in pick_ids:
            pick_ids.append(mid)
        if len(pick_ids) >= 4:
            break
    selected = df[df["model_id"].isin(pick_ids)]

    for ds in TEST_DATASETS:
        pairs_data = load_test_pairs(ds)
        if pairs_data is None:
            continue
        pairs, pair_labels = pairs_data[0], pairs_data[1]

        for prov in PROVIDERS:
            pair_sims = _load_pair_sims(ds, prov)
            if pair_sims is None:
                continue

            gen_sim = pair_sims[pair_labels == 1]
            imp_sim = pair_sims[pair_labels == 0]

            fig, ax = plt.subplots(figsize=(8, 7))
            has_curves = False

            for _, row in selected.iterrows():
                mid = row["model_id"]
                test_key = f"{mid}|{ds}"
                test_scores = test_scores_dict.get(test_key)
                if test_scores is None:
                    continue

                q_gen = compute_pairwise_quality(test_scores, pairs[pair_labels == 1])
                q_imp = compute_pairwise_quality(test_scores, pairs[pair_labels == 0])
                det_result = compute_ranked_det(gen_sim, imp_sim, q_gen, q_imp)
                groups = det_result.get("groups", {})

                color = FAMILY_COLORS.get(row["family"], "#333333")
                for grp, style in [("bottom", ":"), ("middle", "--"), ("top", "-")]:
                    gdata = groups.get(grp, {})
                    det_inner = gdata.get("det") or {}
                    fmr = det_inner.get("fmr", [])
                    fnmr = det_inner.get("fnmr", [])
                    if len(fmr) > 0 and len(fnmr) > 0:
                        ax.plot(np.array(fmr) * 100, np.array(fnmr) * 100,
                                linestyle=style, color=color, linewidth=1.5,
                                label=f"{mid} ({grp})" if grp == "top" else None)
                        has_curves = True

            ax.set_xlabel("FMR (%)", fontsize=12)
            ax.set_ylabel("FNMR (%)", fontsize=12)
            ax.set_title(f"Ranked DET — {ds} / {prov} (VQI-{score_type.upper()})", fontsize=13)
            if has_curves:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            _savefig(fig, os.path.join(
                out_dir, f"{_prefix(score_type)}_det_{ds}_{prov}.png"))


# --- 8. CDF Shift (best model) ---
def plot_cdf_shift(score_type, test_scores_dict, df, out_dir):
    best = df.loc[df["auc_roc"].idxmax()]
    mid = best["model_id"]

    for ds in TEST_DATASETS:
        test_key = f"{mid}|{ds}"
        test_scores = test_scores_dict.get(test_key)
        if test_scores is None:
            continue

        for prov in PROVIDERS:
            pair_sims = _load_pair_sims(ds, prov)
            if pair_sims is None:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            # Split into quartiles
            q25, q50, q75 = np.percentile(test_scores, [25, 50, 75])
            quartiles = [
                ("Q1 (0-25)", test_scores <= q25),
                ("Q2 (25-50)", (test_scores > q25) & (test_scores <= q50)),
                ("Q3 (50-75)", (test_scores > q50) & (test_scores <= q75)),
                ("Q4 (75-100)", test_scores > q75),
            ]

            # We need per-file sims, but we have pair-level sims
            # Use all sims associated with files in each quartile
            # For simplicity, plot the score distribution per quartile
            colors_q = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
            for (qlabel, qmask), qcolor in zip(quartiles, colors_q):
                q_scores = test_scores[qmask]
                sorted_s = np.sort(q_scores)
                cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
                ax.plot(sorted_s, cdf, color=qcolor, linewidth=2, label=qlabel)

            ax.set_xlabel("VQI Score", fontsize=12)
            ax.set_ylabel("CDF", fontsize=12)
            ax.set_title(f"CDF Shift — {mid}\n{ds} / {prov}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            _savefig(fig, os.path.join(
                out_dir, f"{_prefix(score_type)}_cdf_{ds}_{prov}.png"))


# --- 9. Score-vs-EER Quadrant Analysis ---
def plot_quadrant_analysis(score_type, test_scores_dict, df, out_dir):
    best = df.loc[df["auc_roc"].idxmax()]
    mid = best["model_id"]

    for ds in TEST_DATASETS:
        test_key = f"{mid}|{ds}"
        test_scores = test_scores_dict.get(test_key)
        if test_scores is None:
            continue

        for prov in PROVIDERS:
            col = f"det_sep_{ds}_{prov}"
            if col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot all models: x=mean VQI score, y=ERC@20%
            erc_col = f"erc20_{ds}_{prov}"
            if erc_col not in df.columns:
                plt.close(fig)
                continue

            for _, row in df.iterrows():
                rmid = row["model_id"]
                erc_val = row.get(erc_col, np.nan)
                score_mean = row.get("score_mean", np.nan)
                if np.isnan(erc_val) or np.isnan(score_mean):
                    continue
                ls, mk = _get_line_style(row["paradigm"], row["data_variant"])
                color = FAMILY_COLORS.get(row["family"], "#333333")
                ax.scatter(score_mean, erc_val, color=color, marker=mk,
                           s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

            ax.set_xlabel("Mean VQI Score", fontsize=12)
            ax.set_ylabel("ERC@20% (%)", fontsize=12)
            ax.set_title(f"Score vs ERC — {ds} / {prov} (VQI-{score_type.upper()})",
                         fontsize=12)
            ax.grid(True, alpha=0.3)
            _savefig(fig, os.path.join(
                out_dir, f"{_prefix(score_type)}_quadrant_{ds}_{prov}.png"))


# --- 10. Cross-dataset Summary Bars ---
def plot_cross_dataset_bars(score_type, df, out_dir):
    erc_cols = [c for c in df.columns if c.startswith("erc20_")]
    if not erc_cols:
        return

    # Mean ERC across all dataset-provider combos
    df_plot = df.copy()
    df_plot["mean_erc20"] = df_plot[erc_cols].mean(axis=1)
    df_plot = df_plot.sort_values("mean_erc20", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(df_plot))
    colors = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in df_plot.iterrows()]
    ax.barh(y_pos, df_plot["mean_erc20"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["model_id"], fontsize=8)
    ax.set_xlabel("Mean ERC@20% (%)", fontsize=12)
    ax.set_title(f"Cross-Dataset ERC@20% — Top 15 (VQI-{score_type.upper()})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_cross_dataset_erc.png"))


# --- 11. Feature Importance Comparison ---
def plot_feature_importance(score_type, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    families = ["rf", "xgboost", "lightgbm"]
    titles = ["RF (Gini)", "XGBoost (Gain)", "LightGBM (Gain)"]

    for ax, fam, title in zip(axes, families, titles):
        for dv, ls in [("20K", "-"), ("58K", "--")]:
            model, load_type = load_model(fam, "clf", dv, score_type)
            if model is None:
                continue
            if fam == "rf":
                imp = model.feature_importances_
            elif fam == "xgboost":
                imp = model.feature_importances_
            elif fam == "lightgbm":
                imp = model.feature_importances_
            else:
                continue

            top_idx = np.argsort(imp)[-20:]
            ax.barh(range(20), imp[top_idx], alpha=0.6,
                    label=f"{dv}", edgecolor="black", linewidth=0.3)

        ax.set_title(f"{title}", fontsize=12)
        ax.set_xlabel("Importance")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Feature Importance — Original vs Expanded (VQI-{score_type.upper()})",
                 fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_feature_importance_all.png"))


# --- 12. OOB Convergence (RF) ---
def plot_oob_convergence(score_type, df, out_dir):
    """Bar chart comparing all RF variants: OOB (if available) + validation AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rf_rows = df[df["family"] == "rf"].copy()
    if rf_rows.empty:
        plt.close(fig)
        return

    # Left: Validation AUC for all RF variants
    ax = axes[0]
    rf_sorted = rf_rows.sort_values("auc_roc", ascending=True)
    colors = ["#1f77b4" if r["data_variant"] == "20K" else "#ff7f0e" for _, r in rf_sorted.iterrows()]
    bars = ax.barh(range(len(rf_sorted)), rf_sorted["auc_roc"],
                    color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(rf_sorted)))
    ax.set_yticklabels(rf_sorted["model_id"], fontsize=9)
    for bar, auc_val in zip(bars, rf_sorted["auc_roc"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{auc_val:.4f}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Validation AUC-ROC", fontsize=11)
    ax.set_title("RF Validation AUC", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    # Right: OOB scores (where available) + F1
    ax2 = axes[1]
    rf_sorted2 = rf_rows.sort_values("f1", ascending=True)
    colors2 = ["#1f77b4" if r["data_variant"] == "20K" else "#ff7f0e" for _, r in rf_sorted2.iterrows()]
    bars2 = ax2.barh(range(len(rf_sorted2)), rf_sorted2["f1"],
                      color=colors2, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(rf_sorted2)))
    ax2.set_yticklabels(rf_sorted2["model_id"], fontsize=9)
    for bar, f1_val in zip(bars2, rf_sorted2["f1"]):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{f1_val:.4f}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Validation F1", fontsize=11)
    ax2.set_title("RF Validation F1", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"RF Comparison — Original vs Expanded (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_oob_convergence.png"))


# --- 13. Metrics Comparison Bars ---
def plot_metrics_bars(score_type, df, out_dir):
    metrics = ["auc_roc", "f1", "accuracy", "brier"]
    labels = ["AUC-ROC", "F1 Score", "Accuracy", "Brier Score"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    df_sorted = df.sort_values("auc_roc", ascending=False).head(14)

    for ax, metric, label in zip(axes, metrics, labels):
        colors = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in df_sorted.iterrows()]
        hatches = ["/" if r["data_variant"] == "58K" else "" for _, r in df_sorted.iterrows()]

        bars = ax.barh(range(len(df_sorted)), df_sorted[metric],
                       color=colors, edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["model_id"], fontsize=7)
        ax.set_xlabel(label, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    fig.suptitle(f"Metrics Comparison — Top 14 (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_metrics_comparison_all.png"))


# --- 14. Inference Speed Bars ---
def plot_inference_speed(score_type, df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_sorted = df.sort_values("ms_per_sample", ascending=True)

    colors = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in df_sorted.iterrows()]
    ax.barh(range(len(df_sorted)), df_sorted["ms_per_sample"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(50, color="red", linestyle="--", linewidth=2, label="50ms threshold")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["model_id"], fontsize=7)
    ax.set_xlabel("Inference Time (ms/sample)", fontsize=12)
    ax.set_title(f"Inference Speed — All Models (VQI-{score_type.upper()})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_inference_speed_all.png"))


# --- 15. Calibration Before/After ---
def plot_calibration_before_after(score_type, y_val, val_scores_dict, df, out_dir):
    """Show calibration for top models with isotonic regression applied."""
    from sklearn.isotonic import IsotonicRegression
    top = df.nlargest(6, "auc_roc")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 6:
            break
        ax = axes[i]
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        probas = scores / 100.0

        # Before
        frac_before, mean_before = calibration_curve(y_val, probas, n_bins=10)
        ax.plot(mean_before, frac_before, "b-o", label="Before", linewidth=2)

        # Isotonic calibration (fit on val set itself for illustration)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal_probas = iso.fit_transform(probas, y_val)
        frac_after, mean_after = calibration_curve(y_val, cal_probas, n_bins=10)
        ax.plot(mean_after, frac_after, "r-s", label="After", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(mid, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Calibration Before/After — Top 6 (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_calibration_before_after_all.png"))


# --- 16. Score Distributions Calibrated ---
def plot_score_distributions_calibrated(score_type, y_val, val_scores_dict, df, out_dir):
    from sklearn.isotonic import IsotonicRegression
    top = df.nlargest(8, "auc_roc")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 8:
            break
        ax = axes[i]
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        probas = scores / 100.0
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal_scores = np.clip(np.round(iso.fit_transform(probas, y_val) * 100), 0, 100)

        ax.hist(cal_scores[y_val == 0], bins=range(0, 101, 5), alpha=0.6, color="blue",
                label="C0", density=True)
        ax.hist(cal_scores[y_val == 1], bins=range(0, 101, 5), alpha=0.6, color="red",
                label="C1", density=True)
        ax.set_title(f"{mid} (cal)", fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Calibrated Score Distributions — Top 8 (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(
        out_dir, f"{_prefix(score_type)}_score_distributions_calibrated_all.png"))


# --- 17. AUC Scatter: Original vs Expanded ---
def plot_auc_scatter(score_type, df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 8))

    families_used = set()
    for fam in MODEL_FAMILIES + ["ridge"]:
        for par in PARADIGMS:
            orig = df[(df["family"] == fam) & (df["paradigm"] == par) &
                      (df["data_variant"] == "20K")]
            exp = df[(df["family"] == fam) & (df["paradigm"] == par) &
                     (df["data_variant"] == "58K")]
            if orig.empty or exp.empty:
                continue

            x = orig.iloc[0]["auc_roc"]
            y = exp.iloc[0]["auc_roc"]
            _, mk = _get_line_style(par, "20K")
            color = FAMILY_COLORS.get(fam, "#333333")
            ax.scatter(x, y, color=color, marker=mk, s=120, edgecolors="black",
                       linewidth=1, zorder=5)
            ax.annotate(f"{FAMILY_LABELS.get(fam, fam)} ({par})",
                        (x, y), fontsize=7, textcoords="offset points",
                        xytext=(5, 5))
            families_used.add(fam)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("AUC-ROC (Original 20K)", fontsize=12)
    ax.set_ylabel("AUC-ROC (Expanded 58K)", fontsize=12)
    ax.set_title(f"Data Expansion Effect on AUC (VQI-{score_type.upper()})", fontsize=14)
    ax.grid(True, alpha=0.3)
    # Legend: circle=clf, square=reg
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Classifier"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, label="Regressor"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_auc_scatter_orig_vs_exp.png"))


# --- 18. MSE Scatter: Original vs Expanded (regressors only) ---
def plot_mse_scatter(score_type, df, y_val, val_scores_dict, out_dir):
    fig, ax = plt.subplots(figsize=(9, 8))

    reg_df = df[df["paradigm"] == "reg"]
    for fam in REG_FAMILIES:
        orig = reg_df[(reg_df["family"] == fam) & (reg_df["data_variant"] == "20K")]
        exp = reg_df[(reg_df["family"] == fam) & (reg_df["data_variant"] == "58K")]
        if orig.empty or exp.empty:
            continue

        x = orig.iloc[0]["brier"]
        y = exp.iloc[0]["brier"]
        color = FAMILY_COLORS.get(fam, "#333333")
        ax.scatter(x, y, color=color, marker="s", s=120, edgecolors="black",
                   linewidth=1, zorder=5)
        ax.annotate(FAMILY_LABELS.get(fam, fam), (x, y), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Brier Score (Original 20K)", fontsize=12)
    ax.set_ylabel("Brier Score (Expanded 58K)", fontsize=12)
    ax.set_title(f"Data Expansion Effect on Regressors (VQI-{score_type.upper()})", fontsize=14)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_mse_scatter_orig_vs_exp.png"))


# --- 19. Heatmap: Models × Datasets ---
def plot_erc_heatmap(score_type, df, out_dir):
    erc_cols = [c for c in df.columns if c.startswith("erc20_")]
    if not erc_cols:
        return

    # Rows = models sorted by mean ERC, cols = dataset_provider
    df_plot = df.copy()
    df_plot["mean_erc"] = df_plot[erc_cols].mean(axis=1)
    df_plot = df_plot.sort_values("mean_erc", ascending=False)

    heatmap_data = df_plot[erc_cols].values
    row_labels = df_plot["model_id"].values
    col_labels = [c.replace("erc20_", "") for c in erc_cols]

    fig, ax = plt.subplots(figsize=(14, max(12, len(df_plot) * 0.35)))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6)

    plt.colorbar(im, ax=ax, label="ERC@20% (%)")
    ax.set_title(f"ERC@20% Heatmap — All Models (VQI-{score_type.upper()})", fontsize=14)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_erc_heatmap.png"))


# --- 20. Classification vs Regression ERC ---
def plot_clf_vs_reg_erc(score_type, test_scores_dict, df, out_dir):
    for ds in TEST_DATASETS:
        pairs_data = load_test_pairs(ds)
        if pairs_data is None:
            continue
        pairs, pair_labels = pairs_data[0], pairs_data[1]

        for prov in PROVIDERS:
            pair_sims = _load_pair_sims(ds, prov)
            if pair_sims is None:
                continue

            gen_sim = pair_sims[pair_labels == 1]
            imp_sim = pair_sims[pair_labels == 0]
            tau10 = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr=0.10)

            fig, ax = plt.subplots(figsize=(9, 7))

            # For each family, plot clf vs reg (best data variant)
            for fam in MODEL_FAMILIES:
                for par, ls in [("clf", "-"), ("reg", "--")]:
                    # Pick best data variant
                    for dv in ["58K", "20K"]:
                        mid = model_id(fam if par == "clf" else
                                       (fam if fam != "logreg" else "ridge"),
                                       par, dv)
                        test_key = f"{mid}|{ds}"
                        test_scores = test_scores_dict.get(test_key)
                        if test_scores is not None:
                            break
                    else:
                        continue

                    q_gen = compute_pairwise_quality(
                        test_scores, pairs[pair_labels == 1])
                    q_imp = compute_pairwise_quality(
                        test_scores, pairs[pair_labels == 0])
                    erc = compute_erc(gen_sim, imp_sim, q_gen, q_imp, tau10)
                    color = FAMILY_COLORS.get(fam, "#333333")
                    ax.plot(erc["reject_fracs"], erc["fnmr_values"],
                            linestyle=ls, color=color, linewidth=1.5,
                            label=f"{FAMILY_LABELS[fam]} ({par})")

            ax.set_xlabel("Rejection Fraction", fontsize=12)
            ax.set_ylabel("FNMR", fontsize=12)
            ax.set_title(f"Clf vs Reg ERC — {ds} / {prov} (VQI-{score_type.upper()})",
                         fontsize=12)
            ax.set_xlim(0, 0.8)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            _savefig(fig, os.path.join(
                out_dir, f"{_prefix(score_type)}_clf_vs_reg_erc_{ds}_{prov}.png"))


# --- 21. Radar/Spider Chart ---
def plot_radar_chart(score_type, df, out_dir):
    top4 = df.nlargest(4, "auc_roc")
    erc_cols = [c for c in df.columns if c.startswith("erc20_")]

    # Metrics for radar: AUC, F1, 1-Brier, mean_ERC, 1/speed, score_std
    categories = ["AUC-ROC", "F1", "1-Brier", "Mean ERC@20%", "Speed (1/ms)", "Score Spread"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for _, row in top4.iterrows():
        mid = row["model_id"]
        mean_erc = row[erc_cols].mean() if erc_cols else 0

        values = [
            row["auc_roc"],
            row["f1"],
            1 - row["brier"],
            max(mean_erc / 50, 0),  # Normalize to ~[0,1]
            min(1.0 / max(row["ms_per_sample"], 0.01), 1.0),
            min(row["score_std"] / 30, 1.0),
        ]
        values += values[:1]

        color = FAMILY_COLORS.get(row["family"], "#333333")
        ax.plot(angles, values, color=color, linewidth=2, label=mid)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(f"Radar — Top 4 (VQI-{score_type.upper()})", fontsize=14, pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_radar_top4.png"))


# --- 22. Training Size Effect ---
def plot_training_size_effect(score_type, df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    for fam in MODEL_FAMILIES + ["ridge"]:
        for par in PARADIGMS:
            subset = df[(df["family"] == fam) & (df["paradigm"] == par)]
            if len(subset) < 2:
                continue
            orig = subset[subset["data_variant"] == "20K"]
            exp = subset[subset["data_variant"] == "58K"]
            if orig.empty or exp.empty:
                continue

            x = [20288, 58102]
            y = [orig.iloc[0]["auc_roc"], exp.iloc[0]["auc_roc"]]
            ls = "-" if par == "clf" else "--"
            color = FAMILY_COLORS.get(fam, "#333333")
            ax.plot(x, y, linestyle=ls, color=color, marker="o", linewidth=2,
                    label=f"{FAMILY_LABELS.get(fam, fam)} ({par})")

    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(f"Training Size Effect (VQI-{score_type.upper()})", fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_training_size_effect.png"))


# --- 23. DET Separation Ratio Bars ---
def plot_det_separation_bars(score_type, df, out_dir):
    det_cols = [c for c in df.columns if c.startswith("det_sep_")]
    if not det_cols:
        return

    df_plot = df.copy()
    df_plot["mean_det_sep"] = df_plot[det_cols].mean(axis=1)
    df_plot = df_plot.sort_values("mean_det_sep", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in df_plot.iterrows()]
    ax.barh(range(len(df_plot)), df_plot["mean_det_sep"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["model_id"], fontsize=8)
    ax.set_xlabel("Mean DET Separation Ratio", fontsize=12)
    ax.set_title(f"DET Separation — Top 15 (VQI-{score_type.upper()})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_det_separation_bars.png"))


# --- 24. Score Spread Boxplot ---
def plot_score_spread(score_type, y_val, val_scores_dict, df, out_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    df_sorted = df.sort_values("auc_roc", ascending=False)

    data_list = []
    labels = []
    colors = []

    for _, row in df_sorted.iterrows():
        mid = row["model_id"]
        scores = val_scores_dict.get(mid)
        if scores is None:
            continue
        data_list.append(scores)
        labels.append(mid)
        colors.append(FAMILY_COLORS.get(row["family"], "#333333"))

    bp = ax.boxplot(data_list, vert=False, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("VQI Score", fontsize=12)
    ax.set_title(f"Score Spread — All Models (VQI-{score_type.upper()})", fontsize=14)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis="x")
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_score_spread_boxplot.png"))


# --- 25. Final Recommendation Dashboard ---
def plot_recommendation_dashboard(score_type, df, out_dir):
    erc_cols = [c for c in df.columns if c.startswith("erc20_")]
    df_plot = df.copy()
    if erc_cols:
        df_plot["mean_erc20"] = df_plot[erc_cols].mean(axis=1)
    else:
        df_plot["mean_erc20"] = 0

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: AUC-ROC ranking
    ax = axes[0, 0]
    top10 = df_plot.nlargest(10, "auc_roc")
    colors = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in top10.iterrows()]
    ax.barh(range(10), top10["auc_roc"], color=colors, edgecolor="black")
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10["model_id"], fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Top 10 by AUC-ROC")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Mean ERC@20%
    ax = axes[0, 1]
    top10_erc = df_plot.nlargest(10, "mean_erc20")
    colors_erc = [FAMILY_COLORS.get(r["family"], "#333333") for _, r in top10_erc.iterrows()]
    ax.barh(range(10), top10_erc["mean_erc20"], color=colors_erc, edgecolor="black")
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_erc["model_id"], fontsize=8)
    ax.set_xlabel("Mean ERC@20% (%)")
    ax.set_title("Top 10 by ERC@20%")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 3: Score quality (std vs AUC)
    ax = axes[1, 0]
    for _, row in df_plot.iterrows():
        _, mk = _get_line_style(row["paradigm"], row["data_variant"])
        color = FAMILY_COLORS.get(row["family"], "#333333")
        ax.scatter(row["auc_roc"], row["score_std"], color=color, marker=mk,
                   s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("AUC-ROC")
    ax.set_ylabel("Score Std Dev")
    ax.set_title("AUC vs Score Diversity")
    ax.grid(True, alpha=0.3)

    # Panel 4: Speed vs AUC
    ax = axes[1, 1]
    for _, row in df_plot.iterrows():
        _, mk = _get_line_style(row["paradigm"], row["data_variant"])
        color = FAMILY_COLORS.get(row["family"], "#333333")
        ax.scatter(row["ms_per_sample"], row["auc_roc"], color=color, marker=mk,
                   s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50ms")
    ax.set_xlabel("Inference (ms/sample)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Speed vs Accuracy Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"RECOMMENDATION DASHBOARD — VQI-{score_type.upper()}", fontsize=16, y=1.02)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"{_prefix(score_type)}_recommendation_dashboard.png"))


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(score_type, df, out_dir):
    """Generate markdown recommendation report."""
    erc_cols = [c for c in df.columns if c.startswith("erc20_")]
    df_r = df.copy()
    if erc_cols:
        df_r["mean_erc20"] = df_r[erc_cols].mean(axis=1)
    else:
        df_r["mean_erc20"] = 0

    # Find RF baseline
    rf_clf_20k = df_r[(df_r["family"] == "rf") & (df_r["paradigm"] == "clf") &
                       (df_r["data_variant"] == "20K")]
    rf_auc = rf_clf_20k.iloc[0]["auc_roc"] if not rf_clf_20k.empty else 0
    rf_erc = rf_clf_20k.iloc[0]["mean_erc20"] if not rf_clf_20k.empty else 0

    # Selection criteria
    df_r["passes_auc"] = df_r["auc_roc"] >= rf_auc
    df_r["passes_speed"] = df_r["ms_per_sample"] < 50
    # ERC non-negative on all datasets
    for c in erc_cols:
        df_r[f"pos_{c}"] = df_r[c] >= 0
    erc_pos_cols = [f"pos_{c}" for c in erc_cols]
    if erc_pos_cols:
        df_r["passes_erc"] = df_r[erc_pos_cols].all(axis=1)
    else:
        df_r["passes_erc"] = True

    df_r["passes_all"] = df_r["passes_auc"] & df_r["passes_speed"] & df_r["passes_erc"]

    candidates = df_r[df_r["passes_all"]].sort_values("mean_erc20", ascending=False)

    lines = [
        f"# X1.12 Comprehensive Comparison — VQI-{score_type.upper()}",
        "",
        f"**Total models evaluated:** {len(df_r)}",
        f"**RF baseline AUC:** {rf_auc:.4f}",
        f"**RF baseline ERC@20%:** {rf_erc:.1f}%",
        "",
        "## Top 10 by AUC-ROC",
        "",
        "| Rank | Model | AUC-ROC | F1 | Brier | ms/sample | Mean ERC@20% |",
        "|------|-------|---------|----|----|-----------|-------------|",
    ]

    for i, (_, row) in enumerate(df_r.nlargest(10, "auc_roc").iterrows()):
        lines.append(
            f"| {i+1} | {row['model_id']} | {row['auc_roc']:.4f} | "
            f"{row['f1']:.4f} | {row['brier']:.4f} | {row['ms_per_sample']:.2f} | "
            f"{row['mean_erc20']:.1f}% |"
        )

    lines.extend([
        "",
        "## Candidates Passing All Criteria",
        "",
        f"**Candidates:** {len(candidates)}",
        "",
    ])

    if not candidates.empty:
        lines.extend([
            "| Model | AUC-ROC | Mean ERC@20% | Speed (ms) |",
            "|-------|---------|-------------|-----------|",
        ])
        for _, row in candidates.head(10).iterrows():
            lines.append(
                f"| {row['model_id']} | {row['auc_roc']:.4f} | "
                f"{row['mean_erc20']:.1f}% | {row['ms_per_sample']:.2f} |"
            )

        best = candidates.iloc[0]
        lines.extend([
            "",
            f"## RECOMMENDATION: **{best['model_id']}**",
            "",
            f"- AUC-ROC: {best['auc_roc']:.4f} (RF baseline: {rf_auc:.4f})",
            f"- Mean ERC@20%: {best['mean_erc20']:.1f}% (RF baseline: {rf_erc:.1f}%)",
            f"- Inference: {best['ms_per_sample']:.2f} ms/sample",
            f"- Score range: [{best['score_min']}-{best['score_max']}]",
        ])
    else:
        lines.append("**No model passes all criteria. RF remains default.**")

    report_path = os.path.join(out_dir, f"{_prefix(score_type)}_recommendation.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("  Report: %s", report_path)

    # Save full metrics CSV
    csv_path = os.path.join(out_dir, f"{_prefix(score_type)}_all_metrics.csv")
    df_r.to_csv(csv_path, index=False)
    logger.info("  Metrics: %s", csv_path)


# ============================================================================
# Main
# ============================================================================

def run_score_type(score_type, skip_eval=False):
    """Run full comparison for one score type."""
    logger.info("=" * 70)
    logger.info("X1.12 Comprehensive Comparison — VQI-%s", score_type.upper())
    logger.info("=" * 70)

    out_dir = os.path.join(REPORT_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for cached eval results
    cache_path = os.path.join(OUTPUT_DIR, f"{_prefix(score_type)}_eval_cache.npz")
    csv_cache = os.path.join(OUTPUT_DIR, f"{_prefix(score_type)}_eval_metrics.csv")

    if skip_eval and os.path.exists(csv_cache):
        logger.info("Loading cached evaluation results...")
        df = pd.read_csv(csv_cache)
        cache = np.load(cache_path, allow_pickle=True)
        val_scores_dict = cache["val_scores"].item()
        test_scores_dict = cache["test_scores"].item()
        y_val = cache["y_val"]
    else:
        df, val_scores_dict, test_scores_dict, y_val = evaluate_all_models(score_type)

        # Cache results
        df.to_csv(csv_cache, index=False)
        np.savez(cache_path, val_scores=val_scores_dict,
                 test_scores=test_scores_dict, y_val=y_val)
        logger.info("Cached evaluation results to %s", OUTPUT_DIR)

    logger.info("Evaluated %d models", len(df))

    # Generate all 25 plot types
    logger.info("Generating plots...")

    plot_roc_overlay(score_type, y_val, val_scores_dict, df, out_dir)           # 1
    plot_score_distributions(score_type, y_val, val_scores_dict, df, out_dir)    # 2
    plot_confusion_matrices(score_type, y_val, val_scores_dict, df, out_dir)     # 3
    plot_calibration(score_type, y_val, val_scores_dict, df, out_dir)            # 4
    plot_precision_recall(score_type, y_val, val_scores_dict, df, out_dir)       # 5
    plot_erc_curves(score_type, test_scores_dict, df, out_dir)                   # 6
    plot_det_curves(score_type, test_scores_dict, df, out_dir)                   # 7
    plot_cdf_shift(score_type, test_scores_dict, df, out_dir)                    # 8
    plot_quadrant_analysis(score_type, test_scores_dict, df, out_dir)            # 9
    plot_cross_dataset_bars(score_type, df, out_dir)                             # 10
    plot_feature_importance(score_type, out_dir)                                 # 11
    plot_oob_convergence(score_type, df, out_dir)                                # 12
    plot_metrics_bars(score_type, df, out_dir)                                   # 13
    plot_inference_speed(score_type, df, out_dir)                                # 14
    plot_calibration_before_after(score_type, y_val, val_scores_dict, df, out_dir)  # 15
    plot_score_distributions_calibrated(score_type, y_val, val_scores_dict, df, out_dir)  # 16
    plot_auc_scatter(score_type, df, out_dir)                                    # 17
    plot_mse_scatter(score_type, df, y_val, val_scores_dict, out_dir)            # 18
    plot_erc_heatmap(score_type, df, out_dir)                                    # 19
    plot_clf_vs_reg_erc(score_type, test_scores_dict, df, out_dir)               # 20
    plot_radar_chart(score_type, df, out_dir)                                    # 21
    plot_training_size_effect(score_type, df, out_dir)                           # 22
    plot_det_separation_bars(score_type, df, out_dir)                            # 23
    plot_score_spread(score_type, y_val, val_scores_dict, df, out_dir)           # 24
    plot_recommendation_dashboard(score_type, df, out_dir)                       # 25

    # Generate report
    generate_report(score_type, df, out_dir)

    logger.info("VQI-%s COMPLETE — %d models, ~25 plot types", score_type.upper(), len(df))


def main():
    parser = argparse.ArgumentParser(description="X1.12 Comprehensive Comparison")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, use cached results")
    args = parser.parse_args()

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]

    for st in score_types:
        run_score_type(st, skip_eval=args.skip_eval)

    logger.info("=" * 70)
    logger.info("X1.12 COMPLETE — All comparisons done")
    logger.info("  Reports: %s", os.path.join(REPORT_DIR, "comparison"))
    logger.info("  Data: %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
