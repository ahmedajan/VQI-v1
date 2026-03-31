"""Dimensionality Reduction Optimization for VQI v4.0.

Tests ALL DR techniques with Ridge (S, 20K) + XGBoost (V, 58K expanded):
  - Full features (no DR) — v3.0 baseline
  - PCA at 80%, 85%, 90%, 95%, 99%
  - FA (Factor Analysis, BIC-selected components)
  - ICA (Independent Component Analysis, PA-selected components)

Evaluates on validation + 5 test datasets (ERC@20%, DET separation, AUC, etc.).
Selects the best DR method and saves winning models.

Usage:
    python scripts/dr_optimization.py
"""

import logging
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import (
    load_training_data, load_validation_data, load_test_features,
    load_test_pairs, TEST_DATASETS,
)
from vqi.evaluation.erc import (
    compute_erc, compute_pairwise_quality, find_tau_for_fnmr,
)
from vqi.evaluation.det import compute_ranked_det

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "Final Model")

PROVIDERS = ["P1_ECAPA", "P3_ECAPA2"]
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DR Configurations
# ============================================================================

DR_CONFIGS = [
    # (name, dr_type, extra_params)
    ("Full (430/133)", "full", {}),
    ("PCA-80%", "pca", {"variance": 0.80}),
    ("PCA-85%", "pca", {"variance": 0.85}),
    ("PCA-90%", "pca", {"variance": 0.90}),
    ("PCA-95%", "pca", {"variance": 0.95}),
    ("PCA-99%", "pca", {"variance": 0.99}),
    ("FA-BIC", "existing", {"prefix": "fa"}),
    ("ICA-PA", "existing", {"prefix": "ica"}),
]


# ============================================================================
# Data Loading
# ============================================================================

def load_expanded_training_data_v():
    """Load expanded V training data (58K)."""
    base = os.path.join(DATA_DIR, "step6", "full_feature", "training_expanded_v")
    X = np.load(os.path.join(base, "X_train.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "y_train.npy")).astype(np.float32)
    return X, y


def load_pair_sims(dataset, provider):
    """Load pair similarity scores for a dataset/provider."""
    csv_path = os.path.join(
        DATA_DIR, "step8", "full_feature", "test_scores",
        f"pair_scores_{dataset}_{provider}.csv",
    )
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)["cos_sim_snorm"].values


# ============================================================================
# DR Transform Helpers
# ============================================================================

def build_dr_pipeline(dr_type, params, X_train, score_type):
    """Build and fit a DR pipeline. Returns (scaler, transformer, n_components).

    For 'full': returns (scaler, None, n_features).
    For 'pca': fits new StandardScaler + PCA.
    For 'existing': loads pre-fitted scaler + transformer from models/.
    """
    st = score_type  # 's' or 'v'

    if dr_type == "full":
        # Full features: just a scaler, no DR
        scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler, None, X_train.shape[1]

    elif dr_type == "pca":
        variance = params["variance"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        pca = PCA(n_components=variance, random_state=RANDOM_STATE)
        pca.fit(X_scaled)
        return scaler, pca, pca.n_components_

    elif dr_type == "existing":
        prefix = params["prefix"]  # 'fa' or 'ica'
        scaler_path = os.path.join(MODELS_DIR, f"vqi_{prefix}_scaler_{st}.joblib")
        trans_path = os.path.join(MODELS_DIR, f"vqi_{prefix}_transformer_{st}.joblib")

        if not os.path.exists(scaler_path) or not os.path.exists(trans_path):
            logger.warning("  Missing %s transformer for %s, skipping",
                           prefix.upper(), st.upper())
            return None, None, 0

        scaler = joblib.load(scaler_path)
        transformer = joblib.load(trans_path)
        # Determine n_components from transformer
        if hasattr(transformer, 'n_components_'):
            n_comp = transformer.n_components_
        elif hasattr(transformer, 'n_components'):
            n_comp = transformer.n_components
        elif hasattr(transformer, 'components_'):
            n_comp = transformer.components_.shape[0]
        else:
            n_comp = -1
        return scaler, transformer, n_comp

    else:
        raise ValueError(f"Unknown dr_type: {dr_type}")


def transform_data(X, scaler, transformer):
    """Apply scaler + optional transformer."""
    X_clean = np.where(~np.isfinite(X), 0.0, X)
    X_scaled = scaler.transform(X_clean)
    # Clean up any overflow from near-zero scale values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if transformer is not None:
        return transformer.transform(X_scaled)
    return X_scaled


# ============================================================================
# Model Training
# ============================================================================

def train_ridge(X_train, y_train):
    """Train Ridge regressor with CV alpha selection."""
    alphas = np.logspace(-3, 3, 50)
    ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge.fit(X_train, y_train.astype(np.float64))
    logger.info("    Ridge: best alpha=%.4f", ridge.alpha_)
    return ridge


def train_xgboost(X_train, y_train):
    """Train XGBoost regressor."""
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train.astype(np.float64))
    return xgb


def predict_scores(model, X):
    """Predict VQI scores [0-100]."""
    raw = model.predict(X)
    return np.clip(np.round(raw * 100), 0, 100).astype(int)


# ============================================================================
# Evaluation
# ============================================================================

def erc_at_reject(erc, target_reject):
    rfs = np.array(erc["reject_fracs"])
    fnmrs = np.array(erc["fnmr_values"])
    if len(rfs) == 0:
        return np.nan
    idx = np.argmin(np.abs(rfs - target_reject))
    baseline = fnmrs[0] if len(fnmrs) > 0 else np.nan
    if baseline == 0:
        return 0.0
    return (baseline - fnmrs[idx]) / baseline * 100


def evaluate_model(scores_val, y_val, test_score_dicts):
    """Compute validation metrics + test ERC@20% / DET separation."""
    probas = scores_val / 100.0
    preds = (probas >= 0.5).astype(int)

    row = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_val, probas),
        "brier": brier_score_loss(y_val, probas),
        "score_min": int(scores_val.min()),
        "score_max": int(scores_val.max()),
        "score_mean": float(scores_val.mean()),
        "score_std": float(scores_val.std()),
    }

    erc20_vals = []
    det_sep_vals = []

    for ds in TEST_DATASETS:
        test_scores = test_score_dicts.get(ds)
        if test_scores is None:
            continue
        pairs, pair_labels = load_test_pairs(ds)

        for prov in PROVIDERS:
            pair_sims = load_pair_sims(ds, prov)
            if pair_sims is None:
                continue

            q_gen = compute_pairwise_quality(test_scores, pairs[pair_labels == 1])
            q_imp = compute_pairwise_quality(test_scores, pairs[pair_labels == 0])
            gen_sim = pair_sims[pair_labels == 1]
            imp_sim = pair_sims[pair_labels == 0]

            tau10 = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr=0.10)
            erc = compute_erc(gen_sim, imp_sim, q_gen, q_imp, tau10)
            erc20 = erc_at_reject(erc, 0.20)
            row[f"erc20_{ds}_{prov}"] = erc20
            if not np.isnan(erc20):
                erc20_vals.append(erc20)

            det_result = compute_ranked_det(gen_sim, imp_sim, q_gen, q_imp)
            sep = det_result.get("eer_separation", np.nan)
            row[f"det_sep_{ds}_{prov}"] = sep
            if not np.isnan(sep):
                det_sep_vals.append(sep)

    row["mean_erc20"] = np.mean(erc20_vals) if erc20_vals else np.nan
    row["mean_det_sep"] = np.mean(det_sep_vals) if det_sep_vals else np.nan
    return row


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def run_dr_config(config_name, dr_type, params,
                  X_train_s, y_train_s, X_val_s, y_val_s,
                  X_train_v, y_train_v, X_val_v, y_val_v,
                  test_features_s, test_features_v):
    """Train and evaluate one DR configuration for both S and V."""
    logger.info("=" * 60)
    logger.info("DR Config: %s", config_name)
    logger.info("=" * 60)

    result = {"dr_config": config_name, "dr_type": dr_type}

    # --- VQI-S ---
    logger.info("  VQI-S: building DR pipeline...")
    scaler_s, trans_s, n_comp_s = build_dr_pipeline(dr_type, params, X_train_s, "s")
    if scaler_s is None:
        logger.warning("  VQI-S: SKIPPED (missing transformer)")
        return None

    result["s_components"] = n_comp_s

    X_train_s_dr = transform_data(X_train_s, scaler_s, trans_s)
    X_val_s_dr = transform_data(X_val_s, scaler_s, trans_s)

    logger.info("  VQI-S: %d -> %d features, training Ridge...",
                X_train_s.shape[1], X_train_s_dr.shape[1])
    model_s = train_ridge(X_train_s_dr, y_train_s)

    val_scores_s = predict_scores(model_s, X_val_s_dr)

    # Inference speed
    t0 = time.perf_counter()
    for _ in range(100):
        predict_scores(model_s, X_val_s_dr[:10])
    s_ms = (time.perf_counter() - t0) / 100 / 10 * 1000
    result["s_ms_per_sample"] = s_ms

    # Score test sets
    test_scores_s = {}
    for ds, X_test in test_features_s.items():
        X_test_dr = transform_data(X_test, scaler_s, trans_s)
        test_scores_s[ds] = predict_scores(model_s, X_test_dr)

    s_metrics = evaluate_model(val_scores_s, y_val_s, test_scores_s)
    for k, v in s_metrics.items():
        result[f"s_{k}"] = v

    # --- VQI-V ---
    logger.info("  VQI-V: building DR pipeline...")
    scaler_v, trans_v, n_comp_v = build_dr_pipeline(dr_type, params, X_train_v, "v")
    if scaler_v is None:
        logger.warning("  VQI-V: SKIPPED (missing transformer)")
        return None

    result["v_components"] = n_comp_v

    X_train_v_dr = transform_data(X_train_v, scaler_v, trans_v)
    X_val_v_dr = transform_data(X_val_v, scaler_v, trans_v)

    logger.info("  VQI-V: %d -> %d features, training XGBoost...",
                X_train_v.shape[1], X_train_v_dr.shape[1])
    model_v = train_xgboost(X_train_v_dr, y_train_v)

    val_scores_v = predict_scores(model_v, X_val_v_dr)

    # Inference speed
    t0 = time.perf_counter()
    for _ in range(100):
        predict_scores(model_v, X_val_v_dr[:10])
    v_ms = (time.perf_counter() - t0) / 100 / 10 * 1000
    result["v_ms_per_sample"] = v_ms

    # Score test sets
    test_scores_v = {}
    for ds, X_test in test_features_v.items():
        X_test_dr = transform_data(X_test, scaler_v, trans_v)
        test_scores_v[ds] = predict_scores(model_v, X_test_dr)

    v_metrics = evaluate_model(val_scores_v, y_val_v, test_scores_v)
    for k, v in v_metrics.items():
        result[f"v_{k}"] = v

    # Store objects for potential saving
    result["_objects"] = {
        "scaler_s": scaler_s, "trans_s": trans_s, "model_s": model_s,
        "scaler_v": scaler_v, "trans_v": trans_v, "model_v": model_v,
        "val_scores_s": val_scores_s, "val_scores_v": val_scores_v,
        "test_scores_s": test_scores_s, "test_scores_v": test_scores_v,
    }

    logger.info("  S: %d comp, AUC=%.4f, ERC@20%%=%.1f%%, range=[%d-%d], %.3fms",
                n_comp_s, s_metrics["auc_roc"], s_metrics.get("mean_erc20", 0),
                s_metrics["score_min"], s_metrics["score_max"], s_ms)
    logger.info("  V: %d comp, AUC=%.4f, ERC@20%%=%.1f%%, range=[%d-%d], %.3fms",
                n_comp_v, v_metrics["auc_roc"], v_metrics.get("mean_erc20", 0),
                v_metrics["score_min"], v_metrics["score_max"], v_ms)

    return result


# ============================================================================
# Comparison Plots
# ============================================================================

def plot_comparison(results_df):
    """Generate comparison plots across all DR methods."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    n_configs = len(results_df)
    x = np.arange(n_configs)
    labels = results_df["dr_config"].values
    width = 0.35

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    # 1. AUC-ROC comparison
    ax = axes[0, 0]
    ax.bar(x - width / 2, results_df["s_auc_roc"], width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_auc_roc"], width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Mean ERC@20%
    ax = axes[0, 1]
    ax.bar(x - width / 2, results_df["s_mean_erc20"].fillna(0), width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_mean_erc20"].fillna(0), width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("Mean ERC@20% (%)")
    ax.set_title("Mean ERC@20% by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Number of components
    ax = axes[1, 0]
    ax.bar(x - width / 2, results_df["s_components"], width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_components"], width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("# Components")
    ax.set_title("Dimensionality by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. F1 Score
    ax = axes[1, 1]
    ax.bar(x - width / 2, results_df["s_f1"], width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_f1"], width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Score spread (std)
    ax = axes[2, 0]
    ax.bar(x - width / 2, results_df["s_score_std"], width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_score_std"], width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("Score Std Dev")
    ax.set_title("Score Spread by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Inference speed
    ax = axes[2, 1]
    ax.bar(x - width / 2, results_df["s_ms_per_sample"], width, label="VQI-S", color="#1f77b4")
    ax.bar(x + width / 2, results_df["v_ms_per_sample"], width, label="VQI-V", color="#2ca02c")
    ax.set_xlabel("DR Method")
    ax.set_ylabel("ms/sample")
    ax.set_title("Inference Speed by DR Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("DR Optimization: All Methods Comparison\n"
                 "Ridge (VQI-S, 20K) + XGBoost (VQI-V, 58K)",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "dr_optimization_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", path)

    # Radar chart for top-4 configs
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                              subplot_kw=dict(polar=True))
    categories = ["AUC", "F1", "1-Brier", "ERC@20%", "Speed", "Spread"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax_idx, (st, st_label) in enumerate([("s", "VQI-S"), ("v", "VQI-V")]):
        ax = axes[ax_idx]
        # Top 4 by combined score
        df_sort = results_df.sort_values(
            f"{st}_mean_erc20", ascending=False, na_position="last"
        ).head(4)

        for i, (_, row) in enumerate(df_sort.iterrows()):
            values = [
                row[f"{st}_auc_roc"],
                row[f"{st}_f1"],
                1 - row[f"{st}_brier"],
                max(row.get(f"{st}_mean_erc20", 0) / 50, 0),
                min(1.0 / max(row[f"{st}_ms_per_sample"], 0.01), 1.0),
                min(row[f"{st}_score_std"] / 30, 1.0),
            ]
            values += values[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values, color=color, linewidth=2,
                    label=row["dr_config"])
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(f"{st_label} — Top 4 by ERC@20%", fontsize=12, pad=20)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "dr_optimization_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar plot: %s", path)


# ============================================================================
# Winner Selection & Model Saving
# ============================================================================

def select_winner(results_df, all_results):
    """Select the best DR configuration."""
    # Criteria: maximize combined mean ERC@20% (S + V)
    # with AUC >= RF baseline (S: 0.8719, V: 0.8812)
    RF_AUC_S = 0.8719
    RF_AUC_V = 0.8812

    df = results_df.copy()
    df["combined_erc"] = df["s_mean_erc20"].fillna(0) + df["v_mean_erc20"].fillna(0)

    # Prefer configs that pass AUC thresholds
    candidates = df[
        (df["s_auc_roc"] >= RF_AUC_S) &
        (df["v_auc_roc"] >= RF_AUC_V)
    ]

    if candidates.empty:
        logger.warning("No config passes AUC constraints. Relaxing...")
        candidates = df

    best_idx = candidates["combined_erc"].idxmax()
    best_name = candidates.loc[best_idx, "dr_config"]

    # Find matching result with objects
    for r in all_results:
        if r is not None and r["dr_config"] == best_name:
            return r

    return None


def save_winning_models(winner):
    """Save v4.0 models from the winning configuration."""
    objs = winner["_objects"]
    dr_type = winner["dr_type"]
    config_name = winner["dr_config"]

    logger.info("\nSaving v4.0 models for: %s", config_name)

    # VQI-S
    joblib.dump(objs["scaler_s"],
                os.path.join(MODELS_DIR, "vqi_v4_scaler_s.joblib"))
    if objs["trans_s"] is not None:
        joblib.dump(objs["trans_s"],
                    os.path.join(MODELS_DIR, "vqi_v4_transformer_s.joblib"))
    joblib.dump(objs["model_s"],
                os.path.join(MODELS_DIR, "vqi_v4_model_s.joblib"))

    # VQI-V
    joblib.dump(objs["scaler_v"],
                os.path.join(MODELS_DIR, "vqi_v4_scaler_v.joblib"))
    if objs["trans_v"] is not None:
        joblib.dump(objs["trans_v"],
                    os.path.join(MODELS_DIR, "vqi_v4_transformer_v.joblib"))
    objs["model_v"].save_model(
        os.path.join(MODELS_DIR, "vqi_v4_model_v.json"))

    # Save metadata
    meta = {
        "dr_config": config_name,
        "dr_type": dr_type,
        "s_components": winner["s_components"],
        "v_components": winner["v_components"],
        "s_auc_roc": winner["s_auc_roc"],
        "v_auc_roc": winner["v_auc_roc"],
        "s_mean_erc20": winner.get("s_mean_erc20", None),
        "v_mean_erc20": winner.get("v_mean_erc20", None),
        "has_transformer_s": objs["trans_s"] is not None,
        "has_transformer_v": objs["trans_v"] is not None,
    }
    import json
    with open(os.path.join(MODELS_DIR, "vqi_v4_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save val/test scores cache for downstream pipeline
    np.savez(
        os.path.join(REPORT_DIR, "v4_scores_cache.npz"),
        val_scores_s=objs["val_scores_s"],
        val_scores_v=objs["val_scores_v"],
        test_scores_s=objs["test_scores_s"],
        test_scores_v=objs["test_scores_v"],
        allow_pickle=True,
    )

    logger.info("  Saved to: %s/vqi_v4_*", MODELS_DIR)
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("vqi_v4_")]
    for f in sorted(model_files):
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        logger.info("    %s (%s)", f, _fmt_size(size))


def _fmt_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    os.makedirs(REPORT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Dimensionality Reduction Optimization for VQI v4.0")
    logger.info("  DR Methods: %d configs", len(DR_CONFIGS))
    logger.info("  Models: Ridge (S, 20K) + XGBoost (V, 58K)")
    logger.info("  Test sets: %s", ", ".join(TEST_DATASETS))
    logger.info("=" * 70)

    # --- Load all data ---
    logger.info("\nLoading training data...")
    X_train_s, y_train_s = load_training_data("s")
    logger.info("  S training: %s", X_train_s.shape)
    X_train_v, y_train_v = load_expanded_training_data_v()
    logger.info("  V training (58K expanded): %s", X_train_v.shape)

    logger.info("\nLoading validation data...")
    X_val_s, y_val_s = load_validation_data("s")
    X_val_v, y_val_v = load_validation_data("v")
    logger.info("  S val: %s, V val: %s", X_val_s.shape, X_val_v.shape)

    logger.info("\nLoading test features...")
    test_features_s = {}
    test_features_v = {}
    for ds in TEST_DATASETS:
        test_features_s[ds] = load_test_features("s", ds)
        test_features_v[ds] = load_test_features("v", ds)
        logger.info("  %s: S=%s, V=%s", ds,
                     test_features_s[ds].shape, test_features_v[ds].shape)

    # --- Run all DR configurations ---
    all_results = []
    for config_name, dr_type, params in DR_CONFIGS:
        result = run_dr_config(
            config_name, dr_type, params,
            X_train_s, y_train_s, X_val_s, y_val_s,
            X_train_v, y_train_v, X_val_v, y_val_v,
            test_features_s, test_features_v,
        )
        all_results.append(result)

    # --- Build comparison table ---
    valid_results = [r for r in all_results if r is not None]
    table_rows = []
    for r in valid_results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        table_rows.append(row)

    results_df = pd.DataFrame(table_rows)

    # --- Print summary ---
    print("\n" + "=" * 120)
    print("DR OPTIMIZATION RESULTS — Ridge (S, 20K) + XGBoost (V, 58K)")
    print("=" * 120)
    summary_cols = [
        "dr_config", "s_components", "s_auc_roc", "s_f1", "s_mean_erc20",
        "s_score_min", "s_score_max", "s_score_std", "s_ms_per_sample",
        "v_components", "v_auc_roc", "v_f1", "v_mean_erc20",
        "v_score_min", "v_score_max", "v_score_std", "v_ms_per_sample",
    ]
    existing_cols = [c for c in summary_cols if c in results_df.columns]
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    print(results_df[existing_cols].to_string(index=False, float_format="%.4f"))

    # --- Ranking ---
    print("\n" + "-" * 80)
    print("RANKING by Mean ERC@20% (S):")
    rank_s = results_df.sort_values("s_mean_erc20", ascending=False, na_position="last")
    for i, (_, row) in enumerate(rank_s.iterrows()):
        print(f"  {i+1}. {row['dr_config']:20s} — S ERC@20%={row.get('s_mean_erc20', 0):6.1f}%, "
              f"AUC={row['s_auc_roc']:.4f}, {row['s_components']} comp")

    print("\nRANKING by Mean ERC@20% (V):")
    rank_v = results_df.sort_values("v_mean_erc20", ascending=False, na_position="last")
    for i, (_, row) in enumerate(rank_v.iterrows()):
        print(f"  {i+1}. {row['dr_config']:20s} — V ERC@20%={row.get('v_mean_erc20', 0):6.1f}%, "
              f"AUC={row['v_auc_roc']:.4f}, {row['v_components']} comp")

    # --- Select winner ---
    winner = select_winner(results_df, valid_results)
    if winner is None:
        logger.error("No valid winner found!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f">>> WINNER: {winner['dr_config']}")
    print(f"    S: {winner['s_components']} components, "
          f"AUC={winner['s_auc_roc']:.4f}, "
          f"ERC@20%={winner.get('s_mean_erc20', 0):.1f}%")
    print(f"    V: {winner['v_components']} components, "
          f"AUC={winner['v_auc_roc']:.4f}, "
          f"ERC@20%={winner.get('v_mean_erc20', 0):.1f}%")
    print("=" * 80)

    # --- Save ---
    save_winning_models(winner)

    csv_path = os.path.join(REPORT_DIR, "dr_optimization_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    # --- Plot ---
    plot_comparison(results_df)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("DR OPTIMIZATION COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Winner: %s", winner["dr_config"])
    logger.info("  Models: %s/vqi_v4_*", MODELS_DIR)
    logger.info("  Reports: %s", REPORT_DIR)
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Review results and confirm winner")
    logger.info("  2. Run: python scripts/deploy_v4.py")
    logger.info("  3. Run: python scripts/run_step7_v4.py")
    logger.info("  4. Run: python scripts/run_step8_v4.py --dataset all")
    logger.info("  5. Run: python scripts/generate_conformance_output_v4.py")
    logger.info("  6. Run: python scripts/regenerate_final_model_reports.py")


if __name__ == "__main__":
    main()
