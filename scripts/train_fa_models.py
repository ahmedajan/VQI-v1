"""
Factor Analysis-Reduced Random Forest Model Training & Analysis

Trains RF models on FA-reduced features and compares OOB accuracy
against full-feature baselines.

FA component count selected via BIC minimization (two-pass sweep:
coarse grid then fine refinement around minimum).

Usage:
    python train_fa_models.py
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
import yaml
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants (matching train_rf.py)
# ---------------------------------------------------------------------------

FIXED_PARAMS = dict(
    criterion="gini",
    min_samples_leaf=5,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    bootstrap=True,
)

N_ESTIMATORS_GRID = [200, 300, 400, 500, 750, 1000]
MAX_FEATURES_GRID = [5, 8, 10, 12, "sqrt"]

# BIC sweep config
BIC_SWEEP = {
    "s": {"coarse_start": 20, "coarse_stop": 420, "coarse_step": 20},
    "v": {"coarse_start": 5, "coarse_stop": 125, "coarse_step": 5},
}
BIC_FINE_HALFWIDTH = 1  # fine sweep: coarse_min +/- 1*coarse_step
BIC_FINE_STEP = {"s": 5, "v": 1}  # fine sweep step (larger for S due to slow fits)

# Original baselines
BASELINES = {
    "s": {
        "n_features": 430,
        "n_estimators": 1000,
        "max_features": 8,
        "oob_accuracy": 0.817577,
        "training_accuracy": 0.977228,
    },
    "v": {
        "n_features": 133,
        "n_estimators": 1000,
        "max_features": 5,
        "oob_accuracy": 0.820584,
        "training_accuracy": 0.978657,
    },
}

# Old PCA-90%-matched values (for reference in plots)
OLD_COMPONENTS = {"s": 99, "v": 47}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_data(score_type):
    """Load X_train, y_train from data/step6/full_feature/training/ or training_v/."""
    if score_type == "s":
        data_dir = os.path.join(PROJECT_ROOT, "data", "step6", "full_feature", "training")
    else:
        data_dir = os.path.join(PROJECT_ROOT, "data", "step6", "full_feature", "training_v")

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))
    return X, y


# ---------------------------------------------------------------------------
# BIC-based component selection
# ---------------------------------------------------------------------------

def compute_fa_bic(X_scaled, n_components, N):
    """Compute BIC for a FactorAnalysis model.

    BIC = -2 * N * mean_LL + k * log(N)
    where k = p*n - n*(n-1)//2 + p  (FA free parameters)
    """
    p = X_scaled.shape[1]
    try:
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        fa.fit(X_scaled)
        mean_ll = fa.score(X_scaled)  # mean log-likelihood per sample
    except Exception:
        return float("inf"), float("-inf")

    # Free parameters: loadings (p*n) minus rotation constraint (n*(n-1)//2) plus noise variances (p)
    k = p * n_components - n_components * (n_components - 1) // 2 + p
    bic = -2.0 * N * mean_ll + k * np.log(N)
    return bic, mean_ll


def bic_sweep(X_scaled, score_type):
    """Two-pass BIC sweep: coarse then fine. Returns (optimal_n, all_results_df)."""
    logger = logging.getLogger("bic_sweep")
    cfg = BIC_SWEEP[score_type]
    N = X_scaled.shape[0]
    p = X_scaled.shape[1]

    # --- Coarse sweep ---
    coarse_range = list(range(cfg["coarse_start"], cfg["coarse_stop"] + 1, cfg["coarse_step"]))
    # Cap at p
    coarse_range = [n for n in coarse_range if n < p]
    logger.info("Coarse BIC sweep: %d values (%d to %d, step %d)",
                len(coarse_range), coarse_range[0], coarse_range[-1], cfg["coarse_step"])

    results = []
    for n in coarse_range:
        bic, ll = compute_fa_bic(X_scaled, n, N)
        results.append({"n_components": n, "bic": bic, "mean_ll": ll, "phase": "coarse"})
        logger.info("  n=%d -> BIC=%.1f, LL=%.4f", n, bic, ll)

    # Find coarse minimum
    coarse_df = pd.DataFrame(results)
    coarse_best_idx = coarse_df["bic"].idxmin()
    coarse_best_n = int(coarse_df.loc[coarse_best_idx, "n_components"])
    logger.info("Coarse best: n=%d (BIC=%.1f)", coarse_best_n, coarse_df.loc[coarse_best_idx, "bic"])

    # --- Fine sweep (skip if minimum is at boundary of coarse range) ---
    at_boundary = (coarse_best_n == coarse_range[-1])
    if at_boundary:
        logger.info("BIC minimum at upper boundary (%d) -- skipping fine sweep "
                     "(data supports many factors)", coarse_best_n)
    else:
        half = BIC_FINE_HALFWIDTH * cfg["coarse_step"]
        fine_step = BIC_FINE_STEP[score_type]
        fine_start = max(2, coarse_best_n - half)
        fine_stop = min(p - 1, coarse_best_n + half)
        fine_range = [n for n in range(fine_start, fine_stop + 1, fine_step)
                      if n not in set(coarse_range)]
        logger.info("Fine BIC sweep: %d values (%d to %d, step %d)",
                     len(fine_range), fine_start, fine_stop, fine_step)

        for n in fine_range:
            bic, ll = compute_fa_bic(X_scaled, n, N)
            results.append({"n_components": n, "bic": bic, "mean_ll": ll, "phase": "fine"})
            logger.info("  n=%d -> BIC=%.1f, LL=%.4f", n, bic, ll)

    # Combine and find overall minimum
    df = pd.DataFrame(results).sort_values("n_components").reset_index(drop=True)
    best_idx = df["bic"].idxmin()
    optimal_n = int(df.loc[best_idx, "n_components"])
    best_bic = df.loc[best_idx, "bic"]
    logger.info("BIC optimal: n=%d (BIC=%.1f)", optimal_n, best_bic)

    return optimal_n, df


def plot_bic_curve(bic_df, score_type, optimal_n, reports_dir):
    """Dual panel: BIC curve (left) + LL curve (right), marking BIC optimum and old PCA-90% value."""
    old_n = OLD_COMPONENTS[score_type]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: BIC curve
    coarse = bic_df[bic_df["phase"] == "coarse"]
    fine = bic_df[bic_df["phase"] == "fine"]

    ax1.plot(coarse["n_components"], coarse["bic"], "o-", color="steelblue",
             markersize=4, label="Coarse sweep", alpha=0.8)
    if len(fine) > 0:
        ax1.plot(fine["n_components"], fine["bic"], "s-", color="darkorange",
                 markersize=3, label="Fine sweep", alpha=0.8)
    ax1.axvline(x=optimal_n, color="red", linestyle="--", alpha=0.7,
                label=f"BIC minimum: {optimal_n}")
    ax1.axvline(x=old_n, color="gray", linestyle=":", alpha=0.5,
                label=f"Old PCA-90% match: {old_n}")
    ax1.set_xlabel("Number of Factors")
    ax1.set_ylabel("BIC")
    ax1.set_title(f"FA BIC vs Components -- VQI-{score_type.upper()}")
    ax1.legend(fontsize=8)

    # Right: LL curve
    sorted_df = bic_df.sort_values("n_components")
    ax2.plot(sorted_df["n_components"], sorted_df["mean_ll"], "o-", color="darkgreen", markersize=3)
    ax2.axvline(x=optimal_n, color="red", linestyle="--", alpha=0.7,
                label=f"BIC minimum: {optimal_n}")
    ax2.axvline(x=old_n, color="gray", linestyle=":", alpha=0.5,
                label=f"Old PCA-90% match: {old_n}")
    ax2.set_xlabel("Number of Factors")
    ax2.set_ylabel("Mean Log-Likelihood per Sample")
    ax2.set_title(f"FA Log-Likelihood vs Components -- VQI-{score_type.upper()}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(reports_dir, f"fa_bic_vs_components_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved BIC curve -> %s", path)


def grid_search(X, y, output_dir):
    """Same 30-config grid as original: OOB + top-5 CV."""
    logger = logging.getLogger("grid_search")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    n_configs = len(N_ESTIMATORS_GRID) * len(MAX_FEATURES_GRID)
    logger.info("Starting grid search: %d configs", n_configs)

    for n_est in N_ESTIMATORS_GRID:
        for max_feat in MAX_FEATURES_GRID:
            # Cap max_features to number of FA components
            if isinstance(max_feat, int) and max_feat > X.shape[1]:
                logger.info(
                    "  Skipping n_est=%d, max_feat=%d (exceeds %d factors)",
                    n_est, max_feat, X.shape[1],
                )
                continue

            params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": max_feat}
            clf = RandomForestClassifier(**params)
            clf.fit(X, y)
            oob_acc = clf.oob_score_

            rows.append({
                "n_estimators": n_est,
                "max_features": str(max_feat),
                "oob_error": round(1.0 - oob_acc, 6),
                "oob_accuracy": round(oob_acc, 6),
                "cv_accuracy_mean": None,
                "cv_accuracy_std": None,
            })
            logger.info(
                "  n_est=%d, max_feat=%s -> OOB_acc=%.4f",
                n_est, max_feat, oob_acc,
            )

    # Sort by OOB error ascending (best first)
    rows.sort(key=lambda r: r["oob_error"])

    # Top-5 get 5-fold CV
    logger.info("Running 5-fold CV for top 5 configs...")
    for row in rows[:5]:
        n_est = row["n_estimators"]
        max_feat = row["max_features"]
        mf = int(max_feat) if max_feat.isdigit() else max_feat
        params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": mf}
        clf = RandomForestClassifier(**params)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        row["cv_accuracy_mean"] = round(float(np.mean(cv_scores)), 6)
        row["cv_accuracy_std"] = round(float(np.std(cv_scores)), 6)
        logger.info(
            "  n_est=%d, max_feat=%s -> CV=%.4f +/- %.4f",
            n_est, max_feat, row["cv_accuracy_mean"], row["cv_accuracy_std"],
        )

    # Save grid results
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Grid search results saved to %s (%d rows)", csv_path, len(df))

    best = rows[0]
    best_mf = int(best["max_features"]) if best["max_features"].isdigit() else best["max_features"]
    best_params = {
        "n_estimators": best["n_estimators"],
        "max_features": best_mf,
    }
    logger.info("Best: n_est=%d, max_feat=%s, OOB_acc=%.4f",
                best_params["n_estimators"], best_params["max_features"], best["oob_accuracy"])
    return best_params


def train_final(X, y, best_params, output_dir, model_path, n_components, selection_method):
    """Train final RF with best params, compute OOB + confusion matrix + precision/recall."""
    logger = logging.getLogger("train_final")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    params = {**FIXED_PARAMS, **best_params}
    logger.info("Training final model: %s", params)

    clf = RandomForestClassifier(**params)
    clf.fit(X, y)

    oob_acc = clf.oob_score_
    train_acc = clf.score(X, y)
    train_preds = clf.predict(X)
    cm = confusion_matrix(y, train_preds)
    report = classification_report(y, train_preds, output_dict=True)

    logger.info("Final: OOB_acc=%.4f, train_acc=%.4f", oob_acc, train_acc)
    logger.info("Confusion matrix:\n%s", cm)

    # Save model
    joblib.dump(clf, model_path)
    logger.info("Model saved to %s", model_path)

    # Save metrics
    metrics = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_features": best_params["max_features"],
        "oob_error": round(1.0 - oob_acc, 6),
        "oob_accuracy": round(oob_acc, 6),
        "training_accuracy": round(float(train_acc), 6),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_components": int(n_components),
        "component_selection_method": selection_method,
        "n_class_0": int(np.sum(y == 0)),
        "n_class_1": int(np.sum(y == 1)),
        "confusion_matrix": cm.tolist(),
        "precision_0": round(float(report["0"]["precision"]), 4),
        "recall_0": round(float(report["0"]["recall"]), 4),
        "precision_1": round(float(report["1"]["precision"]), 4),
        "recall_1": round(float(report["1"]["recall"]), 4),
    }

    yaml_path = os.path.join(output_dir, "training_metrics.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
    logger.info("Metrics saved to %s", yaml_path)

    return metrics


# ---------------------------------------------------------------------------
# FA-Specific Visualizations
# ---------------------------------------------------------------------------

def plot_noise_variance(fa_obj, score_type, reports_dir):
    """Plot distribution of per-feature noise variances (uniquenesses)."""
    noise_var = fa_obj.noise_variance_

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(noise_var)), sorted(noise_var, reverse=True),
           color="darkorange", alpha=0.8)
    ax.axhline(y=np.mean(noise_var), color="red", linestyle="--", alpha=0.7,
               label=f"Mean = {np.mean(noise_var):.3f}")
    ax.set_xlabel("Feature (sorted by noise variance)")
    ax.set_ylabel("Noise Variance (Uniqueness)")
    ax.set_title(f"FA Per-Feature Noise Variance -- VQI-{score_type.upper()} ({len(noise_var)} features)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(reports_dir, f"noise_variance_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved noise variance plot -> %s", path)

    return noise_var


def plot_loadings_heatmap(fa_obj, score_type, reports_dir, n_show=20):
    """Plot factor loadings heatmap: top factors vs top original features."""
    # components_ shape: (n_components, n_features)
    L = fa_obj.components_.T  # transpose to (n_features, n_components)
    abs_L = np.abs(L)

    # Select top n_show factors by max absolute loading
    factor_importance = abs_L.max(axis=0)
    top_factors = np.argsort(factor_importance)[-n_show:][::-1]

    # Select top n_show features by max absolute loading
    feat_importance = abs_L.max(axis=1)
    top_feats = np.argsort(feat_importance)[-n_show:][::-1]

    submatrix = L[np.ix_(top_feats, top_factors)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(submatrix, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Original Feature Index")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"F{i}" for i in top_factors], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"Feat{i}" for i in top_feats], fontsize=7)
    ax.set_title(f"FA Loadings Matrix (Top {n_show} Factors x Top {n_show} Features) -- VQI-{score_type.upper()}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = os.path.join(reports_dir, f"loadings_heatmap_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved loadings heatmap -> %s", path)


def write_analysis(reports_dir, all_metrics, all_noise_vars, all_fa_objs, all_bic_data):
    """Write analysis.md summarizing FA results."""
    lines = [
        "# Factor Analysis Dimensionality Reduction -- Analysis",
        "",
        "**Date:** 2026-03-05",
        "**Method:** StandardScaler -> BIC sweep -> FactorAnalysis(BIC-optimal n) -> RF grid search (same 30-config grid)",
        "",
        "## Component Selection (BIC Minimization)",
        "",
    ]

    for st in ["s", "v"]:
        optimal_n, bic_df = all_bic_data[st]
        best_row = bic_df.loc[bic_df["bic"].idxmin()]
        old_n = OLD_COMPONENTS[st]
        lines.append(f"### VQI-{st.upper()}")
        lines.append("")
        lines.append(f"- **BIC sweep:** coarse ({BIC_SWEEP[st]['coarse_start']}-{BIC_SWEEP[st]['coarse_stop']}, "
                      f"step {BIC_SWEEP[st]['coarse_step']}) + fine (step 1 around minimum)")
        lines.append(f"- **BIC-optimal components:** {optimal_n} (BIC = {best_row['bic']:.1f})")
        lines.append(f"- **Previous (PCA-90% match):** {old_n}")
        lines.append(f"- **Change:** {optimal_n - old_n:+d} components")
        lines.append("")

    lines.extend([
        "## Results Summary",
        "",
        "| Metric | VQI-S Full | VQI-S FA | VQI-V Full | VQI-V FA |",
        "|--------|-----------|---------|-----------|---------|",
    ])

    ms = all_metrics["s"]
    mv = all_metrics["v"]
    bs = BASELINES["s"]
    bv = BASELINES["v"]

    lines.append(f"| Features / Factors | {bs['n_features']} | {ms['n_features']} | {bv['n_features']} | {mv['n_features']} |")
    lines.append(f"| Best n_estimators | {bs['n_estimators']} | {ms['n_estimators']} | {bv['n_estimators']} | {mv['n_estimators']} |")
    lines.append(f"| Best max_features | {bs['max_features']} | {ms['max_features']} | {bv['max_features']} | {mv['max_features']} |")
    lines.append(f"| OOB accuracy | {bs['oob_accuracy']:.4f} | {ms['oob_accuracy']:.4f} | {bv['oob_accuracy']:.4f} | {mv['oob_accuracy']:.4f} |")
    lines.append(f"| OOB diff vs full | -- | {ms['oob_accuracy'] - bs['oob_accuracy']:+.4f} | -- | {mv['oob_accuracy'] - bv['oob_accuracy']:+.4f} |")
    lines.append(f"| Training accuracy | {bs['training_accuracy']:.4f} | {ms['training_accuracy']:.4f} | {bv['training_accuracy']:.4f} | {mv['training_accuracy']:.4f} |")
    lines.append(f"| Precision (Class 0) | -- | {ms['precision_0']:.4f} | -- | {mv['precision_0']:.4f} |")
    lines.append(f"| Recall (Class 0) | -- | {ms['recall_0']:.4f} | -- | {mv['recall_0']:.4f} |")
    lines.append(f"| Precision (Class 1) | -- | {ms['precision_1']:.4f} | -- | {mv['precision_1']:.4f} |")
    lines.append(f"| Recall (Class 1) | -- | {ms['recall_1']:.4f} | -- | {mv['recall_1']:.4f} |")

    lines.append("")
    lines.append("## Factor Analysis Statistics")
    lines.append("")

    for st in ["s", "v"]:
        nv = all_noise_vars[st]
        fa_obj = all_fa_objs[st]
        optimal_n = all_bic_data[st][0]
        communalities = 1.0 - nv

        lines.append(f"### VQI-{st.upper()} ({optimal_n} factors, BIC-selected)")
        lines.append("")
        lines.append(f"- **Convergence iterations:** {fa_obj.n_iter_}")
        lines.append(f"- **Noise variance range:** [{np.min(nv):.4f}, {np.max(nv):.4f}]")
        lines.append(f"- **Noise variance mean:** {np.mean(nv):.4f}")
        lines.append(f"- **Communality range:** [{np.min(communalities):.4f}, {np.max(communalities):.4f}]")
        lines.append(f"- **Communality mean:** {np.mean(communalities):.4f}")
        lines.append(f"- **Features with communality > 0.5:** {np.sum(communalities > 0.5)}")
        lines.append(f"- **Features with communality > 0.8:** {np.sum(communalities > 0.8)}")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "Factor Analysis models observed features as linear combinations of latent factors plus ",
        "per-feature noise. Unlike PCA, FA explicitly separates shared variance (communality) from ",
        "unique variance (noise). Features with low communality are mostly noise and contribute ",
        "little shared information.",
        "",
        "BIC (Bayesian Information Criterion) penalizes model complexity, selecting the number of ",
        "factors that best balances fit (log-likelihood) against parsimony. This replaces the previous ",
        "approach of matching PCA-90%'s component count, allowing FA to select its own optimal ",
        "dimensionality.",
        "",
        "The noise variance plots show which features have the most unique/unexplained variance. ",
        "The loadings heatmaps reveal which original features load onto which factors, showing ",
        "the latent structure.",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `models/vqi_rf_fa_model.joblib` | FA VQI-S RF model |",
        "| `models/vqi_v_rf_fa_model.joblib` | FA VQI-V RF model |",
        "| `models/vqi_fa_scaler_s.joblib` | VQI-S StandardScaler |",
        "| `models/vqi_fa_scaler_v.joblib` | VQI-V StandardScaler |",
        "| `models/vqi_fa_transformer_s.joblib` | VQI-S FA transformer |",
        "| `models/vqi_fa_transformer_v.joblib` | VQI-V FA transformer |",
        "| `data/training_fa/training_metrics.yaml` | VQI-S training metrics |",
        "| `data/training_fa_v/training_metrics.yaml` | VQI-V training metrics |",
        "| `reports/fa_bic_vs_components_s.png` | VQI-S BIC curve |",
        "| `reports/fa_bic_vs_components_v.png` | VQI-V BIC curve |",
        "| `reports/noise_variance_s.png` | VQI-S noise variance plot |",
        "| `reports/noise_variance_v.png` | VQI-V noise variance plot |",
        "| `reports/loadings_heatmap_s.png` | VQI-S loadings heatmap |",
        "| `reports/loadings_heatmap_v.png` | VQI-V loadings heatmap |",
        "| `reports/analysis.md` | This file |",
    ])

    path = os.path.join(reports_dir, "fa_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Analysis report written to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger = logging.getLogger("fa_models")
    t0 = time.time()

    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "step6", "dimensionality_reduction")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_metrics = {}
    all_noise_vars = {}
    all_fa_objs = {}
    all_bic_data = {}

    for score_type in ["s", "v"]:
        prefix = f"VQI-{score_type.upper()}"
        suffix = "_v" if score_type == "v" else ""

        logger.info("=" * 60)
        logger.info("Starting %s FA pipeline (BIC component selection)", prefix)
        logger.info("=" * 60)

        # Output paths
        data_dir = os.path.join(PROJECT_ROOT, "data", "step6", "dimensionality_reduction", f"training_fa{suffix}")
        model_path = os.path.join(models_dir, f"vqi{suffix}_rf_fa_model.joblib")
        scaler_path = os.path.join(models_dir, f"vqi_fa_scaler_{score_type}.joblib")
        fa_path = os.path.join(models_dir, f"vqi_fa_transformer_{score_type}.joblib")

        # 1. Load data
        logger.info("[%s] Loading data...", prefix)
        X, y = load_data(score_type)
        logger.info("[%s] Loaded: X=%s, y=%s", prefix, X.shape, y.shape)

        # 2. Scale
        logger.info("[%s] Applying StandardScaler...", prefix)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. BIC sweep
        logger.info("[%s] Running BIC sweep for component selection...", prefix)
        optimal_n, bic_df = bic_sweep(X_scaled, score_type)
        all_bic_data[score_type] = (optimal_n, bic_df)
        logger.info("[%s] BIC-optimal components: %d (was %d)", prefix, optimal_n, OLD_COMPONENTS[score_type])

        # 4. Plot BIC curve
        plot_bic_curve(bic_df, score_type, optimal_n, reports_dir)

        # 5. Apply FA with BIC-optimal n
        logger.info("[%s] Applying FactorAnalysis (%d components, BIC-selected)...", prefix, optimal_n)
        fa = FactorAnalysis(n_components=optimal_n, random_state=42)
        X_fa = fa.fit_transform(X_scaled)
        logger.info("[%s] FA result: %s (n_iter=%d)", prefix, X_fa.shape, fa.n_iter_)

        # Save scaler and FA transformer
        joblib.dump(scaler, scaler_path)
        joblib.dump(fa, fa_path)
        logger.info("[%s] Saved scaler -> %s", prefix, scaler_path)
        logger.info("[%s] Saved FA transformer -> %s", prefix, fa_path)

        # 6. Visualizations
        logger.info("[%s] Generating FA visualizations...", prefix)
        noise_var = plot_noise_variance(fa, score_type, reports_dir)
        plot_loadings_heatmap(fa, score_type, reports_dir)

        all_noise_vars[score_type] = noise_var
        all_fa_objs[score_type] = fa

        # 7. Grid search
        logger.info("[%s] Running grid search...", prefix)
        best_params = grid_search(X_fa, y, data_dir)

        # 8. Train final model
        logger.info("[%s] Training final model...", prefix)
        metrics = train_final(X_fa, y, best_params, data_dir, model_path,
                              optimal_n, "BIC minimization (two-pass sweep)")

        all_metrics[score_type] = metrics
        logger.info(
            "[%s] Done: OOB_acc=%.4f (baseline=%.4f, diff=%+.4f)",
            prefix, metrics["oob_accuracy"],
            BASELINES[score_type]["oob_accuracy"],
            metrics["oob_accuracy"] - BASELINES[score_type]["oob_accuracy"],
        )

    # Write analysis report
    write_analysis(reports_dir, all_metrics, all_noise_vars, all_fa_objs, all_bic_data)

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("FACTOR ANALYSIS MODEL TRAINING COMPLETE (BIC-based)")
    print("=" * 60)
    for st in ["s", "v"]:
        m = all_metrics[st]
        b = BASELINES[st]
        d = m["oob_accuracy"] - b["oob_accuracy"]
        opt_n = all_bic_data[st][0]
        print(
            f"  VQI-{st.upper()}: {b['n_features']} -> {opt_n} factors (BIC) | "
            f"OOB {m['oob_accuracy']:.4f} (was {b['oob_accuracy']:.4f}, {d:+.4f}) | "
            f"n_est={m['n_estimators']}, max_feat={m['max_features']}"
        )
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
