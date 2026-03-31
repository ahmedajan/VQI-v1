"""
ICA-Reduced Random Forest Model Training & Analysis

Trains RF models on ICA-reduced features and compares OOB accuracy
against full-feature baselines.

ICA component count selected via Parallel Analysis (Horn's method):
retain components whose eigenvalues exceed the 95th percentile of
eigenvalues from column-permuted random data (100 permutations).

Usage:
    python train_ica_models.py
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
from scipy import stats
from sklearn.decomposition import FastICA, PCA
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

# Parallel Analysis config
PA_N_PERMUTATIONS = 100
PA_PERCENTILE = 95
PA_RANDOM_STATE = 42

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
# Parallel Analysis (Horn's method)
# ---------------------------------------------------------------------------

def parallel_analysis(X_scaled, score_type):
    """Horn's Parallel Analysis: retain components where real eigenvalue > 95th percentile of random.

    Returns (n_components, real_eigenvalues, random_thresholds).
    """
    logger = logging.getLogger("parallel_analysis")
    N, p = X_scaled.shape
    rng = np.random.RandomState(PA_RANDOM_STATE)

    # Real eigenvalues from PCA on actual data
    pca = PCA(n_components=p)
    pca.fit(X_scaled)
    real_eigenvalues = pca.explained_variance_

    # Permutation eigenvalues
    logger.info("Running %d permutations for PA (p=%d)...", PA_N_PERMUTATIONS, p)
    random_eigenvalues = np.zeros((PA_N_PERMUTATIONS, p))
    for i in range(PA_N_PERMUTATIONS):
        X_perm = X_scaled.copy()
        for col in range(p):
            rng.shuffle(X_perm[:, col])
        pca_perm = PCA(n_components=p)
        pca_perm.fit(X_perm)
        random_eigenvalues[i] = pca_perm.explained_variance_
        if (i + 1) % 20 == 0:
            logger.info("  Permutation %d/%d done", i + 1, PA_N_PERMUTATIONS)

    # 95th percentile threshold for each component
    random_thresholds = np.percentile(random_eigenvalues, PA_PERCENTILE, axis=0)

    # Count components where real > threshold
    n_components = int(np.sum(real_eigenvalues > random_thresholds))
    logger.info("PA result: %d components retained (real > %dth percentile of random)",
                n_components, PA_PERCENTILE)

    return n_components, real_eigenvalues, random_thresholds


def plot_parallel_analysis(real_eigenvalues, random_thresholds, n_components, score_type, reports_dir):
    """Scree plot with real eigenvalues, random threshold, shaded retained region."""
    old_n = OLD_COMPONENTS[score_type]
    n_show = min(len(real_eigenvalues), max(n_components + 20, 80))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, n_show + 1)

    ax.plot(x, real_eigenvalues[:n_show], "o-", color="steelblue", markersize=3,
            label="Real eigenvalues", linewidth=1.5)
    ax.plot(x, random_thresholds[:n_show], "s-", color="red", markersize=2,
            label=f"Random {PA_PERCENTILE}th percentile", linewidth=1.2, alpha=0.8)

    # Shade retained region
    ax.axvspan(0.5, n_components + 0.5, alpha=0.1, color="green", label=f"Retained: {n_components} ICs")

    # Mark crossover
    if n_components < n_show:
        ax.axvline(x=n_components + 0.5, color="green", linestyle="--", alpha=0.6)

    # Mark old value
    ax.axvline(x=old_n, color="gray", linestyle=":", alpha=0.5,
               label=f"Old PCA-90% match: {old_n}")

    ax.set_xlabel("Component Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Parallel Analysis (Horn's Method) -- VQI-{score_type.upper()}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, n_show + 0.5)
    plt.tight_layout()

    path = os.path.join(reports_dir, f"ica_parallel_analysis_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved parallel analysis plot -> %s", path)


def grid_search(X, y, output_dir):
    """Same 30-config grid as original: OOB + top-5 CV."""
    logger = logging.getLogger("grid_search")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    n_configs = len(N_ESTIMATORS_GRID) * len(MAX_FEATURES_GRID)
    logger.info("Starting grid search: %d configs", n_configs)

    for n_est in N_ESTIMATORS_GRID:
        for max_feat in MAX_FEATURES_GRID:
            # Cap max_features to number of ICA components
            if isinstance(max_feat, int) and max_feat > X.shape[1]:
                logger.info(
                    "  Skipping n_est=%d, max_feat=%d (exceeds %d ICs)",
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
# ICA-Specific Visualizations
# ---------------------------------------------------------------------------

def plot_kurtosis_distribution(X_ica, score_type, reports_dir):
    """Plot kurtosis of each IC -- ICA maximizes non-Gaussianity."""
    kurtosis_vals = stats.kurtosis(X_ica, axis=0, fisher=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(kurtosis_vals)), sorted(kurtosis_vals, reverse=True),
           color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Gaussian (kurtosis=0)")
    ax.set_xlabel("Independent Component (sorted by kurtosis)")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title(f"ICA Component Kurtosis Distribution -- VQI-{score_type.upper()} ({len(kurtosis_vals)} ICs)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(reports_dir, f"kurtosis_distribution_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved kurtosis plot -> %s", path)

    return kurtosis_vals


def plot_mixing_matrix_heatmap(ica_obj, score_type, reports_dir, n_show=20):
    """Plot mixing matrix heatmap: top ICs vs top original features."""
    # mixing_matrix_ shape: (n_features, n_components)
    A = ica_obj.mixing_

    # Use absolute values to find most important connections
    abs_A = np.abs(A)

    # Select top n_show ICs by max absolute mixing weight
    ic_importance = abs_A.max(axis=0)
    top_ics = np.argsort(ic_importance)[-n_show:][::-1]

    # Select top n_show features by max absolute mixing weight
    feat_importance = abs_A.max(axis=1)
    top_feats = np.argsort(feat_importance)[-n_show:][::-1]

    submatrix = A[np.ix_(top_feats, top_ics)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(submatrix, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Independent Component")
    ax.set_ylabel("Original Feature Index")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"IC{i}" for i in top_ics], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"F{i}" for i in top_feats], fontsize=7)
    ax.set_title(f"ICA Mixing Matrix (Top {n_show} ICs x Top {n_show} Features) -- VQI-{score_type.upper()}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = os.path.join(reports_dir, f"mixing_matrix_heatmap_{score_type}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved mixing matrix heatmap -> %s", path)


def write_analysis(reports_dir, all_metrics, all_kurtosis, all_ica_objs, all_pa_data):
    """Write analysis.md summarizing ICA results."""
    lines = [
        "# ICA Dimensionality Reduction -- Analysis",
        "",
        "**Date:** 2026-03-05",
        "**Method:** StandardScaler -> Parallel Analysis -> FastICA(PA-optimal n) -> RF grid search (same 30-config grid)",
        "",
        "## Component Selection (Parallel Analysis -- Horn's Method)",
        "",
        f"- **Permutations:** {PA_N_PERMUTATIONS}",
        f"- **Threshold:** {PA_PERCENTILE}th percentile of random eigenvalues",
        f"- **Random state:** {PA_RANDOM_STATE}",
        "",
    ]

    for st in ["s", "v"]:
        n_pa, real_eig, rand_thresh = all_pa_data[st]
        old_n = OLD_COMPONENTS[st]
        # Find crossover eigenvalue
        crossover_real = real_eig[n_pa - 1] if n_pa > 0 else 0
        crossover_rand = rand_thresh[n_pa - 1] if n_pa > 0 else 0
        lines.append(f"### VQI-{st.upper()}")
        lines.append("")
        lines.append(f"- **PA-optimal components:** {n_pa}")
        lines.append(f"- **Previous (PCA-90% match):** {old_n}")
        lines.append(f"- **Change:** {n_pa - old_n:+d} components")
        lines.append(f"- **Last retained eigenvalue:** {crossover_real:.4f} (threshold: {crossover_rand:.4f})")
        lines.append("")

    lines.extend([
        "## Results Summary",
        "",
        "| Metric | VQI-S Full | VQI-S ICA | VQI-V Full | VQI-V ICA |",
        "|--------|-----------|----------|-----------|----------|",
    ])

    ms = all_metrics["s"]
    mv = all_metrics["v"]
    bs = BASELINES["s"]
    bv = BASELINES["v"]

    lines.append(f"| Features / ICs | {bs['n_features']} | {ms['n_features']} | {bv['n_features']} | {mv['n_features']} |")
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
    lines.append("## ICA Component Statistics")
    lines.append("")

    for st in ["s", "v"]:
        k = all_kurtosis[st]
        ica_obj = all_ica_objs[st]
        n_pa = all_pa_data[st][0]
        lines.append(f"### VQI-{st.upper()} ({n_pa} ICs, PA-selected)")
        lines.append("")
        lines.append(f"- **Convergence iterations:** {ica_obj.n_iter_}")
        lines.append(f"- **Kurtosis range:** [{np.min(k):.2f}, {np.max(k):.2f}]")
        lines.append(f"- **Kurtosis mean:** {np.mean(k):.2f}")
        lines.append(f"- **Kurtosis median:** {np.median(k):.2f}")
        lines.append(f"- **ICs with |kurtosis| > 1 (super-Gaussian):** {np.sum(np.abs(k) > 1)}")
        lines.append(f"- **ICs with |kurtosis| > 3 (highly non-Gaussian):** {np.sum(np.abs(k) > 3)}")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "ICA (Independent Component Analysis) seeks statistically independent components by ",
        "maximizing non-Gaussianity, unlike PCA which maximizes variance. The kurtosis values ",
        "above indicate how non-Gaussian each component is -- higher absolute kurtosis means ",
        "more non-Gaussian and potentially more informative for separating classes.",
        "",
        "Parallel Analysis (Horn's method) determines the number of components by comparing ",
        "real eigenvalues against those from column-permuted random data. Components are retained ",
        "only if their eigenvalue exceeds the 95th percentile of the random distribution, ensuring ",
        "they capture more structure than noise. This replaces the previous approach of matching ",
        "PCA-90%'s component count.",
        "",
        "The mixing matrix heatmaps show how original features contribute to the independent ",
        "components, revealing which features form natural independent groupings.",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `models/vqi_rf_ica_model.joblib` | ICA VQI-S RF model |",
        "| `models/vqi_v_rf_ica_model.joblib` | ICA VQI-V RF model |",
        "| `models/vqi_ica_scaler_s.joblib` | VQI-S StandardScaler |",
        "| `models/vqi_ica_scaler_v.joblib` | VQI-V StandardScaler |",
        "| `models/vqi_ica_transformer_s.joblib` | VQI-S ICA transformer |",
        "| `models/vqi_ica_transformer_v.joblib` | VQI-V ICA transformer |",
        "| `data/training_ica/training_metrics.yaml` | VQI-S training metrics |",
        "| `data/training_ica_v/training_metrics.yaml` | VQI-V training metrics |",
        "| `reports/ica_parallel_analysis_s.png` | VQI-S parallel analysis plot |",
        "| `reports/ica_parallel_analysis_v.png` | VQI-V parallel analysis plot |",
        "| `reports/kurtosis_distribution_s.png` | VQI-S kurtosis plot |",
        "| `reports/kurtosis_distribution_v.png` | VQI-V kurtosis plot |",
        "| `reports/mixing_matrix_heatmap_s.png` | VQI-S mixing matrix |",
        "| `reports/mixing_matrix_heatmap_v.png` | VQI-V mixing matrix |",
        "| `reports/analysis.md` | This file |",
    ])

    path = os.path.join(reports_dir, "ica_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Analysis report written to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger = logging.getLogger("ica_models")
    t0 = time.time()

    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "step6", "dimensionality_reduction")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_metrics = {}
    all_kurtosis = {}
    all_ica_objs = {}
    all_pa_data = {}

    for score_type in ["s", "v"]:
        prefix = f"VQI-{score_type.upper()}"
        suffix = "_v" if score_type == "v" else ""

        logger.info("=" * 60)
        logger.info("Starting %s ICA pipeline (Parallel Analysis component selection)", prefix)
        logger.info("=" * 60)

        # Output paths
        data_dir = os.path.join(PROJECT_ROOT, "data", "step6", "dimensionality_reduction", f"training_ica{suffix}")
        model_path = os.path.join(models_dir, f"vqi{suffix}_rf_ica_model.joblib")
        scaler_path = os.path.join(models_dir, f"vqi_ica_scaler_{score_type}.joblib")
        ica_path = os.path.join(models_dir, f"vqi_ica_transformer_{score_type}.joblib")

        # 1. Load data
        logger.info("[%s] Loading data...", prefix)
        X, y = load_data(score_type)
        logger.info("[%s] Loaded: X=%s, y=%s", prefix, X.shape, y.shape)

        # 2. Scale (shared between PA and ICA)
        logger.info("[%s] Applying StandardScaler...", prefix)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Parallel Analysis
        logger.info("[%s] Running Parallel Analysis for component selection...", prefix)
        n_pa, real_eigenvalues, random_thresholds = parallel_analysis(X_scaled, score_type)
        all_pa_data[score_type] = (n_pa, real_eigenvalues, random_thresholds)
        logger.info("[%s] PA-optimal components: %d (was %d)", prefix, n_pa, OLD_COMPONENTS[score_type])

        # 4. Plot parallel analysis
        plot_parallel_analysis(real_eigenvalues, random_thresholds, n_pa, score_type, reports_dir)

        # 5. Apply ICA with PA-optimal n
        logger.info("[%s] Applying FastICA (%d components, PA-selected)...", prefix, n_pa)
        ica = FastICA(n_components=n_pa, random_state=42, max_iter=1000)
        X_ica = ica.fit_transform(X_scaled)
        logger.info("[%s] ICA result: %s (n_iter=%d)", prefix, X_ica.shape, ica.n_iter_)

        # Save scaler and ICA transformer
        joblib.dump(scaler, scaler_path)
        joblib.dump(ica, ica_path)
        logger.info("[%s] Saved scaler -> %s", prefix, scaler_path)
        logger.info("[%s] Saved ICA transformer -> %s", prefix, ica_path)

        # 6. Visualizations
        logger.info("[%s] Generating ICA visualizations...", prefix)
        kurtosis_vals = plot_kurtosis_distribution(X_ica, score_type, reports_dir)
        plot_mixing_matrix_heatmap(ica, score_type, reports_dir)

        all_kurtosis[score_type] = kurtosis_vals
        all_ica_objs[score_type] = ica

        # 7. Grid search
        logger.info("[%s] Running grid search...", prefix)
        best_params = grid_search(X_ica, y, data_dir)

        # 8. Train final model
        logger.info("[%s] Training final model...", prefix)
        metrics = train_final(X_ica, y, best_params, data_dir, model_path,
                              n_pa, f"Parallel Analysis (Horn's method, {PA_N_PERMUTATIONS} perms, {PA_PERCENTILE}th pctl)")

        all_metrics[score_type] = metrics
        logger.info(
            "[%s] Done: OOB_acc=%.4f (baseline=%.4f, diff=%+.4f)",
            prefix, metrics["oob_accuracy"],
            BASELINES[score_type]["oob_accuracy"],
            metrics["oob_accuracy"] - BASELINES[score_type]["oob_accuracy"],
        )

    # Write analysis report
    write_analysis(reports_dir, all_metrics, all_kurtosis, all_ica_objs, all_pa_data)

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("ICA MODEL TRAINING COMPLETE (Parallel Analysis-based)")
    print("=" * 60)
    for st in ["s", "v"]:
        m = all_metrics[st]
        b = BASELINES[st]
        d = m["oob_accuracy"] - b["oob_accuracy"]
        n_pa = all_pa_data[st][0]
        print(
            f"  VQI-{st.upper()}: {b['n_features']} -> {n_pa} ICs (PA) | "
            f"OOB {m['oob_accuracy']:.4f} (was {b['oob_accuracy']:.4f}, {d:+.4f}) | "
            f"n_est={m['n_estimators']}, max_feat={m['max_features']}"
        )
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
