"""
PCA-Reduced Random Forest Model Training & Comparison

Trains RF models on PCA-reduced features and compares OOB accuracy
against full-feature baselines.

PCA settings (from pca_dimensionality.py analysis):
  90% threshold: VQI-S 99 PCs, VQI-V 47 PCs
  95% threshold: VQI-S 156 PCs, VQI-V 60 PCs

Usage:
    python train_pca_models.py              # train 90% models (default)
    python train_pca_models.py --threshold 95   # train 95% models
    python train_pca_models.py --comparison     # write 3-way comparison only
"""

import argparse
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
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

# PCA components by variance threshold
PCA_COMPONENTS = {
    90: {"s": 99, "v": 47},
    95: {"s": 156, "v": 60},
}

# Original baselines (from training_metrics.yaml)
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


def apply_pca(X, n_components):
    """StandardScaler -> PCA(n_components) -> return (X_pca, pca_obj, scaler_obj)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = np.sum(pca.explained_variance_ratio_)
    logging.info(
        "PCA: %d features -> %d PCs (%.4f variance explained)",
        X.shape[1], n_components, var_explained,
    )
    return X_pca, pca, scaler


def grid_search(X, y, output_dir):
    """Same 30-config grid as original: OOB + top-5 CV."""
    logger = logging.getLogger("grid_search")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    n_configs = len(N_ESTIMATORS_GRID) * len(MAX_FEATURES_GRID)
    logger.info("Starting grid search: %d configs", n_configs)

    for n_est in N_ESTIMATORS_GRID:
        for max_feat in MAX_FEATURES_GRID:
            # Cap max_features to number of PCA components
            if isinstance(max_feat, int) and max_feat > X.shape[1]:
                logger.info(
                    "  Skipping n_est=%d, max_feat=%d (exceeds %d PCs)",
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


def train_final(X, y, best_params, output_dir, model_path):
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


def get_model_size_mb(path):
    """Return file size in MB."""
    return round(os.path.getsize(path) / (1024 * 1024), 2)


def get_pca_paths(threshold, score_type):
    """Return (data_dir, model_path, scaler_path, pca_path) for a given threshold and score type."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    suffix = "" if score_type == "s" else "_v"

    if threshold == 90:
        tag = "pca"
    else:
        tag = f"pca{threshold}"

    data_dir = os.path.join(PROJECT_ROOT, "data", "step6", "dimensionality_reduction", f"training_{tag}{suffix}")
    model_path = os.path.join(
        models_dir,
        f"vqi{'_v' if score_type == 'v' else ''}_rf_{tag}_model.joblib",
    )
    scaler_path = os.path.join(models_dir, f"vqi_{tag}_scaler_{score_type}.joblib")
    pca_path = os.path.join(models_dir, f"vqi_{tag}_transformer_{score_type}.joblib")
    return data_dir, model_path, scaler_path, pca_path


def load_metrics(data_dir):
    """Load training_metrics.yaml from a data directory."""
    yaml_path = os.path.join(data_dir, "training_metrics.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_comparison_3way(out_path):
    """Write 3-way markdown comparison: Full vs 95% PCA vs 90% PCA."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_s = BASELINES["s"]
    base_v = BASELINES["v"]

    # Load metrics for both thresholds
    metrics = {}
    for threshold in [90, 95]:
        metrics[threshold] = {}
        for st in ["s", "v"]:
            data_dir, _, _, _ = get_pca_paths(threshold, st)
            metrics[threshold][st] = load_metrics(data_dir)

    # Model sizes
    models_dir = os.path.join(PROJECT_ROOT, "models")
    size_full_s = get_model_size_mb(os.path.join(models_dir, "vqi_rf_model.joblib"))
    size_full_v = get_model_size_mb(os.path.join(models_dir, "vqi_v_rf_model.joblib"))

    sizes = {}
    for threshold in [90, 95]:
        sizes[threshold] = {}
        for st in ["s", "v"]:
            _, model_path, _, _ = get_pca_paths(threshold, st)
            sizes[threshold][st] = get_model_size_mb(model_path)

    def diff(m, st):
        return m["oob_accuracy"] - BASELINES[st]["oob_accuracy"]

    m90s = metrics[90]["s"]
    m90v = metrics[90]["v"]
    m95s = metrics[95]["s"]
    m95v = metrics[95]["v"]

    lines = [
        "# PCA-Reduced RF Model Comparison (3-Way)",
        "",
        "**Date:** 2026-02-21",
        "**Method:** StandardScaler -> PCA -> RF grid search (same 30-config grid)",
        "",
        "## 3-Way Comparison: Full vs 95% PCA vs 90% PCA",
        "",
        "| Metric | VQI-S Full | VQI-S 95% | VQI-S 90% | VQI-V Full | VQI-V 95% | VQI-V 90% |",
        "|--------|-----------|-----------|-----------|-----------|-----------|-----------|",
        f"| Features / PCs | {base_s['n_features']} | {m95s['n_features']} | {m90s['n_features']} | {base_v['n_features']} | {m95v['n_features']} | {m90v['n_features']} |",
        f"| Best n_estimators | {base_s['n_estimators']} | {m95s['n_estimators']} | {m90s['n_estimators']} | {base_v['n_estimators']} | {m95v['n_estimators']} | {m90v['n_estimators']} |",
        f"| Best max_features | {base_s['max_features']} | {m95s['max_features']} | {m90s['max_features']} | {base_v['max_features']} | {m95v['max_features']} | {m90v['max_features']} |",
        f"| OOB accuracy | {base_s['oob_accuracy']:.4f} | {m95s['oob_accuracy']:.4f} | {m90s['oob_accuracy']:.4f} | {base_v['oob_accuracy']:.4f} | {m95v['oob_accuracy']:.4f} | {m90v['oob_accuracy']:.4f} |",
        f"| OOB accuracy diff | — | {diff(m95s, 's'):+.4f} | {diff(m90s, 's'):+.4f} | — | {diff(m95v, 'v'):+.4f} | {diff(m90v, 'v'):+.4f} |",
        f"| Training accuracy | {base_s['training_accuracy']:.4f} | {m95s['training_accuracy']:.4f} | {m90s['training_accuracy']:.4f} | {base_v['training_accuracy']:.4f} | {m95v['training_accuracy']:.4f} | {m90v['training_accuracy']:.4f} |",
        f"| Model size (MB) | {size_full_s} | {sizes[95]['s']} | {sizes[90]['s']} | {size_full_v} | {sizes[95]['v']} | {sizes[90]['v']} |",
        f"| Precision (Class 0) | — | {m95s['precision_0']:.4f} | {m90s['precision_0']:.4f} | — | {m95v['precision_0']:.4f} | {m90v['precision_0']:.4f} |",
        f"| Recall (Class 0) | — | {m95s['recall_0']:.4f} | {m90s['recall_0']:.4f} | — | {m95v['recall_0']:.4f} | {m90v['recall_0']:.4f} |",
        f"| Precision (Class 1) | — | {m95s['precision_1']:.4f} | {m90s['precision_1']:.4f} | — | {m95v['precision_1']:.4f} | {m90v['precision_1']:.4f} |",
        f"| Recall (Class 1) | — | {m95s['recall_1']:.4f} | {m90s['recall_1']:.4f} | — | {m95v['recall_1']:.4f} | {m90v['recall_1']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"The 95% variance threshold retains more discriminative information than the 90% threshold. "
        f"VQI-S goes from 430 features to {m95s['n_features']} PCs (95%) or {m90s['n_features']} PCs (90%), "
        f"with OOB accuracy drops of {diff(m95s, 's'):+.4f} and {diff(m90s, 's'):+.4f} respectively. "
        f"VQI-V goes from {base_v['n_features']} features to {m95v['n_features']} PCs (95%) or {m90v['n_features']} PCs (90%), "
        f"with OOB accuracy drops of {diff(m95v, 'v'):+.4f} and {diff(m90v, 'v'):+.4f} respectively.",
        "",
        f"The 95% threshold recovers approximately "
        f"{abs(diff(m90s, 's')) - abs(diff(m95s, 's')):.4f} pp for VQI-S and "
        f"{abs(diff(m90v, 'v')) - abs(diff(m95v, 'v')):.4f} pp for VQI-V compared to 90%, "
        f"at the cost of {m95s['n_features'] - m90s['n_features']} additional PCs for VQI-S "
        f"and {m95v['n_features'] - m90v['n_features']} for VQI-V. "
        f"The full-feature models remain the production default, as PCA reduction "
        f"consistently underperforms the original feature set.",
        "",
        "## Output Files",
        "",
        "### 90% variance threshold",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `models/vqi_rf_pca_model.joblib` | PCA 90% VQI-S RF model |",
        "| `models/vqi_v_rf_pca_model.joblib` | PCA 90% VQI-V RF model |",
        "| `models/vqi_pca_scaler_s.joblib` | VQI-S StandardScaler (90%) |",
        "| `models/vqi_pca_scaler_v.joblib` | VQI-V StandardScaler (90%) |",
        "| `models/vqi_pca_transformer_s.joblib` | VQI-S PCA transformer (90%) |",
        "| `models/vqi_pca_transformer_v.joblib` | VQI-V PCA transformer (90%) |",
        "| `data/training_pca/training_metrics.yaml` | VQI-S 90% training metrics |",
        "| `data/training_pca_v/training_metrics.yaml` | VQI-V 90% training metrics |",
        "",
        "### 95% variance threshold",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `models/vqi_rf_pca95_model.joblib` | PCA 95% VQI-S RF model |",
        "| `models/vqi_v_rf_pca95_model.joblib` | PCA 95% VQI-V RF model |",
        "| `models/vqi_pca95_scaler_s.joblib` | VQI-S StandardScaler (95%) |",
        "| `models/vqi_pca95_scaler_v.joblib` | VQI-V StandardScaler (95%) |",
        "| `models/vqi_pca95_transformer_s.joblib` | VQI-S PCA transformer (95%) |",
        "| `models/vqi_pca95_transformer_v.joblib` | VQI-V PCA transformer (95%) |",
        "| `data/training_pca95/training_metrics.yaml` | VQI-S 95% training metrics |",
        "| `data/training_pca95_v/training_metrics.yaml` | VQI-V 95% training metrics |",
        "",
        "| `reports/pca/comparison.md` | This file |",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logging.info("3-way comparison report written to %s", out_path)


def train_threshold(threshold):
    """Train PCA models for a given variance threshold (90 or 95)."""
    logger = logging.getLogger("pca_models")
    t0 = time.time()

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    pca_comps = PCA_COMPONENTS[threshold]
    all_metrics = {}

    for score_type in ["s", "v"]:
        prefix = f"VQI-{score_type.upper()}"
        n_pcs = pca_comps[score_type]

        logger.info("=" * 60)
        logger.info("Starting %s PCA-%d%% pipeline (%d PCs)", prefix, threshold, n_pcs)
        logger.info("=" * 60)

        # Output paths
        out_dir, model_path, scaler_path, pca_path = get_pca_paths(threshold, score_type)

        # 1. Load data
        logger.info("[%s] Loading data...", prefix)
        X, y = load_data(score_type)
        logger.info("[%s] Loaded: X=%s, y=%s", prefix, X.shape, y.shape)

        # 2. Apply PCA
        logger.info("[%s] Applying PCA (%d components)...", prefix, n_pcs)
        X_pca, pca_obj, scaler_obj = apply_pca(X, n_pcs)
        logger.info("[%s] PCA result: %s", prefix, X_pca.shape)

        # Save scaler and PCA transformer
        joblib.dump(scaler_obj, scaler_path)
        joblib.dump(pca_obj, pca_path)
        logger.info("[%s] Saved scaler -> %s", prefix, scaler_path)
        logger.info("[%s] Saved PCA transformer -> %s", prefix, pca_path)

        # 3. Grid search
        logger.info("[%s] Running grid search...", prefix)
        best_params = grid_search(X_pca, y, out_dir)

        # 4. Train final model
        logger.info("[%s] Training final model...", prefix)
        metrics = train_final(X_pca, y, best_params, out_dir, model_path)

        all_metrics[score_type] = metrics
        logger.info(
            "[%s] Done: OOB_acc=%.4f (baseline=%.4f, diff=%+.4f)",
            prefix, metrics["oob_accuracy"],
            BASELINES[score_type]["oob_accuracy"],
            metrics["oob_accuracy"] - BASELINES[score_type]["oob_accuracy"],
        )

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"PCA-{threshold}% MODEL TRAINING COMPLETE")
    print("=" * 60)
    for st in ["s", "v"]:
        m = all_metrics[st]
        b = BASELINES[st]
        d = m["oob_accuracy"] - b["oob_accuracy"]
        print(
            f"  VQI-{st.upper()}: {b['n_features']} -> {m['n_features']} PCs | "
            f"OOB {m['oob_accuracy']:.4f} (was {b['oob_accuracy']:.4f}, {d:+.4f}) | "
            f"n_est={m['n_estimators']}, max_feat={m['max_features']}"
        )
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="PCA-Reduced RF Model Training")
    parser.add_argument(
        "--threshold", type=int, choices=[90, 95], default=90,
        help="PCA variance threshold (90 or 95). Default: 90.",
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="Write 3-way comparison report only (no training). Requires both 90%% and 95%% models.",
    )
    args = parser.parse_args()

    setup_logging()
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "step6", "dimensionality_reduction")
    os.makedirs(reports_dir, exist_ok=True)

    if args.comparison:
        logging.info("Writing 3-way comparison report (no training)...")
        write_comparison_3way(os.path.join(reports_dir, "comparison.md"))
        print(f"Report: {os.path.join(reports_dir, 'comparison.md')}")
    else:
        train_threshold(args.threshold)

        # If both thresholds have been trained, write 3-way comparison
        try:
            write_comparison_3way(os.path.join(reports_dir, "comparison.md"))
            print(f"Report: {os.path.join(reports_dir, 'comparison.md')}")
        except FileNotFoundError:
            logging.info(
                "Only %d%% models trained so far. Run with --threshold %d to complete 3-way comparison.",
                args.threshold, 95 if args.threshold == 90 else 90,
            )


if __name__ == "__main__":
    main()
