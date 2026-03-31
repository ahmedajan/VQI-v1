"""
Step X1.3: Evaluate All Models on Validation + Test Sets

Evaluates all 7 models (RF + 6 alternatives) on held-out validation and test data
using identical metrics for both VQI-S and VQI-V:
  A) Classification: AUC-ROC, AUC-PR, F1, accuracy, precision, recall, confusion matrix
  B) Calibration: Brier score
  C) VQI-specific: ERC curves, ranked DET curves (pair-based, per dataset, per provider)
  D) Cross-system: P4/P5 ERC
  E) Inference speed: time per sample

Usage:
    python scripts/x1_evaluate_models.py [--score-type s|v|both] [--resume]
"""

import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import (
    TEST_DATASETS,
    load_test_features,
    load_test_pairs,
    load_validation_data,
)
from vqi.evaluation.det import compute_ranked_det
from vqi.evaluation.erc import (
    compute_erc,
    compute_pairwise_quality,
    find_tau_for_fnmr,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
TEST_SCORES_DIR = os.path.join(DATA_DIR, "step8", "full_feature", "test_scores")

RANDOM_STATE = 42
PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]

# Model definitions: (file_pattern, load_type, needs_scaling)
# file_pattern uses {prefix} = "vqi" for S, "vqi_v" for V
MODEL_DEFS = {
    "rf":       ("{prefix}_rf_model.joblib",       "joblib",  False),
    "xgboost":  ("{prefix}_xgboost_model.json",    "xgboost", False),
    "lightgbm": ("{prefix}_lightgbm_model.joblib", "joblib",  False),
    "logreg":   ("{prefix}_logreg_model.joblib",    "joblib",  True),
    "svm":      ("{prefix}_svm_model.joblib",       "joblib",  True),
    "mlp":      ("{prefix}_mlp_model.pt",           "pytorch", True),
    "tabnet":   ("{prefix}_tabnet_model.zip",       "tabnet",  True),
}

MODEL_ORDER = ["rf", "xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"]

logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _load_model(name, score_type):
    """Load a trained model and return (model, load_type, needs_scaling)."""
    file_pattern, load_type, needs_scaling = MODEL_DEFS[name]
    prefix = "vqi_v" if score_type == "v" else "vqi"
    filename = file_pattern.format(prefix=prefix)
    path = os.path.join(MODELS_DIR, filename)

    if load_type == "joblib":
        model = joblib.load(path)
    elif load_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(path)
    elif load_type == "pytorch":
        import torch
        import torch.nn as nn

        config_path = os.path.join(MODELS_DIR, f"{prefix}_mlp_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        class VQI_MLP(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_dim, h),
                        nn.ReLU(),
                        nn.BatchNorm1d(h),
                        nn.Dropout(dropout),
                    ])
                    prev_dim = h
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        model = VQI_MLP(cfg["input_dim"], cfg["hidden_layers"], cfg["dropout"])
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        model.eval()
        model._cfg = cfg  # attach config for reference
    elif load_type == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier()
        # TabNet load_model expects full path with .zip extension
        model.load_model(path)
    else:
        raise ValueError(f"Unknown load_type: {load_type}")

    return model, load_type, needs_scaling


def _load_scaler(score_type):
    """Load the feature scaler for models that need it."""
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(MODELS_DIR, f"x1{suffix}_feature_scaler.joblib")
    return joblib.load(path)


def _predict_proba(model, load_type, X):
    """Get P(class=1) predictions from a model."""
    if load_type == "pytorch":
        import torch
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            logits = model(X_t).squeeze(-1).numpy()
            return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    elif load_type == "tabnet":
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


def _to_vqi_scores(probas):
    """Convert probabilities to VQI scores [0-100]."""
    return np.clip(np.round(probas * 100), 0, 100).astype(int)


# ---------------------------------------------------------------------------
# A) Classification metrics on validation set
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true, probas):
    """Compute all classification metrics at threshold=0.5."""
    preds = (probas >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probas)),
        "auc_pr": float(average_precision_score(y_true, probas)),
        "brier_score": float(brier_score_loss(y_true, probas)),
    }


# ---------------------------------------------------------------------------
# C) ERC + DET on test sets (pair-based)
# ---------------------------------------------------------------------------
def _load_pair_scores(dataset, provider):
    """Load cosine similarity scores for pairs from a provider."""
    csv_path = os.path.join(
        TEST_SCORES_DIR, f"pair_scores_{dataset}_{provider}.csv"
    )
    df = pd.read_csv(csv_path)
    return df["cos_sim_snorm"].values


def compute_test_erc(vqi_scores, pairs, pair_labels, pair_sims):
    """Compute ERC for a set of test pairs given VQI scores and provider sims."""
    quality = compute_pairwise_quality(vqi_scores.astype(float), pairs)

    gen_mask = pair_labels == 1
    imp_mask = pair_labels == 0

    genuine_sim = pair_sims[gen_mask]
    impostor_sim = pair_sims[imp_mask]
    quality_gen = quality[gen_mask]
    quality_imp = quality[imp_mask]

    if len(genuine_sim) == 0 or len(impostor_sim) == 0:
        return None

    results = {}
    for target_fnmr, label in [(0.01, "fnmr1"), (0.10, "fnmr10")]:
        tau = find_tau_for_fnmr(genuine_sim, impostor_sim, target_fnmr)
        erc = compute_erc(genuine_sim, impostor_sim, quality_gen, quality_imp, tau)

        # Compute FNMR reduction at 10%, 20%, 30% rejection
        baseline_fnmr = erc["fnmr_values"][0]
        reductions = {}
        for target_rf in [0.10, 0.20, 0.30]:
            idx = np.argmin(np.abs(erc["reject_fracs"] - target_rf))
            fnmr_at_rf = erc["fnmr_values"][idx]
            if baseline_fnmr > 0 and not np.isnan(fnmr_at_rf):
                red_pct = (1.0 - fnmr_at_rf / baseline_fnmr) * 100
            else:
                red_pct = 0.0
            reductions[f"red_{int(target_rf*100)}pct"] = round(red_pct, 2)

        results[label] = {
            "baseline_fnmr": round(float(baseline_fnmr), 6),
            "tau": round(float(tau), 6),
            **reductions,
        }

    return results


def compute_test_det(vqi_scores, pairs, pair_labels, pair_sims):
    """Compute ranked DET for a set of test pairs given VQI scores."""
    quality = compute_pairwise_quality(vqi_scores.astype(float), pairs)

    gen_mask = pair_labels == 1
    imp_mask = pair_labels == 0

    genuine_sim = pair_sims[gen_mask]
    impostor_sim = pair_sims[imp_mask]
    quality_gen = quality[gen_mask]
    quality_imp = quality[imp_mask]

    if len(genuine_sim) == 0 or len(impostor_sim) == 0:
        return None

    det_result = compute_ranked_det(
        genuine_sim, impostor_sim, quality_gen, quality_imp
    )

    # Extract summary stats
    summary = {
        "eer_separation": round(float(det_result["eer_separation"]), 4)
        if not np.isnan(det_result["eer_separation"]) else None,
        "q_low": round(float(det_result["q_low"]), 2),
        "q_high": round(float(det_result["q_high"]), 2),
    }
    for group_name in ["bottom", "middle", "top"]:
        g = det_result["groups"][group_name]
        det = g.get("det")
        summary[f"eer_{group_name}"] = round(float(det["eer"]), 6) if det else None
        summary[f"n_gen_{group_name}"] = g["n_genuine"]
        summary[f"n_imp_{group_name}"] = g["n_impostor"]

    return summary


# ---------------------------------------------------------------------------
# E) Inference speed
# ---------------------------------------------------------------------------
def measure_inference_speed(model, load_type, X_sample, n_repeats=1000):
    """Measure inference time per sample."""
    # Warm up
    for _ in range(10):
        _predict_proba(model, load_type, X_sample)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _predict_proba(model, load_type, X_sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    n_samples = X_sample.shape[0]
    return {
        "ms_per_sample_mean": round(float(np.mean(times) / n_samples), 4),
        "ms_per_sample_p50": round(float(np.median(times) / n_samples), 4),
        "ms_per_sample_p99": round(float(np.percentile(times, 99) / n_samples), 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_one_model(name, score_type, X_val, y_val, scaler, test_data):
    """Evaluate a single model on all metrics. Returns dict of results."""
    logger.info("Evaluating %s (%s)...", name.upper(), score_type.upper())
    _flush()

    model, load_type, needs_scaling = _load_model(name, score_type)

    # Prepare features
    X_val_model = scaler.transform(X_val) if needs_scaling else X_val
    X_val_model = X_val_model.astype(np.float32)

    # A) Classification metrics on validation
    val_probas = _predict_proba(model, load_type, X_val_model)
    cls_metrics = compute_classification_metrics(y_val, val_probas)
    val_scores = _to_vqi_scores(val_probas)

    # Save validation predictions
    suffix = "_v" if score_type == "v" else ""
    np.save(os.path.join(X1_DIR, f"{name}{suffix}_val_predictions.npy"), val_probas)

    # Confusion matrix
    cm = confusion_matrix(y_val, (val_probas >= 0.5).astype(int))
    np.save(os.path.join(X1_DIR, f"{name}{suffix}_confusion_matrix.npy"), cm)

    logger.info("  Val AUC=%.4f F1=%.4f Acc=%.4f Brier=%.4f",
                cls_metrics["auc_roc"], cls_metrics["f1"],
                cls_metrics["accuracy"], cls_metrics["brier_score"])
    _flush()

    # C) ERC + DET on test sets
    erc_results = {}
    det_results = {}

    for ds in TEST_DATASETS:
        X_test, pairs, pair_labels = test_data[ds]
        X_test_model = scaler.transform(X_test) if needs_scaling else X_test
        X_test_model = X_test_model.astype(np.float32)

        # Handle NaN rows (extraction failures) — predict 50 for NaN rows
        nan_mask = np.any(np.isnan(X_test_model), axis=1)
        test_probas = np.full(X_test_model.shape[0], 0.5, dtype=np.float64)
        if (~nan_mask).sum() > 0:
            test_probas[~nan_mask] = _predict_proba(
                model, load_type, X_test_model[~nan_mask]
            )
        test_scores = _to_vqi_scores(test_probas)

        # Save test predictions
        np.save(os.path.join(X1_DIR, f"{name}{suffix}_test_{ds}_predictions.npy"),
                test_probas)

        # ERC + DET per provider
        erc_results[ds] = {}
        det_results[ds] = {}

        for prov in PROVIDERS:
            pair_sims = _load_pair_scores(ds, prov)
            erc_r = compute_test_erc(test_scores, pairs, pair_labels, pair_sims)
            det_r = compute_test_det(test_scores, pairs, pair_labels, pair_sims)
            erc_results[ds][prov] = erc_r
            det_results[ds][prov] = det_r

        # Log best ERC reduction for this dataset (P1, fnmr10, 30% reject)
        p1_erc = erc_results[ds].get("P1_ECAPA")
        if p1_erc and "fnmr10" in p1_erc:
            red = p1_erc["fnmr10"].get("red_30pct", 0)
            logger.info("  %s P1 ERC@30%%rej: %.1f%% reduction", ds, red)

    _flush()

    # E) Inference speed (use first 10 val samples)
    X_speed = X_val_model[:10]
    speed = measure_inference_speed(model, load_type, X_speed, n_repeats=500)
    logger.info("  Speed: %.4f ms/sample (mean)", speed["ms_per_sample_mean"])
    _flush()

    return {
        "model": name,
        "score_type": score_type,
        **cls_metrics,
        **speed,
        "erc": erc_results,
        "det": det_results,
    }


def evaluate_all(score_type, resume_from=None):
    """Evaluate all 7 models for a given score type."""
    logger.info("=" * 70)
    logger.info("Evaluating all models for %s", "VQI-S" if score_type == "s" else "VQI-V")
    logger.info("=" * 70)
    _flush()

    # Load data
    X_val, y_val = load_validation_data(score_type)
    logger.info("Validation: X=%s, y=%s (class0=%d, class1=%d)",
                X_val.shape, y_val.shape,
                int((y_val == 0).sum()), int((y_val == 1).sum()))

    scaler = _load_scaler(score_type)

    # Load test data
    test_data = {}
    for ds in TEST_DATASETS:
        X_test = load_test_features(score_type, ds)
        pairs, labels = load_test_pairs(ds)
        test_data[ds] = (X_test, pairs, labels)
        logger.info("Test %s: X=%s, pairs=%s", ds, X_test.shape, pairs.shape)
    _flush()

    # Determine which models to skip on resume
    checkpoint_path = os.path.join(X1_DIR, f"_eval_checkpoint{'_v' if score_type == 'v' else ''}.json")
    completed_models = set()
    if resume_from and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            completed_models = set(json.load(f).get("completed", []))
        logger.info("Resuming — skipping: %s", completed_models)

    all_results = []
    suffix = "_v" if score_type == "v" else ""

    # Load any existing partial results
    metrics_path = os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv")
    if resume_from and os.path.exists(metrics_path):
        existing_df = pd.read_csv(metrics_path)
        for _, row in existing_df.iterrows():
            all_results.append(row.to_dict())

    for name in MODEL_ORDER:
        if name in completed_models:
            logger.info("Skipping %s (already evaluated)", name)
            continue

        result = evaluate_one_model(name, score_type, X_val, y_val, scaler, test_data)

        # Extract scalar metrics for CSV (exclude nested dicts)
        scalar = {k: v for k, v in result.items()
                  if not isinstance(v, dict)}
        all_results.append(scalar)

        # Save detailed ERC/DET results as JSON
        detail_path = os.path.join(X1_DIR, f"{name}{suffix}_eval_detail.json")
        detail = {
            "model": name,
            "score_type": score_type,
            "erc": result["erc"],
            "det": result["det"],
        }
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, default=str)

        # Save partial CSV
        df = pd.DataFrame([r for r in all_results if isinstance(r, dict)])
        df.to_csv(metrics_path, index=False)

        # Update checkpoint
        completed_models.add(name)
        with open(checkpoint_path, "w") as f:
            json.dump({"completed": list(completed_models)}, f)

        logger.info("Checkpoint: %s done (%d/%d)", name, len(completed_models), len(MODEL_ORDER))
        _flush()

    # Final save
    df = pd.DataFrame([r for r in all_results if isinstance(r, dict)])
    df.to_csv(metrics_path, index=False)

    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logger.info("Saved: %s (%d models)", metrics_path, len(df))
    _flush()

    return df


def print_summary(score_type):
    """Print a formatted comparison table."""
    suffix = "_v" if score_type == "v" else ""
    metrics_path = os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv")
    df = pd.read_csv(metrics_path)

    label = "VQI-S" if score_type == "s" else "VQI-V"
    print(f"\n{'=' * 90}")
    print(f"X1.3 Model Comparison — {label}")
    print(f"{'=' * 90}")
    print(f"{'Model':<10} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Acc':>8} "
          f"{'Brier':>8} {'ms/samp':>8}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
              f"{row['f1']:>8.4f} {row['accuracy']:>8.4f} "
              f"{row['brier_score']:>8.4f} {row['ms_per_sample_mean']:>8.4f}")
    print("=" * 90)

    # Print ERC summary from detail JSON
    print(f"\nERC Summary (FNMR reduction % at 30% rejection, P1 FNMR@10%):")
    print(f"{'Model':<10}", end="")
    for ds in TEST_DATASETS:
        print(f" {ds:>12}", end="")
    print()
    print("-" * (10 + 13 * len(TEST_DATASETS)))

    for name in MODEL_ORDER:
        detail_path = os.path.join(X1_DIR, f"{name}{suffix}_eval_detail.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path, "r") as f:
            detail = json.load(f)
        print(f"{name:<10}", end="")
        for ds in TEST_DATASETS:
            erc = detail["erc"].get(ds, {}).get("P1_ECAPA", {})
            fnmr10 = erc.get("fnmr10", {})
            red = fnmr10.get("red_30pct", 0) if fnmr10 else 0
            print(f" {red:>11.1f}%", end="")
        print()

    # Print DET EER separation summary
    print(f"\nDET EER Separation (bottom/top, P1, higher=better):")
    print(f"{'Model':<10}", end="")
    for ds in TEST_DATASETS:
        print(f" {ds:>12}", end="")
    print()
    print("-" * (10 + 13 * len(TEST_DATASETS)))

    for name in MODEL_ORDER:
        detail_path = os.path.join(X1_DIR, f"{name}{suffix}_eval_detail.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path, "r") as f:
            detail = json.load(f)
        print(f"{name:<10}", end="")
        for ds in TEST_DATASETS:
            det = detail["det"].get(ds, {}).get("P1_ECAPA", {})
            sep = det.get("eer_separation")
            if sep is not None:
                print(f" {float(sep):>11.2f}x", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="X1.3: Evaluate all models")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(X1_DIR, exist_ok=True)

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]

    for st in score_types:
        evaluate_all(st, resume_from=args.resume)
        print_summary(st)

    print("\n" + "=" * 70)
    print("X1.3 Evaluation COMPLETE")
    print("=" * 70)
