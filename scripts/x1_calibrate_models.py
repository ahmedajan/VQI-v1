"""
Step X1.11: Post-Hoc Calibration

Applies isotonic regression calibration to all 7 classifiers (M1-M7) for both
VQI-S and VQI-V. Calibration maps raw predict_proba outputs to better-calibrated
probabilities, potentially fixing the extreme score distribution problem.

Metrics: ECE, MCE, Brier score before and after calibration.

Usage:
    python scripts/x1_calibrate_models.py [--score-type s|v|both]
"""

import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.x1_prepare_data import load_validation_data

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "x1")

MODEL_ORDER = ["rf", "xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"]
MODEL_LABELS = {
    "rf": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "logreg": "LogReg", "svm": "SVM", "mlp": "MLP", "tabnet": "TabNet",
}
MODEL_COLORS = {
    "rf": "#1f77b4", "xgboost": "#ff7f0e", "lightgbm": "#2ca02c",
    "logreg": "#d62728", "svm": "#9467bd", "mlp": "#8c564b", "tabnet": "#e377c2",
}

logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _savefig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", os.path.basename(path))


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(y_true))


def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


def calibrate_model(name, score_type, y_val, val_probas):
    """Fit isotonic regression calibrator and compute before/after metrics."""
    suffix = "_v" if score_type == "v" else ""

    # Before calibration
    before = {
        "ece": compute_ece(y_val, val_probas),
        "mce": compute_mce(y_val, val_probas),
        "brier": float(brier_score_loss(y_val, val_probas)),
    }

    # Fit isotonic regression
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_probas, y_val)

    # Calibrated predictions
    cal_probas = calibrator.predict(val_probas)

    # After calibration
    after = {
        "ece": compute_ece(y_val, cal_probas),
        "mce": compute_mce(y_val, cal_probas),
        "brier": float(brier_score_loss(y_val, cal_probas)),
    }

    # Save calibrator
    cal_path = os.path.join(MODELS_DIR, f"x1_calibrator_{name}{suffix}.joblib")
    joblib.dump(calibrator, cal_path)

    # Save metrics
    for label, metrics in [("before", before), ("after", after)]:
        path = os.path.join(X1_DIR, f"calibration_{name}{suffix}_{label}.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Save calibrated predictions
    np.save(os.path.join(X1_DIR, f"{name}{suffix}_val_predictions_calibrated.npy"), cal_probas)

    logger.info("  %s: ECE %.4f->%.4f, MCE %.4f->%.4f, Brier %.4f->%.4f",
                MODEL_LABELS[name],
                before["ece"], after["ece"],
                before["mce"], after["mce"],
                before["brier"], after["brier"])

    return before, after, cal_probas


def plot_calibration_before_after(score_type, y_val, raw_preds, cal_preds):
    """Plot reliability diagrams before and after calibration."""
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, name in enumerate(MODEL_ORDER):
        # Before
        ax = axes[0][i] if i < 4 else axes[1][i - 4]
        if i >= 4:
            ax = axes[1][i - 4]
        else:
            ax = axes[0][i]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob_true_raw, prob_pred_raw = calibration_curve(y_val, raw_preds[name], n_bins=10)
            prob_true_cal, prob_pred_cal = calibration_curve(y_val, cal_preds[name], n_bins=10)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.plot(prob_pred_raw, prob_true_raw, "o-", color="red", alpha=0.6, label="Before", linewidth=1.5)
        ax.plot(prob_pred_cal, prob_true_cal, "s-", color="blue", alpha=0.8, label="After", linewidth=1.5)
        ax.set_title(MODEL_LABELS[name], fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if i == 0 or i == 4:
            ax.set_ylabel("Fraction Positive")
        if i >= 3:
            ax.set_xlabel("Mean Predicted Prob")

    # Hide unused subplot
    axes[1][3].set_visible(False)

    fig.suptitle(f"Calibration Before/After — {label}", fontsize=14)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_calibration_before_after.png"))


def plot_score_distributions_calibrated(score_type, cal_preds):
    """Plot score distributions after calibration."""
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, name in enumerate(MODEL_ORDER):
        ax = axes[i]
        scores = np.clip(np.round(cal_preds[name] * 100), 0, 100).astype(int)
        ax.hist(scores, bins=np.arange(0, 102, 5), color=MODEL_COLORS[name],
                alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{MODEL_LABELS[name]} (calibrated)", fontsize=11)
        ax.set_xlabel("VQI Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

    axes[7].set_visible(False)
    fig.suptitle(f"Score Distributions After Calibration — {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_score_distributions_calibrated.png"))


def calibrate_all(score_type):
    """Calibrate all 7 models for a given score type."""
    suffix = "_v" if score_type == "v" else ""
    label = "VQI-S" if score_type == "s" else "VQI-V"

    logger.info("=" * 60)
    logger.info("Calibrating all models for %s", label)
    logger.info("=" * 60)

    X_val, y_val = load_validation_data(score_type)

    raw_preds = {}
    cal_preds = {}
    results = []

    for name in MODEL_ORDER:
        pred_path = os.path.join(X1_DIR, f"{name}{suffix}_val_predictions.npy")
        val_probas = np.load(pred_path)
        raw_preds[name] = val_probas

        before, after, cal_probas = calibrate_model(name, score_type, y_val, val_probas)
        cal_preds[name] = cal_probas

        results.append({
            "model": name,
            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "mce_before": before["mce"],
            "mce_after": after["mce"],
            "brier_before": before["brier"],
            "brier_after": after["brier"],
            "ece_improvement": before["ece"] - after["ece"],
        })

    _flush()

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(X1_DIR, f"calibration_summary{suffix}.csv"), index=False)

    # Plots
    logger.info("Generating calibration plots...")
    plot_calibration_before_after(score_type, y_val, raw_preds, cal_preds)
    plot_score_distributions_calibrated(score_type, cal_preds)

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Calibration Summary — {label}")
    print(f"{'=' * 80}")
    print(f"{'Model':<10} {'ECE Before':>10} {'ECE After':>10} {'MCE Before':>10} {'MCE After':>10} {'Brier Bef':>10} {'Brier Aft':>10}")
    print("-" * 80)
    for r in results:
        print(f"{MODEL_LABELS[r['model']]:<10} {r['ece_before']:>10.4f} {r['ece_after']:>10.4f} "
              f"{r['mce_before']:>10.4f} {r['mce_after']:>10.4f} "
              f"{r['brier_before']:>10.4f} {r['brier_after']:>10.4f}")
    print("=" * 80)

    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="X1.11: Calibrate all models")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]
    for st in score_types:
        calibrate_all(st)

    print("\n" + "=" * 60)
    print("X1.11 Post-Hoc Calibration COMPLETE")
    print("=" * 60)
