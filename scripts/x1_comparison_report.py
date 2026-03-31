"""
Step X1.4: Generate Comparison Report

Produces a comprehensive markdown report with visualizations comparing all 7 models
for both VQI-S and VQI-V. Generates ~40 plots covering ROC, score distributions,
calibration, ERC overlays, DET, inference speed, and feature importance.

Usage:
    python scripts/x1_comparison_report.py
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
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.x1_prepare_data import (
    TEST_DATASETS,
    load_test_features,
    load_test_pairs,
    load_validation_data,
)
from vqi.evaluation.erc import compute_erc, compute_pairwise_quality, find_tau_for_fnmr

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "x1")
TEST_SCORES_DIR = os.path.join(DATA_DIR, "step8", "full_feature", "test_scores")

MODEL_ORDER = ["rf", "xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"]
MODEL_LABELS = {
    "rf": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "logreg": "LogReg", "svm": "SVM", "mlp": "MLP", "tabnet": "TabNet",
}
MODEL_COLORS = {
    "rf": "#1f77b4", "xgboost": "#ff7f0e", "lightgbm": "#2ca02c",
    "logreg": "#d62728", "svm": "#9467bd", "mlp": "#8c564b", "tabnet": "#e377c2",
}
PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]

logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _savefig(fig, path, dpi=150):
    """Save figure and close."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", os.path.basename(path))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_val_predictions(score_type):
    """Load validation predictions for all models."""
    suffix = "_v" if score_type == "v" else ""
    preds = {}
    for name in MODEL_ORDER:
        path = os.path.join(X1_DIR, f"{name}{suffix}_val_predictions.npy")
        if os.path.exists(path):
            preds[name] = np.load(path)
    return preds


def load_test_predictions(score_type, dataset):
    """Load test predictions for all models for one dataset."""
    suffix = "_v" if score_type == "v" else ""
    preds = {}
    for name in MODEL_ORDER:
        path = os.path.join(X1_DIR, f"{name}{suffix}_test_{dataset}_predictions.npy")
        if os.path.exists(path):
            preds[name] = np.load(path)
    return preds


def load_pair_scores(dataset, provider):
    """Load pair similarity scores."""
    path = os.path.join(TEST_SCORES_DIR, f"pair_scores_{dataset}_{provider}.csv")
    df = pd.read_csv(path)
    return df["cos_sim_snorm"].values


# ---------------------------------------------------------------------------
# Plot 1: ROC Curve Overlay
# ---------------------------------------------------------------------------
def plot_roc_overlay(score_type, y_val, val_preds):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    for name in MODEL_ORDER:
        if name not in val_preds:
            continue
        fpr, tpr, _ = roc_curve(y_val, val_preds[name])
        suffix_v = "_v" if score_type == "v" else ""
        metrics_df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suffix_v}.csv"))
        auc_val = metrics_df.loc[metrics_df["model"] == name, "auc_roc"].values[0]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} (AUC={auc_val:.4f})", linewidth=1.5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve Comparison — {label} (Validation Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_roc_overlay.png"))


# ---------------------------------------------------------------------------
# Plot 2: Score Distribution Comparison
# ---------------------------------------------------------------------------
def plot_score_distributions(score_type, val_preds):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, name in enumerate(MODEL_ORDER):
        if name not in val_preds:
            continue
        ax = axes[i]
        scores = np.clip(np.round(val_preds[name] * 100), 0, 100).astype(int)
        ax.hist(scores, bins=np.arange(0, 102, 5), color=MODEL_COLORS[name],
                alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{MODEL_LABELS[name]}", fontsize=11)
        ax.set_xlabel("VQI Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(MODEL_ORDER) < len(axes):
        for j in range(len(MODEL_ORDER), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle(f"Score Distributions — {label} (Validation Set)", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_score_distributions.png"))


# ---------------------------------------------------------------------------
# Plot 3: Calibration Comparison
# ---------------------------------------------------------------------------
def plot_calibration(score_type, y_val, val_preds):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")

    for name in MODEL_ORDER:
        if name not in val_preds:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob_true, prob_pred = calibration_curve(y_val, val_preds[name], n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", color=MODEL_COLORS[name],
                label=MODEL_LABELS[name], linewidth=1.5, markersize=5)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Comparison — {label}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_calibration_comparison.png"))


# ---------------------------------------------------------------------------
# Plot 4: Inference Speed
# ---------------------------------------------------------------------------
def plot_inference_speed(score_type):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"
    suffix = "_v" if score_type == "v" else ""

    df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv"))
    df = df.set_index("model").loc[MODEL_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [MODEL_LABELS[m] for m in df["model"]],
        df["ms_per_sample_mean"],
        color=[MODEL_COLORS[m] for m in df["model"]],
        edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(bars, df["ms_per_sample_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("ms / sample")
    ax.set_title(f"Inference Speed — {label}")
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_inference_speed.png"))


# ---------------------------------------------------------------------------
# Plot 5: ERC Overlay per dataset+provider
# ---------------------------------------------------------------------------
def plot_erc_overlays(score_type):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"
    suffix = "_v" if score_type == "v" else ""

    # Load test data
    X_val, y_val = load_validation_data(score_type)

    for ds in TEST_DATASETS:
        X_test = load_test_features(score_type, ds)
        pairs, pair_labels = load_test_pairs(ds)
        test_preds = load_test_predictions(score_type, ds)

        for prov in ["P1_ECAPA", "P3_ECAPA2"]:  # Focus on P1 and P3
            pair_sims = load_pair_scores(ds, prov)

            gen_mask = pair_labels == 1
            imp_mask = pair_labels == 0
            genuine_sim = pair_sims[gen_mask]
            impostor_sim = pair_sims[imp_mask]

            if len(genuine_sim) == 0 or len(impostor_sim) == 0:
                continue

            for target_fnmr, fnmr_label in [(0.10, "FNMR10")]:
                tau = find_tau_for_fnmr(genuine_sim, impostor_sim, target_fnmr)

                fig, ax = plt.subplots(figsize=(8, 6))

                for name in MODEL_ORDER:
                    if name not in test_preds:
                        continue
                    scores = np.clip(np.round(test_preds[name] * 100), 0, 100).astype(int)
                    quality = compute_pairwise_quality(scores.astype(float), pairs)
                    q_gen = quality[gen_mask]
                    q_imp = quality[imp_mask]

                    erc = compute_erc(genuine_sim, impostor_sim, q_gen, q_imp, tau)
                    ax.plot(erc["reject_fracs"], erc["fnmr_values"],
                            color=MODEL_COLORS[name], label=MODEL_LABELS[name],
                            linewidth=1.5)

                ax.set_xlabel("Fraction Rejected")
                ax.set_ylabel("FNMR")
                ax.set_title(f"ERC — {label} / {ds} / {prov} / {fnmr_label}")
                ax.legend(fontsize=8)
                ax.set_xlim(0, 0.8)
                ax.grid(True, alpha=0.3)
                _savefig(fig, os.path.join(
                    REPORTS_DIR,
                    f"{prefix}_erc_{ds}_{prov}_{fnmr_label.lower()}.png"
                ))


# ---------------------------------------------------------------------------
# Plot 6: Feature Importance Comparison
# ---------------------------------------------------------------------------
def plot_feature_importance(score_type):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"
    suffix = "_v" if score_type == "v" else ""
    model_prefix = "vqi_v" if score_type == "v" else "vqi"

    # Load feature names
    sv = "s" if score_type == "s" else "v"
    feat_names_path = os.path.join(DATA_DIR, "step4", "features", f"feature_names_{sv}.json")
    with open(feat_names_path, "r") as f:
        all_names = json.load(f)

    eval_dir = "evaluation" if score_type == "s" else "evaluation_v"
    sel_path = os.path.join(DATA_DIR, "step5", eval_dir, "selected_features.txt")
    with open(sel_path, "r") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    top_k = 20

    fig, axes = plt.subplots(1, 4, figsize=(22, 8))

    # RF Gini importance
    rf_model = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_rf_model.joblib"))
    imp_rf = rf_model.feature_importances_
    top_idx = np.argsort(imp_rf)[-top_k:]
    axes[0].barh([selected_names[i] for i in top_idx], imp_rf[top_idx], color=MODEL_COLORS["rf"])
    axes[0].set_title("RF (Gini)")
    axes[0].set_xlabel("Importance")

    # XGBoost gain
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, f"{model_prefix}_xgboost_model.json"))
    imp_xgb = xgb_model.feature_importances_
    top_idx = np.argsort(imp_xgb)[-top_k:]
    axes[1].barh([selected_names[i] for i in top_idx], imp_xgb[top_idx], color=MODEL_COLORS["xgboost"])
    axes[1].set_title("XGBoost (Gain)")
    axes[1].set_xlabel("Importance")

    # LightGBM gain
    lgb_model = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_lightgbm_model.joblib"))
    imp_lgb = lgb_model.feature_importances_
    imp_lgb = imp_lgb / imp_lgb.sum()  # normalize
    top_idx = np.argsort(imp_lgb)[-top_k:]
    axes[2].barh([selected_names[i] for i in top_idx], imp_lgb[top_idx], color=MODEL_COLORS["lightgbm"])
    axes[2].set_title("LightGBM (Gain)")
    axes[2].set_xlabel("Importance")

    # LogReg coefficients
    lr_model = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_logreg_model.joblib"))
    coef = np.abs(lr_model.coef_[0])
    top_idx = np.argsort(coef)[-top_k:]
    axes[3].barh([selected_names[i] for i in top_idx], coef[top_idx], color=MODEL_COLORS["logreg"])
    axes[3].set_title("LogReg (|Coefficient|)")
    axes[3].set_xlabel("Importance")

    fig.suptitle(f"Feature Importance Comparison — {label} (Top {top_k})", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_feature_importance.png"))


# ---------------------------------------------------------------------------
# Summary metrics bar chart
# ---------------------------------------------------------------------------
def plot_metrics_bars(score_type):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"
    suffix = "_v" if score_type == "v" else ""

    df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv"))
    df = df.set_index("model").loc[MODEL_ORDER].reset_index()

    metrics = [("auc_roc", "AUC-ROC"), ("f1", "F1"), ("accuracy", "Accuracy"), ("brier_score", "Brier Score")]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(
            [MODEL_LABELS[m] for m in df["model"]],
            df[col],
            color=[MODEL_COLORS[m] for m in df["model"]],
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Adjust y-axis for readability
        if col != "brier_score":
            ymin = max(0, df[col].min() - 0.05)
            ax.set_ylim(ymin, min(1, df[col].max() + 0.03))

    fig.suptitle(f"Classification Metrics — {label}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_metrics_comparison.png"))


# ---------------------------------------------------------------------------
# Confusion matrix grid
# ---------------------------------------------------------------------------
def plot_confusion_matrices(score_type):
    prefix = "s" if score_type == "s" else "v"
    label = "VQI-S" if score_type == "s" else "VQI-V"
    suffix = "_v" if score_type == "v" else ""

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, name in enumerate(MODEL_ORDER):
        ax = axes[i]
        cm_path = os.path.join(X1_DIR, f"{name}{suffix}_confusion_matrix.npy")
        if not os.path.exists(cm_path):
            ax.set_visible(False)
            continue
        cm = np.load(cm_path)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        fontsize=12, color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_title(MODEL_LABELS[name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    if len(MODEL_ORDER) < len(axes):
        for j in range(len(MODEL_ORDER), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(REPORTS_DIR, f"{prefix}_confusion_matrices.png"))


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------
def generate_markdown_report():
    """Generate the full comparison report."""
    lines = []
    lines.append("# X1 Model Comparison Report")
    lines.append("")
    lines.append("Comprehensive evaluation of 7 classifiers (RF baseline + 6 alternatives)")
    lines.append("for both VQI-S and VQI-V on held-out validation and test sets.")
    lines.append("")

    for score_type, label in [("s", "VQI-S"), ("v", "VQI-V")]:
        suffix = "_v" if score_type == "v" else ""
        prefix = "s" if score_type == "s" else "v"
        df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv"))
        df = df.set_index("model").loc[MODEL_ORDER].reset_index()
        df_sorted = df.sort_values("auc_roc", ascending=False)

        lines.append(f"## {label}")
        lines.append("")

        # Summary table
        lines.append(f"### 1. Summary Table (ranked by AUC-ROC)")
        lines.append("")
        lines.append("| Rank | Model | AUC-ROC | AUC-PR | F1 | Accuracy | Brier | ms/sample |")
        lines.append("|------|-------|---------|--------|-----|----------|-------|-----------|")
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            bold = "**" if rank == 1 else ""
            lines.append(
                f"| {rank} | {bold}{MODEL_LABELS[row['model']]}{bold} | "
                f"{bold}{row['auc_roc']:.4f}{bold} | {row['auc_pr']:.4f} | "
                f"{row['f1']:.4f} | {row['accuracy']:.4f} | "
                f"{row['brier_score']:.4f} | {row['ms_per_sample_mean']:.3f} |"
            )
        lines.append("")

        # AUC improvement over RF
        rf_auc = df.loc[df["model"] == "rf", "auc_roc"].values[0]
        lines.append(f"RF baseline AUC-ROC: {rf_auc:.4f}")
        lines.append("")
        lines.append("| Model | AUC-ROC | Improvement over RF |")
        lines.append("|-------|---------|-------------------|")
        for _, row in df_sorted.iterrows():
            diff = row["auc_roc"] - rf_auc
            sign = "+" if diff >= 0 else ""
            meets = " **MEETS 2pp threshold**" if diff >= 0.02 else ""
            lines.append(
                f"| {MODEL_LABELS[row['model']]} | {row['auc_roc']:.4f} | "
                f"{sign}{diff*100:.1f} pp{meets} |"
            )
        lines.append("")

        # Plots
        lines.append(f"### 2. ROC Curve Overlay")
        lines.append(f"![ROC Overlay]({prefix}_roc_overlay.png)")
        lines.append("")

        lines.append(f"### 3. Score Distributions")
        lines.append(f"![Score Distributions]({prefix}_score_distributions.png)")
        lines.append("")

        lines.append(f"### 4. Metrics Comparison")
        lines.append(f"![Metrics]({prefix}_metrics_comparison.png)")
        lines.append("")

        lines.append(f"### 5. Calibration")
        lines.append(f"![Calibration]({prefix}_calibration_comparison.png)")
        lines.append("")

        lines.append(f"### 6. Confusion Matrices")
        lines.append(f"![Confusion Matrices]({prefix}_confusion_matrices.png)")
        lines.append("")

        lines.append(f"### 7. Inference Speed")
        lines.append(f"![Speed]({prefix}_inference_speed.png)")
        lines.append("")

        lines.append(f"### 8. Feature Importance")
        lines.append(f"![Feature Importance]({prefix}_feature_importance.png)")
        lines.append("")

        lines.append(f"### 9. ERC Curves (sample — VoxCeleb1, P1)")
        lines.append(f"![ERC VoxCeleb1]({prefix}_erc_voxceleb1_P1_ECAPA_fnmr10.png)")
        lines.append("")

        # ERC summary table
        lines.append(f"### 10. ERC FNMR Reduction Summary (P1, FNMR@10%, 30% rejection)")
        lines.append("")
        lines.append("| Model | " + " | ".join(TEST_DATASETS) + " |")
        lines.append("|-------" + "|------" * len(TEST_DATASETS) + "|")

        for name in MODEL_ORDER:
            detail_path = os.path.join(X1_DIR, f"{name}{suffix}_eval_detail.json")
            if not os.path.exists(detail_path):
                continue
            with open(detail_path, "r") as f:
                detail = json.load(f)
            row_parts = [f"{MODEL_LABELS[name]}"]
            for ds in TEST_DATASETS:
                erc = detail["erc"].get(ds, {}).get("P1_ECAPA", {})
                fnmr10 = erc.get("fnmr10", {})
                red = fnmr10.get("red_30pct", 0) if fnmr10 else 0
                row_parts.append(f"{red:.1f}%")
            lines.append("| " + " | ".join(row_parts) + " |")
        lines.append("")

    # Recommendation section
    lines.append("## Recommendation")
    lines.append("")

    for score_type, label in [("s", "VQI-S"), ("v", "VQI-V")]:
        suffix = "_v" if score_type == "v" else ""
        df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suffix}.csv"))
        df = df.set_index("model").loc[MODEL_ORDER].reset_index()
        best = df.loc[df["auc_roc"].idxmax()]
        rf = df.loc[df["model"] == "rf"].iloc[0]
        diff = best["auc_roc"] - rf["auc_roc"]

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- **Best AUC-ROC:** {MODEL_LABELS[best['model']]} ({best['auc_roc']:.4f}), "
                     f"+{diff*100:.1f} pp over RF ({rf['auc_roc']:.4f})")
        lines.append(f"- **Best calibration (Brier):** {MODEL_LABELS[df.loc[df['brier_score'].idxmin(), 'model']]} "
                     f"({df['brier_score'].min():.4f})")
        fastest = df.loc[df["ms_per_sample_mean"].idxmin()]
        lines.append(f"- **Fastest:** {MODEL_LABELS[fastest['model']]} ({fastest['ms_per_sample_mean']:.3f} ms)")
        lines.append(f"- **Most interpretable:** RF (feature importances, decision paths)")
        lines.append("")

        meets_threshold = diff >= 0.02
        speed_ok = best["ms_per_sample_mean"] < 50
        if meets_threshold and speed_ok:
            lines.append(f"**Recommendation:** Change default to **{MODEL_LABELS[best['model']]}** "
                        f"(+{diff*100:.1f} pp AUC, speed {best['ms_per_sample_mean']:.1f} ms). "
                        f"Meets both the 2 pp AUC threshold and <50ms speed requirement.")
        else:
            lines.append(f"**Recommendation:** Keep RF as default. "
                        f"{'AUC improvement below 2pp threshold.' if not meets_threshold else ''} "
                        f"{'Speed exceeds 50ms.' if not speed_ok else ''}")
        lines.append("")

    # Cross-score comparison
    lines.append("## Cross-Score Comparison")
    lines.append("")
    lines.append("| Score | Best Model | AUC-ROC |")
    lines.append("|-------|-----------|---------|")
    for st, lbl in [("s", "VQI-S"), ("v", "VQI-V")]:
        suf = "_v" if st == "v" else ""
        df = pd.read_csv(os.path.join(X1_DIR, f"comparison_metrics{suf}.csv"))
        best = df.loc[df["auc_roc"].idxmax()]
        lines.append(f"| {lbl} | {MODEL_LABELS[best['model']]} | {best['auc_roc']:.4f} |")
    lines.append("")

    report_path = os.path.join(REPORTS_DIR, "model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved report: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    for score_type in ["s", "v"]:
        label = "VQI-S" if score_type == "s" else "VQI-V"
        logger.info("=" * 60)
        logger.info("Generating plots for %s", label)
        logger.info("=" * 60)
        _flush()

        X_val, y_val = load_validation_data(score_type)
        val_preds = load_val_predictions(score_type)

        logger.info("Generating ROC overlay...")
        plot_roc_overlay(score_type, y_val, val_preds)

        logger.info("Generating score distributions...")
        plot_score_distributions(score_type, val_preds)

        logger.info("Generating calibration comparison...")
        plot_calibration(score_type, y_val, val_preds)

        logger.info("Generating inference speed chart...")
        plot_inference_speed(score_type)

        logger.info("Generating metrics bars...")
        plot_metrics_bars(score_type)

        logger.info("Generating confusion matrices...")
        plot_confusion_matrices(score_type)

        logger.info("Generating feature importance...")
        plot_feature_importance(score_type)

        logger.info("Generating ERC overlays...")
        plot_erc_overlays(score_type)

    logger.info("=" * 60)
    logger.info("Generating markdown report...")
    generate_markdown_report()

    # Count outputs
    n_plots = len([f for f in os.listdir(REPORTS_DIR) if f.endswith(".png")])
    logger.info("Total: %d plots + 1 report", n_plots)
    logger.info("X1.4 Report Generation COMPLETE")
    logger.info("=" * 60)
