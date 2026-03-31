"""Step 4 Visualization Script.

Generates 22 plots + analysis.md for Step 4 feature extraction results.
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gaussian_kde

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "step4", "features")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "step4")
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "step2", "training_set_final.csv")


def load_data():
    """Load feature arrays and labels."""
    feat_s = np.load(os.path.join(FEATURES_DIR, "features_s_train.npy"))
    feat_v = np.load(os.path.join(FEATURES_DIR, "features_v_train.npy"))
    with open(os.path.join(FEATURES_DIR, "feature_names_s.json"), encoding="utf-8") as f:
        names_s = json.load(f)
    with open(os.path.join(FEATURES_DIR, "feature_names_v.json"), encoding="utf-8") as f:
        names_v = json.load(f)
    train_df = pd.read_csv(TRAIN_CSV)
    labels = train_df["label"].values
    return feat_s, feat_v, names_s, names_v, labels


def cohens_d(x0, x1):
    """Cohen's d effect size."""
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return 0.0
    var0, var1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(x1) - np.mean(x0)) / pooled_std


def plot_1_frame_distributions(feat_s, names_s, labels):
    """Plot 1: Frame feature distributions (23 features, mean stat only)."""
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    axes = axes.ravel()
    frame_mean_indices = [i * 19 + 10 for i in range(23)]  # Mean is at offset 10

    for k, idx in enumerate(frame_mean_indices):
        ax = axes[k]
        name = names_s[idx]
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        ax.hist(c0, bins=50, alpha=0.5, label="Class 0", density=True, color="red")
        ax.hist(c1, bins=50, alpha=0.5, label="Class 1", density=True, color="blue")
        ax.set_title(name.replace("_Mean", ""), fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(23, 25):
        axes[k].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Frame Feature Distributions (Mean Statistic, Class 0 vs 1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_distributions_frame.png"), dpi=150)
    plt.close()


def plot_2_global_distributions(feat_s, names_s, labels):
    """Plot 2: Global feature distributions."""
    global_start = 437
    n_global = 107
    n_cols = 12
    n_rows = (n_global + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 2))
    axes = axes.ravel()

    for k in range(n_global):
        idx = global_start + k
        ax = axes[k]
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        # Remove NaN
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        if len(c0) > 0 and len(c1) > 0:
            ax.hist(c0, bins=30, alpha=0.5, density=True, color="red")
            ax.hist(c1, bins=30, alpha=0.5, density=True, color="blue")
        ax.set_title(names_s[idx][:15], fontsize=5)
        ax.tick_params(labelsize=4)
    for k in range(n_global, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("Global Feature Distributions (Class 0=red, Class 1=blue)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_distributions_global.png"), dpi=150)
    plt.close()


def plot_3_correlation_matrix(feat_s, names_s):
    """Plot 3: Spearman correlation heatmap (top 100 by variance)."""
    # Select top 100 features by variance
    variances = np.var(feat_s, axis=0)
    top_indices = np.argsort(variances)[-100:]
    top_names = [names_s[i][:12] for i in top_indices]
    top_feats = feat_s[:, top_indices]

    # Replace non-finite
    top_feats = np.nan_to_num(top_feats, nan=0.0, posinf=0.0, neginf=0.0)

    corr, _ = spearmanr(top_feats)
    if corr.ndim == 0:
        return
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman Correlation (Top 100 Features by Variance)", fontsize=14)
    ax.set_xticks(range(len(top_names)))
    ax.set_xticklabels(top_names, rotation=90, fontsize=3)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_correlation_matrix.png"), dpi=150)
    plt.close()


def plot_4_nan_inf_report(feat_s, feat_v, names_s, names_v):
    """Plot 4: NaN/Inf report table."""
    nan_counts_s = np.sum(~np.isfinite(feat_s), axis=0)
    nan_counts_v = np.sum(~np.isfinite(feat_v), axis=0)

    # Only show features with NaN
    nan_s = [(names_s[i], nan_counts_s[i]) for i in range(len(names_s)) if nan_counts_s[i] > 0]
    nan_v = [(names_v[i], nan_counts_v[i]) for i in range(len(names_v)) if nan_counts_v[i] > 0]

    fig, ax = plt.subplots(figsize=(10, max(4, len(nan_s) * 0.3 + len(nan_v) * 0.3 + 2)))
    ax.axis("off")

    text = "NaN/Inf Report\n" + "=" * 40 + "\n\n"
    text += f"VQI-S: {np.sum(nan_counts_s)} total NaN/Inf across {len(nan_s)} features\n"
    text += f"VQI-V: {np.sum(nan_counts_v)} total NaN/Inf across {len(nan_v)} features\n\n"

    if nan_s:
        text += "VQI-S Features with NaN:\n"
        for name, count in sorted(nan_s, key=lambda x: -x[1])[:20]:
            text += f"  {name}: {count}\n"
    if nan_v:
        text += "\nVQI-V Features with NaN:\n"
        for name, count in sorted(nan_v, key=lambda x: -x[1])[:20]:
            text += f"  {name}: {count}\n"
    if not nan_s and not nan_v:
        text += "No NaN/Inf values found - all features are clean!\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace")
    plt.savefig(os.path.join(REPORTS_DIR, "feature_nan_inf_report.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_6_class_separation(feat_s, names_s, labels):
    """Plot 6: Cohen's d class separation table."""
    d_values = []
    for i in range(feat_s.shape[1]):
        c0 = feat_s[labels == 0, i]
        c1 = feat_s[labels == 1, i]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        d = cohens_d(c0, c1)
        d_values.append((names_s[i], abs(d), d))

    d_values.sort(key=lambda x: -x[1])
    top30 = d_values[:30]

    fig, ax = plt.subplots(figsize=(12, 10))
    names_plot = [t[0][:25] for t in top30]
    values = [t[1] for t in top30]
    colors = ["green" if t[2] > 0 else "red" for t in top30]
    ax.barh(range(len(top30)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Top 30 Features by Class Separation (|Cohen's d|)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_class_separation_table.png"), dpi=150)
    plt.close()
    return d_values


def plot_7_value_ranges(feat_s, names_s):
    """Plot 7: Box plot of feature value ranges."""
    # Select 30 most variable features
    variances = np.var(feat_s, axis=0)
    top_idx = np.argsort(variances)[-30:]
    top_names = [names_s[i][:15] for i in top_idx]
    top_feats = feat_s[:, top_idx]
    top_feats = np.nan_to_num(top_feats, nan=0.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.boxplot(top_feats, vert=True, labels=top_names,
               showfliers=False, whis=[5, 95])
    ax.set_xticklabels(top_names, rotation=90, fontsize=6)
    ax.set_title("Feature Value Ranges (Top 30 by Variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_value_ranges.png"), dpi=150)
    plt.close()


def plot_8_ridgeline(feat_s, names_s, labels, d_values):
    """Plot 8: Ridgeline plot for top 30 features by Cohen's d."""
    top30 = d_values[:30]

    fig, axes = plt.subplots(30, 1, figsize=(10, 20), sharex=False)
    for k, (name, abs_d, d) in enumerate(top30):
        idx = names_s.index(name)
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        ax = axes[k]
        ax.hist(c0, bins=50, alpha=0.5, density=True, color="red", label="C0")
        ax.hist(c1, bins=50, alpha=0.5, density=True, color="blue", label="C1")
        ax.set_ylabel(name[:20], fontsize=5, rotation=0, ha="right")
        ax.set_yticks([])
        ax.tick_params(labelsize=5)
        if k == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Ridgeline: Top 30 Features by |Cohen's d|", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_ridgeline_top30.png"), dpi=150)
    plt.close()


def plot_10_violin(feat_s, names_s, labels, d_values):
    """Plot 10: Violin plots for top 20."""
    top20 = d_values[:20]

    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.ravel()
    for k, (name, abs_d, d) in enumerate(top20):
        idx = names_s.index(name)
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        ax = axes[k]
        parts = ax.violinplot([c0, c1], showmedians=True)
        for pc, color in zip(parts["bodies"], ["red", "blue"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["C0", "C1"], fontsize=8)
        ax.set_title(f"{name[:20]}\nd={d:.2f}", fontsize=7)
    fig.suptitle("Top 20 Features: Violin Plots (Class 0 vs 1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_violin_top20.png"), dpi=150)
    plt.close()


def plot_v_distributions(feat_v, names_v, labels):
    """Plots 15-18: VQI-V distributions by module."""
    modules = [
        ("cepstral", 0, 65, "feature_v_distributions_cepstral.png"),
        ("lp_derived", 65, 98, "feature_v_distributions_lp_derived.png"),
        ("formant_prosodic", 98, 126, "feature_v_distributions_formant_prosodic.png"),
        ("other", 126, 161, "feature_v_distributions_other.png"),
    ]

    for mod_name, start, end, filename in modules:
        n_feats = end - start
        n_cols = min(8, n_feats)
        n_rows = (n_feats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_feats > 1 else np.array([[axes]])
        axes_flat = axes.ravel()

        for k in range(n_feats):
            idx = start + k
            ax = axes_flat[k]
            c0 = feat_v[labels == 0, idx]
            c1 = feat_v[labels == 1, idx]
            c0 = c0[np.isfinite(c0)]
            c1 = c1[np.isfinite(c1)]
            if len(c0) > 0 and len(c1) > 0:
                ax.hist(c0, bins=30, alpha=0.5, density=True, color="red")
                ax.hist(c1, bins=30, alpha=0.5, density=True, color="blue")
            ax.set_title(names_v[idx][:15], fontsize=5)
            ax.tick_params(labelsize=4)
        for k in range(n_feats, len(axes_flat)):
            axes_flat[k].set_visible(False)
        fig.suptitle(f"VQI-V {mod_name} Features (C0=red, C1=blue)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, filename), dpi=150)
        plt.close()


def plot_v_class_separation(feat_v, names_v, labels):
    """Plot 19: VQI-V Cohen's d."""
    d_values = []
    for i in range(feat_v.shape[1]):
        c0 = feat_v[labels == 0, i]
        c1 = feat_v[labels == 1, i]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        d = cohens_d(c0, c1)
        d_values.append((names_v[i], abs(d), d))

    d_values.sort(key=lambda x: -x[1])
    top30 = d_values[:30]

    fig, ax = plt.subplots(figsize=(12, 10))
    names_plot = [t[0][:25] for t in top30]
    values = [t[1] for t in top30]
    ax.barh(range(len(top30)), values, color="purple", alpha=0.7)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("VQI-V: Top 30 Features by Class Separation")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_v_class_separation_table.png"), dpi=150)
    plt.close()
    return d_values


def plot_v_ridgeline(feat_v, names_v, labels, d_values):
    """Plot 20: VQI-V ridgeline top 20."""
    top20 = d_values[:20]
    fig, axes = plt.subplots(20, 1, figsize=(10, 14), sharex=False)
    for k, (name, abs_d, d) in enumerate(top20):
        idx = names_v.index(name)
        c0 = feat_v[labels == 0, idx]
        c1 = feat_v[labels == 1, idx]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        ax = axes[k]
        ax.hist(c0, bins=40, alpha=0.5, density=True, color="red")
        ax.hist(c1, bins=40, alpha=0.5, density=True, color="blue")
        ax.set_ylabel(name[:18], fontsize=5, rotation=0, ha="right")
        ax.set_yticks([])
        ax.tick_params(labelsize=5)
    fig.suptitle("VQI-V Ridgeline: Top 20 by |Cohen's d|", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_v_ridgeline_top20.png"), dpi=150)
    plt.close()


def plot_v_nan_inf(feat_v, names_v):
    """Plot 21: VQI-V NaN/Inf report."""
    nan_counts = np.sum(~np.isfinite(feat_v), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    text = "VQI-V NaN/Inf Report\n" + "=" * 40 + "\n\n"
    nan_feats = [(names_v[i], nan_counts[i]) for i in range(len(names_v)) if nan_counts[i] > 0]
    text += f"Total NaN/Inf: {np.sum(nan_counts)} across {len(nan_feats)} features\n\n"
    if nan_feats:
        for name, count in sorted(nan_feats, key=lambda x: -x[1])[:15]:
            text += f"  {name}: {count}\n"
    else:
        text += "No NaN/Inf values found!\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace")
    plt.savefig(os.path.join(REPORTS_DIR, "feature_v_nan_inf_report.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_v_correlation(feat_v, names_v):
    """Plot 22: VQI-V Spearman correlation heatmap."""
    feat_clean = np.nan_to_num(feat_v, nan=0.0)
    corr, _ = spearmanr(feat_clean)
    if corr.ndim == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("VQI-V Spearman Correlation Matrix (161x161)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_v_correlation_matrix.png"), dpi=150)
    plt.close()


def _compute_d_sorted(feat_s, names_s, labels):
    """Compute Cohen's d for all features, return sorted by |d|."""
    d_values = []
    for i in range(feat_s.shape[1]):
        c0 = feat_s[labels == 0, i]
        c1 = feat_s[labels == 1, i]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        d = cohens_d(c0, c1)
        d_values.append((names_s[i], abs(d), d))
    d_values.sort(key=lambda x: -x[1])
    return d_values


def plot_5_histogram_examples(feat_s, names_s, labels, d_values):
    """Plot 5: Top 5 features by |Cohen's d|, 10-bin side-by-side bar histogram."""
    top5 = d_values[:5]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for k, (name, abs_d, d) in enumerate(top5):
        idx = names_s.index(name)
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        ax = axes[k]
        all_vals = np.concatenate([c0, c1])
        bin_edges = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 11)
        width = (bin_edges[1] - bin_edges[0]) * 0.4
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        h0, _ = np.histogram(c0, bins=bin_edges)
        h1, _ = np.histogram(c1, bins=bin_edges)
        ax.bar(centers - width / 2, h0, width=width, color="red", alpha=0.7, label="Class 0")
        ax.bar(centers + width / 2, h1, width=width, color="blue", alpha=0.7, label="Class 1")
        ax.set_title(f"{name[:22]}\n|d|={abs_d:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Top 5 Features by |Cohen's d| - 10-Bin Histograms", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "histogram_examples.png"), dpi=150)
    plt.close()


def _categorize_feature(name):
    """Map a feature name to a category for radar chart."""
    n = name.lower()
    if "snr" in n or "noise" in n or "click" in n or "dropout" in n or "saturation" in n or "agc" in n:
        return "Noise"
    if "reverb" in n or "rt60" in n or "c50" in n or "srmr" in n:
        return "Reverb"
    if "spectral" in n or "ltas" in n or "alpha" in n or "hammarberg" in n:
        return "Spectral"
    if any(x in n for x in ["sf_", "ss_", "sc_", "scf_", "shr_", "sflux_",
                             "se_", "sbw_", "skurt_", "sskew_"]):
        return "Spectral"
    if "hnr" in n or "nhr" in n or "f0" in n or "pitch" in n or "voice" in n or "unvoiced" in n:
        return "Voice"
    if "jitter" in n or "shimmer" in n or "mdvp" in n or "avqi" in n or "dsi" in n or "csid" in n:
        return "Clinical"
    if "formant" in n or "f1_" in n or "f2_" in n or "f3_" in n or "vocal" in n:
        return "Formant"
    if "tremor" in n:
        return "Tremor"
    if "energy" in n or "power" in n or "onset" in n or "e_" in n or "pc_" in n or "clipping" in n:
        return "Dynamics"
    if "cpp" in n or "naq" in n or "qoq" in n or "hrf" in n or "psp" in n or "gci" in n or "goi" in n or "gne" in n:
        return "Glottal"
    if "h1h2" in n or "h1a3" in n:
        return "Glottal"
    if "mfcc" in n or "delta" in n or "ac_" in n or "autocorr" in n:
        return "Temporal"
    if "zcr" in n or "speech" in n or "pause" in n or "rate" in n or "turn" in n or "continuity" in n or "interrupt" in n:
        return "Prosody"
    if "dnsmos" in n or "nisqa" in n or "sii" in n or "modulation" in n:
        return "Intelligibility"
    if "dc" in n or "hum" in n or "bandwidth" in n or "quantiz" in n or "musical" in n:
        return "Channel"
    if any(x in n for x in ["subband", "low_to_high", "sr_"]):
        return "Sub-band"
    return "Other"


def plot_9_radar_by_category(d_values_s):
    """Plot 9: Spider chart of mean |Cohen's d| per feature category."""
    cat_d = {}
    for name, abs_d, _ in d_values_s:
        cat = _categorize_feature(name)
        cat_d.setdefault(cat, []).append(abs_d)

    categories = sorted(cat_d.keys())
    means = [np.mean(cat_d[c]) for c in categories]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    means += means[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, means, "o-", linewidth=2, color="#2563eb")
    ax.fill(angles, means, alpha=0.15, color="#2563eb")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title("Mean |Cohen's d| by Feature Category", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_radar_by_category.png"), dpi=150)
    plt.close()


def plot_11_kde_grid(feat_s, names_s, labels, d_values):
    """Plot 11: KDE class overlay grid for top 30 features."""
    top30 = d_values[:30]
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.ravel()

    for k, (name, abs_d, d) in enumerate(top30):
        idx = names_s.index(name)
        c0 = feat_s[labels == 0, idx]
        c1 = feat_s[labels == 1, idx]
        c0 = c0[np.isfinite(c0)]
        c1 = c1[np.isfinite(c1)]
        ax = axes[k]

        try:
            lo = min(np.percentile(c0, 1), np.percentile(c1, 1))
            hi = max(np.percentile(c0, 99), np.percentile(c1, 99))
            if hi - lo < 1e-12:
                hi = lo + 1
            xs = np.linspace(lo, hi, 200)
            kde0 = gaussian_kde(c0, bw_method=0.3)
            kde1 = gaussian_kde(c1, bw_method=0.3)
            ax.fill_between(xs, kde0(xs), alpha=0.4, color="red", label="Class 0")
            ax.fill_between(xs, kde1(xs), alpha=0.4, color="blue", label="Class 1")
        except Exception:
            ax.text(0.5, 0.5, "KDE failed", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{name[:20]}\n|d|={abs_d:.2f}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=6)

    fig.suptitle("KDE Class Overlay: Top 30 Features by |Cohen's d|", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_kde_class_overlay_grid.png"), dpi=150)
    plt.close()


def plot_12_andrews_curves(feat_s, names_s, labels, d_values):
    """Plot 12: Andrews curves for top 10 features, 200 random samples/class."""
    top10 = d_values[:10]
    top10_idx = [names_s.index(n) for n, _, _ in top10]

    np.random.seed(42)
    c0_idx = np.where(labels == 0)[0]
    c1_idx = np.where(labels == 1)[0]
    s0 = np.random.choice(c0_idx, min(200, len(c0_idx)), replace=False)
    s1 = np.random.choice(c1_idx, min(200, len(c1_idx)), replace=False)

    data0 = feat_s[s0][:, top10_idx]
    data1 = feat_s[s1][:, top10_idx]
    data0 = np.nan_to_num(data0, nan=0.0)
    data1 = np.nan_to_num(data1, nan=0.0)

    # Standardize each column
    combined = np.vstack([data0, data1])
    mu = np.mean(combined, axis=0)
    sd = np.std(combined, axis=0)
    sd[sd < 1e-12] = 1.0
    data0 = (data0 - mu) / sd
    data1 = (data1 - mu) / sd

    t = np.linspace(-np.pi, np.pi, 200)

    def andrews(row, t):
        result = row[0] / np.sqrt(2)
        for i in range(1, len(row)):
            if i % 2 == 1:
                result += row[i] * np.sin((i // 2 + 1) * t)
            else:
                result += row[i] * np.cos((i // 2) * t)
        return result

    fig, ax = plt.subplots(figsize=(12, 6))
    for row in data0:
        ax.plot(t, andrews(row, t), color="red", alpha=0.05, linewidth=0.5)
    for row in data1:
        ax.plot(t, andrews(row, t), color="blue", alpha=0.05, linewidth=0.5)

    # Dummy handles for legend
    ax.plot([], [], color="red", linewidth=2, label="Class 0 (N=200)")
    ax.plot([], [], color="blue", linewidth=2, label="Class 1 (N=200)")
    ax.legend(fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.set_title("Andrews Curves (Top 10 Features by |Cohen's d|)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_andrews_curves_by_class.png"), dpi=150)
    plt.close()


def plot_13_correlation_network(feat_s, names_s):
    """Plot 13: Correlation network graph (top 50 features by variance)."""
    variances = np.var(feat_s, axis=0)
    top50_idx = np.argsort(variances)[-50:]
    top50_names = [names_s[i][:15] for i in top50_idx]
    top50_feats = np.nan_to_num(feat_s[:, top50_idx], nan=0.0)

    corr, _ = spearmanr(top50_feats)
    if corr.ndim == 0:
        return
    N = len(top50_names)
    adj = np.abs(corr) > 0.7
    np.fill_diagonal(adj, False)

    # Spring layout using force-directed placement
    np.random.seed(42)
    pos = np.random.randn(N, 2) * 2.0
    for _ in range(300):
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pos[i] - pos[j]
                dist = max(np.linalg.norm(diff), 0.01)
                # Repulsion (all pairs)
                forces[i] += diff / (dist ** 2) * 0.5
                # Attraction (connected pairs)
                if adj[i, j]:
                    forces[i] -= diff * dist * 0.01
        pos += np.clip(forces, -0.5, 0.5) * 0.1

    # Center and scale
    pos -= pos.mean(axis=0)
    scale = np.max(np.abs(pos))
    if scale > 0:
        pos /= scale

    # Count edges per node for sizing
    edge_count = adj.sum(axis=1)
    sizes = 50 + edge_count * 30

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                rho_val = abs(corr[i, j])
                alpha = 0.2 + 0.6 * (rho_val - 0.7) / 0.3
                alpha = min(alpha, 0.8)
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="#94a3b8", alpha=alpha, linewidth=0.5 + rho_val)

    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=edge_count,
               cmap="YlOrRd", edgecolors="black", linewidths=0.5, zorder=5)

    # Labels for nodes with edges
    for i in range(N):
        if edge_count[i] > 0:
            ax.annotate(top50_names[i], (pos[i, 0], pos[i, 1]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    n_edges = int(adj.sum() / 2)
    ax.set_title(f"Feature Correlation Network (|rho| > 0.7, {n_edges} edges among top 50 by variance)",
                 fontsize=11)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_correlation_network.png"), dpi=150)
    plt.close()


def write_analysis(feat_s, feat_v, names_s, names_v, labels, d_values_s, d_values_v):
    """Write analysis.md."""
    n_train = len(labels)
    n_c0 = int(np.sum(labels == 0))
    n_c1 = int(np.sum(labels == 1))

    # NaN/Inf stats
    nan_s = int(np.sum(~np.isfinite(feat_s)))
    nan_v = int(np.sum(~np.isfinite(feat_v)))

    # Feature stats
    mean_s = np.nanmean(feat_s, axis=0)
    std_s = np.nanstd(feat_s, axis=0)

    text = f"""# Step 4: Feature Extraction Analysis

## Overview
- **Training samples:** {n_train} (Class 0: {n_c0}, Class 1: {n_c1})
- **VQI-S features:** {feat_s.shape[1]} (437 frame-level + 107 global)
- **VQI-V features:** {feat_v.shape[1]}
- **Total features:** {feat_s.shape[1] + feat_v.shape[1]}

## Data Quality
- **VQI-S NaN/Inf:** {nan_s} ({nan_s / (n_train * feat_s.shape[1]) * 100:.4f}%)
- **VQI-V NaN/Inf:** {nan_v} ({nan_v / (n_train * feat_v.shape[1]) * 100:.4f}%)

## Top VQI-S Features by Class Separation (|Cohen's d|)

| Rank | Feature | |Cohen's d| | Direction |
|------|---------|-----------|-----------|
"""
    for i, (name, abs_d, d) in enumerate(d_values_s[:20]):
        direction = "C1 > C0" if d > 0 else "C0 > C1"
        text += f"| {i+1} | {name} | {abs_d:.3f} | {direction} |\n"

    text += f"""
## Top VQI-V Features by Class Separation (|Cohen's d|)

| Rank | Feature | |Cohen's d| | Direction |
|------|---------|-----------|-----------|
"""
    for i, (name, abs_d, d) in enumerate(d_values_v[:20]):
        direction = "C1 > C0" if d > 0 else "C0 > C1"
        text += f"| {i+1} | {name} | {abs_d:.3f} | {direction} |\n"

    text += f"""
## Feature Statistics Summary

### VQI-S
- Mean of means: {np.mean(mean_s):.4f}
- Features with zero variance: {int(np.sum(std_s < 1e-12))}
- Features with |Cohen's d| > 0.5: {sum(1 for _, d, _ in d_values_s if d > 0.5)}
- Features with |Cohen's d| > 0.2: {sum(1 for _, d, _ in d_values_s if d > 0.2)}

### VQI-V
- Features with |Cohen's d| > 0.5: {sum(1 for _, d, _ in d_values_v if d > 0.5)}
- Features with |Cohen's d| > 0.2: {sum(1 for _, d, _ in d_values_v if d > 0.2)}

## Computation
- Feature extraction time per file: see extraction_log_train.csv
- Modules: 23 frame-level + 32 global + 5 VQI-V = 60 feature modules
- Shared intermediates computed once per file
"""

    with open(os.path.join(REPORTS_DIR, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Loading data...")
    feat_s, feat_v, names_s, names_v, labels = load_data()
    print(f"Loaded: S={feat_s.shape}, V={feat_v.shape}, labels={labels.shape}")

    print("Plot 1: Frame distributions...")
    plot_1_frame_distributions(feat_s, names_s, labels)

    print("Plot 2: Global distributions...")
    plot_2_global_distributions(feat_s, names_s, labels)

    print("Plot 3: Correlation matrix...")
    plot_3_correlation_matrix(feat_s, names_s)

    print("Plot 4: NaN/Inf report...")
    plot_4_nan_inf_report(feat_s, feat_v, names_s, names_v)

    print("Plot 6: Class separation...")
    d_values_s = plot_6_class_separation(feat_s, names_s, labels)

    print("Plot 7: Value ranges...")
    plot_7_value_ranges(feat_s, names_s)

    print("Plot 8: Ridgeline top 30...")
    plot_8_ridgeline(feat_s, names_s, labels, d_values_s)

    print("Plot 10: Violin top 20...")
    plot_10_violin(feat_s, names_s, labels, d_values_s)

    print("Plot 5: Histogram examples (top 5)...")
    plot_5_histogram_examples(feat_s, names_s, labels, d_values_s)

    print("Plot 9: Radar by category...")
    plot_9_radar_by_category(d_values_s)

    print("Plot 11: KDE class overlay grid...")
    plot_11_kde_grid(feat_s, names_s, labels, d_values_s)

    print("Plot 12: Andrews curves...")
    plot_12_andrews_curves(feat_s, names_s, labels, d_values_s)

    print("Plot 13: Correlation network...")
    plot_13_correlation_network(feat_s, names_s)

    print("Plots 15-18: VQI-V distributions...")
    plot_v_distributions(feat_v, names_v, labels)

    print("Plot 19: VQI-V class separation...")
    d_values_v = plot_v_class_separation(feat_v, names_v, labels)

    print("Plot 20: VQI-V ridgeline...")
    plot_v_ridgeline(feat_v, names_v, labels, d_values_v)

    print("Plot 21: VQI-V NaN/Inf...")
    plot_v_nan_inf(feat_v, names_v)

    print("Plot 22: VQI-V correlation...")
    plot_v_correlation(feat_v, names_v)

    print("Writing analysis.md...")
    write_analysis(feat_s, feat_v, names_s, names_v, labels, d_values_s, d_values_v)

    print(f"All visualizations saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
