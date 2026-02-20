"""
Step 3 Visualizations — Preprocessing Pipeline
Generates 11 plots + statistical tests for the preprocessing step.

Plots:
  1. vad_mask_examples.png        — 6 waveforms with VAD overlay
  2. vad_ratio_distribution.png   — VAD ratio histogram (reuse Step 2.1 data)
  3. speech_duration_after_vad.png— Speech duration distribution (reuse Step 2.1 data)
  4. actionable_feedback_counts.png — Bar chart of quality check triggers
  5. normalization_effect.png      — Before/after DC removal + peak norm
  6. vad_ratio_ridgeline_by_dataset.png — VAD ratio per dataset
  7. duration_before_after_vad_scatter.png — Scatter: total vs speech duration
  8. spectrogram_vad_examples.png  — Mel spectrograms with VAD overlay
  9. actionable_feedback_by_dataset.png — Feedback flags grouped by dataset
  10. bootstrap_ci_vad_ratio.png   — Bootstrap 95% CI for mean VAD ratio per dataset
  11. ks_test_vad_ratio.png        — KS test p-values between dataset pairs

Statistical tests (M.3 spec):
  - Chi-squared test: feedback flag rates across datasets
  - KS test: VAD ratio distributions across dataset pairs
  - Bootstrap 95% CI: mean VAD ratio per dataset

Usage:
    python scripts/visualize_step3.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from scipy import stats as scipy_stats
import librosa
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqi.preprocessing.audio_loader import load_audio
from vqi.preprocessing.normalize import dc_remove_and_normalize
from vqi.preprocessing.vad import energy_vad, reconstruct_from_mask
from vqi.core.vqi_algorithm import check_actionable_feedback

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "step3")
DURATIONS_CSV = os.path.join(DATA_DIR, "labels", "train_pool_durations.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

# Use a fixed set of 6 diverse samples for waveform/spectrogram plots
# (different datasets, different speech ratios)
SAMPLE_WAVS = None  # populated in main()


def _pick_samples(df, n=6):
    """Pick n diverse samples from different datasets."""
    samples = []
    datasets = df["dataset_source"].unique()
    np.random.seed(42)
    for ds in sorted(datasets):
        sub = df[df["dataset_source"] == ds]
        if len(sub) == 0:
            continue
        # Pick one sample per dataset, cycling through if n > n_datasets
        idx = np.random.choice(len(sub), size=1)[0]
        row = sub.iloc[idx]
        if os.path.exists(row["filename"]):
            samples.append(row)
        if len(samples) >= n:
            break
    # If we don't have enough, sample more from remaining
    if len(samples) < n:
        remaining = df[~df.index.isin([s.name for s in samples])]
        remaining = remaining[remaining["filename"].apply(os.path.exists)]
        extra = remaining.sample(n - len(samples), random_state=42)
        for _, row in extra.iterrows():
            samples.append(row)
    return samples[:n]


def plot_1_vad_mask_examples(samples):
    """Plot 1: Waveform + VAD mask overlay for 6 samples."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, row in enumerate(samples):
        ax = axes[i]
        wav = load_audio(row["filename"])
        wav_norm = dc_remove_and_normalize(wav)
        mask, dur, ratio = energy_vad(wav_norm)

        # Time axes
        t_wav = np.arange(len(wav_norm)) / 16000
        t_mask = np.arange(len(mask)) * 160 / 16000

        ax.plot(t_wav, wav_norm, color="steelblue", alpha=0.6, linewidth=0.3)
        # Overlay VAD mask as shaded regions
        ax.fill_between(t_mask, -1, 1, where=mask, color="green", alpha=0.15,
                         label="Speech" if i == 0 else None)
        ax.set_ylim(-1, 1)
        ds = row["dataset_source"]
        ax.set_title(f"{ds} | ratio={ratio:.2f} | dur={dur:.1f}s", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("VAD Mask Examples (6 samples)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(["Waveform", "Speech region"], loc="upper right", fontsize=9)
    out = os.path.join(REPORT_DIR, "vad_mask_examples.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [1/9] Saved {out}")


def plot_2_vad_ratio_distribution(df):
    """Plot 2: VAD ratio histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["speech_ratio"], bins=100, color="steelblue", edgecolor="white",
            linewidth=0.3, alpha=0.8)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1.5, label="Min threshold (0.05)")
    ax.set_xlabel("VAD Speech Ratio", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("VAD Speech Ratio Distribution (Train Pool, N={:,})".format(len(df)),
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Stats text
    stats = (f"Mean: {df['speech_ratio'].mean():.3f}\n"
             f"Median: {df['speech_ratio'].median():.3f}\n"
             f"Std: {df['speech_ratio'].std():.3f}\n"
             f"< 0.05: {(df['speech_ratio'] < 0.05).sum():,}")
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="lightyellow", edgecolor="gray"))
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "vad_ratio_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [2/9] Saved {out}")


def plot_3_speech_duration_after_vad(df):
    """Plot 3: Speech duration (after VAD) distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["speech_duration_sec"], bins=100, color="coral", edgecolor="white",
            linewidth=0.3, alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Min threshold (1.0s)")
    ax.set_xlabel("Speech Duration (seconds)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Speech Duration After VAD (Train Pool, N={:,})".format(len(df)),
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    stats = (f"Mean: {df['speech_duration_sec'].mean():.2f}s\n"
             f"Median: {df['speech_duration_sec'].median():.2f}s\n"
             f"Min: {df['speech_duration_sec'].min():.2f}s\n"
             f"Max: {df['speech_duration_sec'].max():.2f}s\n"
             f"< 1.0s: {(df['speech_duration_sec'] < 1.0).sum():,}")
    ax.text(0.75, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="lightyellow", edgecolor="gray"))
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "speech_duration_after_vad.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [3/9] Saved {out}")


def plot_4_actionable_feedback_counts(df):
    """Plot 4: Estimated actionable feedback counts from duration data.

    TooShort and InsufficientSpeech can be estimated from duration data.
    TooQuiet and Clipped need raw waveform -- estimated from a random sample.
    """
    # From duration data
    n_too_short = int((df["speech_duration_sec"] < 1.0).sum())
    n_insufficient = int((df["speech_ratio"] < 0.05).sum())

    # Sample 1000 files for amplitude-based checks
    np.random.seed(42)
    sample_df = df[df["filename"].apply(os.path.exists)].sample(
        min(1000, len(df)), random_state=42
    )
    n_too_quiet = 0
    n_clipped = 0
    n_sampled = 0
    for _, row in sample_df.iterrows():
        try:
            wav = load_audio(row["filename"])
            peak = np.abs(wav).max()
            if peak < 0.001:
                n_too_quiet += 1
            clip_ratio = np.sum(np.abs(wav) >= 0.99) / len(wav)
            if clip_ratio > 0.10:
                n_clipped += 1
            n_sampled += 1
        except Exception:
            continue

    # Scale estimates to full pool
    scale = len(df) / n_sampled if n_sampled > 0 else 1.0
    n_too_quiet_est = int(n_too_quiet * scale)
    n_clipped_est = int(n_clipped * scale)

    labels = ["TooShort\n(speech<1.0s)", "TooQuiet\n(peak<0.001)*",
              "Clipped\n(ratio>10%)*", "InsufficientSpeech\n(ratio<5%)"]
    counts = [n_too_short, n_too_quiet_est, n_clipped_est, n_insufficient]
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count (estimated)", fontsize=11)
    ax.set_title("Actionable Quality Feedback Triggers (Train Pool, N={:,})".format(len(df)),
                 fontsize=13, fontweight="bold")
    ax.text(0.98, 0.95, f"*Estimated from {n_sampled:,} sample",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            style="italic", color="gray")
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "actionable_feedback_counts.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [4/9] Saved {out}")
    return {
        "n_too_short": n_too_short,
        "n_too_quiet_est": n_too_quiet_est,
        "n_clipped_est": n_clipped_est,
        "n_insufficient": n_insufficient,
        "n_sampled": n_sampled,
    }


def plot_5_normalization_effect(samples):
    """Plot 5: Before/after normalization for 2 samples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for i in range(min(2, len(samples))):
        row = samples[i]
        wav = load_audio(row["filename"])
        wav_norm = dc_remove_and_normalize(wav)
        t = np.arange(len(wav)) / 16000

        # Before
        axes[i, 0].plot(t, wav, color="steelblue", linewidth=0.3)
        axes[i, 0].set_title(f"Raw: DC={wav.mean():.4f}, peak={np.abs(wav).max():.4f}",
                             fontsize=9)
        axes[i, 0].set_ylim(-1.1, 1.1)
        axes[i, 0].set_ylabel("Amplitude", fontsize=9)
        axes[i, 0].set_xlabel("Time (s)", fontsize=8)

        # After
        axes[i, 1].plot(t, wav_norm, color="coral", linewidth=0.3)
        axes[i, 1].set_title(f"Normalized: DC={wav_norm.mean():.6f}, peak={np.abs(wav_norm).max():.4f}",
                             fontsize=9)
        axes[i, 1].set_ylim(-1.1, 1.1)
        axes[i, 1].set_ylabel("Amplitude", fontsize=9)
        axes[i, 1].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle("Normalization Effect (DC Removal + Peak Norm to 0.95)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(REPORT_DIR, "normalization_effect.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [5/9] Saved {out}")


def plot_6_vad_ratio_ridgeline(df):
    """Plot 6: VAD ratio distribution per dataset (ridgeline-style)."""
    datasets = sorted(df["dataset_source"].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(n_ds, 1, figsize=(10, 2 * n_ds), sharex=True)
    if n_ds == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_ds))
    for i, ds in enumerate(datasets):
        sub = df[df["dataset_source"] == ds]["speech_ratio"]
        axes[i].hist(sub, bins=80, color=colors[i], edgecolor="white",
                     linewidth=0.2, alpha=0.8, density=True)
        axes[i].set_ylabel(ds.replace("_", "\n"), fontsize=8, rotation=0,
                          ha="right", va="center")
        axes[i].set_yticks([])
        axes[i].text(0.02, 0.85, f"N={len(sub):,} | mean={sub.mean():.3f}",
                     transform=axes[i].transAxes, fontsize=8)
    axes[-1].set_xlabel("VAD Speech Ratio", fontsize=11)
    fig.suptitle("VAD Speech Ratio by Dataset", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(REPORT_DIR, "vad_ratio_ridgeline_by_dataset.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [6/9] Saved {out}")


def plot_7_duration_scatter(df):
    """Plot 7: Total duration vs speech duration scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Subsample for performance
    n_plot = min(50000, len(df))
    sub = df.sample(n_plot, random_state=42) if len(df) > n_plot else df

    ax.scatter(sub["total_duration_sec"], sub["speech_duration_sec"],
               s=0.5, alpha=0.15, color="steelblue", rasterized=True)
    # Diagonal (perfect: all audio is speech)
    lim = max(sub["total_duration_sec"].max(), sub["speech_duration_sec"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, alpha=0.5, label="x=y (100% speech)")
    ax.set_xlabel("Total Audio Duration (seconds)", fontsize=11)
    ax.set_ylabel("Speech Duration After VAD (seconds)", fontsize=11)
    ax.set_title("Total vs. Speech Duration (N={:,} shown)".format(n_plot),
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "duration_before_after_vad_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [7/9] Saved {out}")


def plot_8_spectrogram_vad(samples):
    """Plot 8: Mel-spectrogram with VAD overlay for 6 samples."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for i, row in enumerate(samples):
        ax = axes_flat[i]
        wav = load_audio(row["filename"])
        wav_norm = dc_remove_and_normalize(wav)
        mask, dur, ratio = energy_vad(wav_norm)

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=wav_norm, sr=16000, n_mels=80,
                                           hop_length=160, n_fft=512)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=16000, hop_length=160, x_axis="time",
                                 y_axis="mel", ax=ax, cmap="magma")
        # Overlay VAD: draw red lines at top/bottom for non-speech frames
        t_mask = np.arange(len(mask)) * 160 / 16000
        non_speech = ~mask
        if np.any(non_speech):
            for start_idx in np.where(np.diff(non_speech.astype(int), prepend=0) == 1)[0]:
                end_idx_arr = np.where(np.diff(non_speech.astype(int), prepend=0) == -1)[0]
                end_candidates = end_idx_arr[end_idx_arr > start_idx]
                if len(end_candidates) > 0:
                    end_idx = end_candidates[0]
                else:
                    end_idx = len(mask)
                t_s = start_idx * 160 / 16000
                t_e = end_idx * 160 / 16000
                ax.axvspan(t_s, t_e, color="white", alpha=0.25)

        ds = row["dataset_source"]
        ax.set_title(f"{ds} | ratio={ratio:.2f}", fontsize=9)

    fig.suptitle("Mel Spectrogram + VAD (white = non-speech)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(REPORT_DIR, "spectrogram_vad_examples.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [8/9] Saved {out}")


def plot_9_feedback_by_dataset(df):
    """Plot 9: Actionable feedback flags per dataset."""
    datasets = sorted(df["dataset_source"].unique())
    results = {ds: {"TooShort": 0, "InsufficientSpeech": 0} for ds in datasets}

    for ds in datasets:
        sub = df[df["dataset_source"] == ds]
        results[ds]["TooShort"] = int((sub["speech_duration_sec"] < 1.0).sum())
        results[ds]["InsufficientSpeech"] = int((sub["speech_ratio"] < 0.05).sum())

    ds_labels = [ds.replace("_", "\n") for ds in datasets]
    too_short = [results[ds]["TooShort"] for ds in datasets]
    insuff = [results[ds]["InsufficientSpeech"] for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, too_short, width, label="TooShort (<1.0s)", color="#e74c3c")
    bars2 = ax.bar(x + width / 2, insuff, width, label="InsufficientSpeech (<5%)", color="#3498db")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=8)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Actionable Feedback by Dataset (duration-based only)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "actionable_feedback_by_dataset.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [9/9] Saved {out}")


def compute_statistical_tests(df):
    """Compute all M.3 statistical tests, return results dict."""
    datasets = sorted(df["dataset_source"].unique())
    results = {}

    # ---- 1. Chi-squared test: feedback flag rates across datasets ----
    # Build contingency table: rows = datasets, cols = [flagged, not_flagged]
    flagged_col = ((df["speech_duration_sec"] < 1.0) | (df["speech_ratio"] < 0.05)).astype(int)
    contingency = pd.crosstab(df["dataset_source"], flagged_col)
    # Ensure both columns exist
    if 0 not in contingency.columns:
        contingency[0] = 0
    if 1 not in contingency.columns:
        contingency[1] = 0
    contingency = contingency[[0, 1]]
    chi2, chi2_p, chi2_dof, chi2_expected = scipy_stats.chi2_contingency(contingency)
    results["chi2"] = {
        "statistic": float(chi2),
        "p_value": float(chi2_p),
        "dof": int(chi2_dof),
        "contingency": contingency.to_dict(),
    }
    print(f"\n  Chi-squared test (feedback rates across datasets):")
    print(f"    chi2 = {chi2:.4f}, p = {chi2_p:.4e}, dof = {chi2_dof}")

    # ---- 2. KS test: VAD ratio distributions across dataset pairs ----
    ks_results = {}
    for ds_a, ds_b in combinations(datasets, 2):
        ratios_a = df[df["dataset_source"] == ds_a]["speech_ratio"].values
        ratios_b = df[df["dataset_source"] == ds_b]["speech_ratio"].values
        ks_stat, ks_p = scipy_stats.ks_2samp(ratios_a, ratios_b)
        pair_key = f"{ds_a} vs {ds_b}"
        ks_results[pair_key] = {"statistic": float(ks_stat), "p_value": float(ks_p)}
        print(f"    KS test [{pair_key}]: D = {ks_stat:.4f}, p = {ks_p:.4e}")
    results["ks_tests"] = ks_results

    # ---- 3. Bootstrap 95% CI for mean VAD ratio per dataset ----
    n_bootstrap = 10000
    np.random.seed(42)
    bootstrap_results = {}
    for ds in datasets:
        ratios = df[df["dataset_source"] == ds]["speech_ratio"].values
        boot_means = np.array([
            np.mean(np.random.choice(ratios, size=len(ratios), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        point = float(np.mean(ratios))
        bootstrap_results[ds] = {"mean": point, "ci_lo": ci_lo, "ci_hi": ci_hi}
        print(f"    Bootstrap CI [{ds}]: mean = {point:.5f}, 95% CI = [{ci_lo:.5f}, {ci_hi:.5f}]")
    results["bootstrap_ci"] = bootstrap_results

    return results


def plot_10_bootstrap_ci(stat_results):
    """Plot 10: Bootstrap 95% CI for mean VAD ratio per dataset."""
    boot = stat_results["bootstrap_ci"]
    datasets = sorted(boot.keys())
    means = [boot[ds]["mean"] for ds in datasets]
    ci_lo = [boot[ds]["ci_lo"] for ds in datasets]
    ci_hi = [boot[ds]["ci_hi"] for ds in datasets]
    errors_lo = [m - lo for m, lo in zip(means, ci_lo)]
    errors_hi = [hi - m for m, hi in zip(means, ci_hi)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    ax.errorbar(x, means, yerr=[errors_lo, errors_hi], fmt="o", capsize=8,
                capthick=2, markersize=10, color="steelblue", ecolor="coral",
                elinewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("_", "\n") for ds in datasets], fontsize=9)
    ax.set_ylabel("Mean VAD Speech Ratio", fontsize=11)
    ax.set_title("Bootstrap 95% CI: Mean VAD Ratio per Dataset (N=10,000 resamples)",
                 fontsize=12, fontweight="bold")

    # Annotate values
    for i, ds in enumerate(datasets):
        ax.annotate(f"{means[i]:.4f}\n[{ci_lo[i]:.4f}, {ci_hi[i]:.4f}]",
                    (x[i], means[i]), textcoords="offset points", xytext=(0, 18),
                    ha="center", fontsize=8, color="gray")

    ax.set_ylim(min(ci_lo) - 0.005, max(ci_hi) + 0.005)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "bootstrap_ci_vad_ratio.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [10/11] Saved {out}")


def plot_11_ks_test_heatmap(stat_results):
    """Plot 11: KS test statistics between dataset pairs."""
    ks = stat_results["ks_tests"]
    # Extract unique datasets from pair keys
    all_ds = set()
    for key in ks:
        a, b = key.split(" vs ")
        all_ds.add(a)
        all_ds.add(b)
    datasets = sorted(all_ds)
    n = len(datasets)

    # Build matrix
    D_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))
    for key, vals in ks.items():
        a, b = key.split(" vs ")
        i, j = datasets.index(a), datasets.index(b)
        D_matrix[i, j] = D_matrix[j, i] = vals["statistic"]
        p_matrix[i, j] = p_matrix[j, i] = vals["p_value"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D_matrix, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([ds.replace("_", "\n") for ds in datasets], fontsize=9)
    ax.set_yticklabels([ds.replace("_", "\n") for ds in datasets], fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="gray")
            else:
                p_str = f"{p_matrix[i, j]:.1e}" if p_matrix[i, j] < 0.001 else f"{p_matrix[i, j]:.3f}"
                ax.text(j, i, f"D={D_matrix[i, j]:.4f}\np={p_str}",
                        ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="KS Statistic (D)", shrink=0.8)
    ax.set_title("KS Test: VAD Ratio Distribution Similarity Between Datasets",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "ks_test_vad_ratio.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [11/11] Saved {out}")


def main():
    print("=" * 60)
    print("Step 3 Visualizations: Preprocessing Pipeline")
    print("=" * 60)

    # Load Step 2.1 duration data
    print("\nLoading duration data...")
    df = pd.read_csv(DURATIONS_CSV)
    print(f"  Loaded {len(df):,} rows from {DURATIONS_CSV}")

    # Pick 6 diverse samples
    print("  Selecting 6 diverse audio samples...")
    samples = _pick_samples(df, n=6)
    print(f"  Selected {len(samples)} samples from datasets: "
          + ", ".join(s["dataset_source"] for s in samples))

    # Generate plots
    print("\nGenerating plots...")
    plot_1_vad_mask_examples(samples)
    plot_2_vad_ratio_distribution(df)
    plot_3_speech_duration_after_vad(df)
    feedback_stats = plot_4_actionable_feedback_counts(df)
    plot_5_normalization_effect(samples)
    plot_6_vad_ratio_ridgeline(df)
    plot_7_duration_scatter(df)
    plot_8_spectrogram_vad(samples)
    plot_9_feedback_by_dataset(df)

    # Statistical tests (M.3 spec)
    print("\nComputing statistical tests (M.3 spec)...")
    stat_results = compute_statistical_tests(df)
    plot_10_bootstrap_ci(stat_results)
    plot_11_ks_test_heatmap(stat_results)

    # Save statistical results as JSON for analysis.md
    stat_out = os.path.join(REPORT_DIR, "statistical_tests.json")
    with open(stat_out, "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f"\n  Saved statistical test results to {stat_out}")

    print("\n" + "=" * 60)
    print("All 11 plots + statistical tests generated successfully!")
    print(f"Output: {REPORT_DIR}")
    print("=" * 60)

    return feedback_stats, stat_results


if __name__ == "__main__":
    stats, stat_results = main()
