"""
Step 1.6b Completion Visualizations and Statistical Analysis.

Produces:
1. Genuine vs Impostor score distributions (raw) per provider
2. Genuine vs Impostor score distributions (s-norm) per provider
3. Genuine score distribution across providers (overlay)
4. Score statistics summary table
5. Per-provider d-prime (Fisher discriminant ratio)
6. Genuine score vs speaker utterance count scatter
7. Score correlation across providers
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUTPUT_DIR = r"D:\VQI\implementation\reports"
SCORES_DIR = r"D:\VQI\implementation\data\step1\provider_scores"
INDEX_CSV = r"D:\VQI\implementation\data\step1\embeddings\train_pool_index.csv"

PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]
SHORT = {"P1_ECAPA": "ecapa", "P2_RESNET": "resnet", "P3_ECAPA2": "ecapa2"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)


def load_scores():
    """Load genuine scores for all providers."""
    scores = {}
    for pn in PROVIDERS:
        csv_path = os.path.join(SCORES_DIR, f"scores_{pn}_{SHORT[pn]}.csv")
        print(f"  Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        scores[pn] = df
    return scores


def load_impostors():
    """Load impostor score arrays."""
    impostors = {}
    for pn in PROVIDERS:
        raw = np.load(os.path.join(SCORES_DIR, f"impostor_raw_{pn}.npy"))
        norm = np.load(os.path.join(SCORES_DIR, f"impostor_norm_{pn}.npy"))
        impostors[pn] = {"raw": raw, "norm": norm}
        print(f"  {pn} impostors: {len(raw):,} pairs")
    return impostors


def plot_genuine_vs_impostor_raw(scores, impostors):
    """Plot 1: Genuine vs Impostor distributions (raw cosine)."""
    print("Plot 1: Genuine vs Impostor (raw)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pn in zip(axes, PROVIDERS):
        gen = scores[pn]["genuine_raw"].dropna().values
        imp_sample = impostors[pn]["raw"]
        # Sample impostors for plotting (28M is too many)
        if len(imp_sample) > 500000:
            imp_sample = np.random.choice(imp_sample, 500000, replace=False)

        ax.hist(gen, bins=150, alpha=0.6, color="green", density=True, label=f"Genuine (n={len(gen):,})")
        ax.hist(imp_sample, bins=150, alpha=0.6, color="red", density=True, label=f"Impostor (n={len(impostors[pn]['raw']):,})")
        ax.set_title(f"{pn} Raw Cosine Scores")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # d-prime
        mu_g, mu_i = gen.mean(), impostors[pn]["raw"].mean()
        sig_g, sig_i = gen.std(), impostors[pn]["raw"].std()
        dprime = (mu_g - mu_i) / (sig_g + sig_i)
        ax.text(0.02, 0.95, f"d'={dprime:.2f}", transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_genuine_vs_impostor_raw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_genuine_vs_impostor_norm(scores, impostors):
    """Plot 2: Genuine vs Impostor distributions (s-norm)."""
    print("Plot 2: Genuine vs Impostor (s-norm)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pn in zip(axes, PROVIDERS):
        gen = scores[pn]["genuine_norm"].dropna().values
        imp_sample = impostors[pn]["norm"]
        if len(imp_sample) > 500000:
            imp_sample = np.random.choice(imp_sample, 500000, replace=False)

        ax.hist(gen, bins=150, alpha=0.6, color="green", density=True, label=f"Genuine (n={len(gen):,})")
        ax.hist(imp_sample, bins=150, alpha=0.6, color="red", density=True, label=f"Impostor (n={len(impostors[pn]['norm']):,})")
        ax.set_title(f"{pn} S-Norm Scores")
        ax.set_xlabel("S-norm score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        mu_g, mu_i = gen.mean(), impostors[pn]["norm"].mean()
        sig_g, sig_i = gen.std(), impostors[pn]["norm"].std()
        dprime = (mu_g - mu_i) / (sig_g + sig_i)
        ax.text(0.02, 0.95, f"d'={dprime:.2f}", transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_genuine_vs_impostor_snorm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_genuine_overlay(scores):
    """Plot 3: Genuine score distributions overlaid across providers."""
    print("Plot 3: Genuine overlay across providers...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"P1_ECAPA": "tab:blue", "P2_RESNET": "tab:orange", "P3_ECAPA2": "tab:green"}

    for pn in PROVIDERS:
        gen_raw = scores[pn]["genuine_raw"].dropna().values
        gen_norm = scores[pn]["genuine_norm"].dropna().values
        axes[0].hist(gen_raw, bins=150, alpha=0.5, color=colors[pn], density=True, label=pn)
        axes[1].hist(gen_norm, bins=150, alpha=0.5, color=colors[pn], density=True, label=pn)

    axes[0].set_title("Genuine Raw Scores (all providers)")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].legend()
    axes[1].set_title("Genuine S-Norm Scores (all providers)")
    axes[1].set_xlabel("S-norm score")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_genuine_overlay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_dprime_bar(scores, impostors):
    """Plot 4: d-prime bar chart per provider (raw and s-norm)."""
    print("Plot 4: d-prime bar chart...")
    dprimes_raw = []
    dprimes_norm = []

    for pn in PROVIDERS:
        gen_raw = scores[pn]["genuine_raw"].dropna().values
        gen_norm = scores[pn]["genuine_norm"].dropna().values
        imp_raw = impostors[pn]["raw"]
        imp_norm = impostors[pn]["norm"]

        dp_raw = (gen_raw.mean() - imp_raw.mean()) / (gen_raw.std() + imp_raw.std())
        dp_norm = (gen_norm.mean() - imp_norm.mean()) / (gen_norm.std() + imp_norm.std())
        dprimes_raw.append(dp_raw)
        dprimes_norm.append(dp_norm)

    x = np.arange(len(PROVIDERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, dprimes_raw, width, label="Raw cosine", color="steelblue")
    bars2 = ax.bar(x + width/2, dprimes_norm, width, label="S-norm", color="coral")

    ax.set_ylabel("d-prime (Fisher Discriminant Ratio)")
    ax.set_title("Provider d-prime: Genuine vs Impostor Separation")
    ax.set_xticks(x)
    ax.set_xticklabels(PROVIDERS)
    ax.legend()

    for bar, val in zip(bars1, dprimes_raw):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, dprimes_norm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_dprime_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_score_vs_utterance_count(scores):
    """Plot 5: Genuine score vs speaker utterance count."""
    print("Plot 5: Score vs utterance count...")
    index_df = pd.read_csv(INDEX_CSV)
    spk_counts = index_df["speaker_id"].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, pn in zip(axes, PROVIDERS):
        df = scores[pn].copy()
        df["spk_count"] = df["speaker_id"].map(spk_counts)
        # Sample for plotting
        sample = df.dropna(subset=["genuine_raw"]).sample(min(50000, len(df)), random_state=42)

        ax.scatter(sample["spk_count"], sample["genuine_raw"], s=1, alpha=0.1, c="steelblue")
        ax.set_xlabel("Utterances per speaker")
        ax.set_ylabel("Genuine raw score")
        ax.set_title(f"{pn}")

        # Binned mean
        bins = pd.cut(sample["spk_count"], bins=20)
        binned = sample.groupby(bins, observed=True)["genuine_raw"].mean()
        bin_centers = [interval.mid for interval in binned.index]
        ax.plot(bin_centers, binned.values, "r-", linewidth=2, label="Binned mean")
        ax.legend()

    plt.suptitle("Genuine Score vs Speaker Utterance Count", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_score_vs_utt_count.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_provider_correlation(scores):
    """Plot 6: Pairwise Spearman correlation of genuine scores across providers."""
    print("Plot 6: Cross-provider correlation...")
    # Merge genuine_raw from all providers
    merged = pd.DataFrame({"idx": range(len(scores[PROVIDERS[0]]))})
    for pn in PROVIDERS:
        merged[pn] = scores[pn]["genuine_raw"].values

    merged_clean = merged.dropna()
    n_clean = len(merged_clean)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]

    for ax, (i, j) in zip(axes, pairs):
        pn_i, pn_j = PROVIDERS[i], PROVIDERS[j]
        x = merged_clean[pn_i].values
        y = merged_clean[pn_j].values
        # Sample for plotting
        if len(x) > 50000:
            idx = np.random.choice(len(x), 50000, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y

        ax.scatter(x_plot, y_plot, s=1, alpha=0.1, c="steelblue")
        rho, _ = spearmanr(x, y)
        ax.set_xlabel(f"{pn_i} genuine_raw")
        ax.set_ylabel(f"{pn_j} genuine_raw")
        ax.set_title(f"Spearman rho = {rho:.4f} (n={n_clean:,})")
        # Diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.5)

    plt.suptitle("Cross-Provider Genuine Score Correlation", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6b_cross_provider_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_summary_csv(scores, impostors):
    """Save summary statistics CSV."""
    print("Saving summary CSV...")
    rows = []
    for pn in PROVIDERS:
        gen_raw = scores[pn]["genuine_raw"].dropna().values
        gen_norm = scores[pn]["genuine_norm"].dropna().values
        imp_raw = impostors[pn]["raw"]
        imp_norm = impostors[pn]["norm"]

        dp_raw = (gen_raw.mean() - imp_raw.mean()) / (gen_raw.std() + imp_raw.std())
        dp_norm = (gen_norm.mean() - imp_norm.mean()) / (gen_norm.std() + imp_norm.std())

        rows.append({
            "provider": pn,
            "genuine_raw_count": len(gen_raw),
            "genuine_raw_mean": f"{gen_raw.mean():.4f}",
            "genuine_raw_std": f"{gen_raw.std():.4f}",
            "genuine_raw_min": f"{gen_raw.min():.4f}",
            "genuine_raw_max": f"{gen_raw.max():.4f}",
            "genuine_norm_mean": f"{gen_norm.mean():.4f}",
            "genuine_norm_std": f"{gen_norm.std():.4f}",
            "impostor_raw_count": len(imp_raw),
            "impostor_raw_mean": f"{imp_raw.mean():.4f}",
            "impostor_raw_std": f"{imp_raw.std():.4f}",
            "impostor_norm_mean": f"{imp_norm.mean():.4f}",
            "impostor_norm_std": f"{imp_norm.std():.4f}",
            "dprime_raw": f"{dp_raw:.4f}",
            "dprime_snorm": f"{dp_norm:.4f}",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "1_6b_summary_statistics.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print("STEP 1.6b SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)


def main():
    print("=" * 60)
    print("Step 1.6b Completion Visualizations")
    print("=" * 60)

    scores = load_scores()
    impostors = load_impostors()

    plot_genuine_vs_impostor_raw(scores, impostors)
    plot_genuine_vs_impostor_norm(scores, impostors)
    plot_genuine_overlay(scores)
    plot_dprime_bar(scores, impostors)
    plot_score_vs_utterance_count(scores)
    plot_cross_provider_correlation(scores)
    save_summary_csv(scores, impostors)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
