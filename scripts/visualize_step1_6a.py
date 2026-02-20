"""
Step 1.6a Completion Visualizations and Statistical Analysis.

Produces:
1. Embedding norm distributions per provider (sanity check)
2. Per-dimension mean/std distributions per provider
3. Cosine similarity distributions (sample genuine + impostor pairs)
4. PCA 2D projection colored by dataset source (sample)
5. Dataset composition bar chart
6. Per-speaker utterance count distribution
7. Summary statistics table (printed + saved as CSV)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

OUTPUT_DIR = r"D:\VQI\implementation\reports"
EMBED_DIR = r"D:\VQI\implementation\data\embeddings"
INDEX_CSV = os.path.join(EMBED_DIR, "train_pool_index.csv")

PROVIDERS = {
    "P1_ECAPA": ("train_pool_P1_ECAPA.npy", 192),
    "P2_RESNET": ("train_pool_P2_RESNET.npy", 256),
    "P3_ECAPA2": ("train_pool_P3_ECAPA2.npy", 192),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)


def load_index():
    print("Loading index CSV...")
    df = pd.read_csv(INDEX_CSV)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_memmap(name, dim, n):
    path = os.path.join(EMBED_DIR, name)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))
    return mm


def plot_norm_distributions(memmaps, n_total):
    """Plot 1: L2 norm distributions per provider."""
    print("Plot 1: Norm distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Sample 100K rows for speed
    sample_idx = np.random.choice(n_total, min(100000, n_total), replace=False)
    sample_idx.sort()

    for ax, (pname, mm) in zip(axes, memmaps.items()):
        chunk = np.array(mm[sample_idx])
        norms = np.linalg.norm(chunk, axis=1)
        # Norms may be nearly identical (all 1.0 for L2-normed) -- use deviation from 1.0
        deviations = np.abs(norms - 1.0)
        if deviations.max() < 1e-6:
            # All norms are exactly 1.0 -- show as verification bar
            ax.bar(["Norm = 1.0"], [len(norms)], color="green", alpha=0.8)
            ax.set_title(f"{pname}: ALL norms = 1.000000")
            ax.set_ylabel("Count")
        else:
            ax.hist(norms, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
            ax.set_title(f"{pname} L2 norms (n={len(sample_idx):,})")
            ax.set_xlabel("L2 norm")
            ax.set_ylabel("Count")
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="1.0")
            ax.legend()
        stats_text = f"min={norms.min():.6f}\nmax={norms.max():.6f}\nmean={norms.mean():.6f}\nstd={norms.std():.2e}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_norm_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_dimension_stats(memmaps, n_total):
    """Plot 2: Per-dimension mean and std."""
    print("Plot 2: Per-dimension mean/std...")
    sample_idx = np.random.choice(n_total, min(50000, n_total), replace=False)
    sample_idx.sort()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for col, (pname, mm) in enumerate(memmaps.items()):
        chunk = np.array(mm[sample_idx])
        dim_means = chunk.mean(axis=0)
        dim_stds = chunk.std(axis=0)

        axes[0, col].bar(range(len(dim_means)), dim_means, color="steelblue", width=1.0)
        axes[0, col].set_title(f"{pname} - Per-dim mean")
        axes[0, col].set_xlabel("Dimension")
        axes[0, col].set_ylabel("Mean")

        axes[1, col].bar(range(len(dim_stds)), dim_stds, color="coral", width=1.0)
        axes[1, col].set_title(f"{pname} - Per-dim std")
        axes[1, col].set_xlabel("Dimension")
        axes[1, col].set_ylabel("Std")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_dimension_stats.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cosine_similarity(memmaps, index_df, n_total):
    """Plot 3: Cosine similarity distributions for genuine and impostor pairs (sample)."""
    print("Plot 3: Cosine similarity distributions...")
    # Sample 5000 genuine pairs and 5000 impostor pairs
    speakers = index_df["speaker_id"].values
    unique_speakers = index_df["speaker_id"].unique()

    # Build speaker -> indices map (sample for speed)
    spk_to_idx = {}
    for spk in unique_speakers[:500]:  # first 500 speakers
        spk_to_idx[spk] = index_df.index[index_df["speaker_id"] == spk].values

    n_pairs = 5000
    genuine_pairs = []
    impostor_pairs = []

    spk_list = list(spk_to_idx.keys())
    # Genuine pairs: same speaker, different utterances
    for _ in range(n_pairs):
        spk = spk_list[np.random.randint(len(spk_list))]
        idxs = spk_to_idx[spk]
        if len(idxs) < 2:
            continue
        i, j = np.random.choice(idxs, 2, replace=False)
        genuine_pairs.append((i, j))

    # Impostor pairs: different speakers
    for _ in range(n_pairs):
        s1, s2 = np.random.choice(len(spk_list), 2, replace=False)
        i = np.random.choice(spk_to_idx[spk_list[s1]])
        j = np.random.choice(spk_to_idx[spk_list[s2]])
        impostor_pairs.append((i, j))

    genuine_pairs = np.array(genuine_pairs)
    impostor_pairs = np.array(impostor_pairs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (pname, mm) in zip(axes, memmaps.items()):
        # Genuine cosine scores
        g_emb1 = np.array(mm[genuine_pairs[:, 0]])
        g_emb2 = np.array(mm[genuine_pairs[:, 1]])
        gen_scores = np.sum(g_emb1 * g_emb2, axis=1)  # cosine = dot product for L2-normed

        # Impostor cosine scores
        i_emb1 = np.array(mm[impostor_pairs[:, 0]])
        i_emb2 = np.array(mm[impostor_pairs[:, 1]])
        imp_scores = np.sum(i_emb1 * i_emb2, axis=1)

        ax.hist(gen_scores, bins=80, alpha=0.6, color="green", label=f"Genuine (n={len(gen_scores)})", density=True)
        ax.hist(imp_scores, bins=80, alpha=0.6, color="red", label=f"Impostor (n={len(imp_scores)})", density=True)
        ax.set_title(f"{pname} Cosine Similarity")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend()

        # Add d-prime
        mu_g, mu_i = gen_scores.mean(), imp_scores.mean()
        sig_g, sig_i = gen_scores.std(), imp_scores.std()
        dprime = (mu_g - mu_i) / (sig_g + sig_i) if (sig_g + sig_i) > 0 else 0
        ax.text(0.02, 0.95, f"d'={dprime:.2f}\nmu_g={mu_g:.3f}\nmu_i={mu_i:.3f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_cosine_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_pca_projection(memmaps, index_df, n_total):
    """Plot 4: PCA 2D projection colored by dataset source."""
    print("Plot 4: PCA projection by dataset...")
    sample_idx = np.random.choice(n_total, min(10000, n_total), replace=False)
    sample_idx.sort()
    sample_datasets = index_df["dataset_source"].values[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dataset_colors = {
        "voxceleb1_dev": "tab:blue",
        "voxceleb2_dev": "tab:orange",
        "librispeech": "tab:green",
        "vctk": "tab:red",
        "voices": "tab:purple",
        "cnceleb": "tab:brown",
    }

    for ax, (pname, mm) in zip(axes, memmaps.items()):
        chunk = np.array(mm[sample_idx])
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(chunk)

        for ds, color in dataset_colors.items():
            mask = sample_datasets == ds
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1], c=color, s=1, alpha=0.3, label=f"{ds} ({mask.sum()})")

        ax.set_title(f"{pname} PCA (var: {pca.explained_variance_ratio_.sum():.2%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6, markerscale=5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_pca_by_dataset.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_dataset_composition(index_df):
    """Plot 5: Dataset composition bar chart."""
    print("Plot 5: Dataset composition...")
    counts = index_df["dataset_source"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(counts)), counts.values, color="steelblue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Number of utterances")
    ax.set_title("Training Pool: Dataset Composition (1,210,451 total)")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:,}\n({val/len(index_df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_dataset_composition.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_speaker_utterance_distribution(index_df):
    """Plot 6: Per-speaker utterance count distribution."""
    print("Plot 6: Speaker utterance count distribution...")
    spk_counts = index_df["speaker_id"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(spk_counts.values, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("Utterances per speaker")
    axes[0].set_ylabel("Number of speakers")
    axes[0].set_title(f"Utterances/Speaker Distribution ({len(spk_counts):,} speakers)")
    stats = f"min={spk_counts.min()}\nmax={spk_counts.max()}\nmedian={spk_counts.median():.0f}\nmean={spk_counts.mean():.1f}"
    axes[0].text(0.7, 0.95, stats, transform=axes[0].transAxes, fontsize=9,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Log scale
    axes[1].hist(spk_counts.values, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Utterances per speaker")
    axes[1].set_ylabel("Number of speakers (log)")
    axes[1].set_yscale("log")
    axes[1].set_title("Same (log scale)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_6a_speaker_utterance_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def compute_summary_statistics(memmaps, index_df, n_total):
    """Compute and save summary statistics CSV."""
    print("Computing summary statistics...")
    sample_idx = np.random.choice(n_total, min(100000, n_total), replace=False)
    sample_idx.sort()

    rows = []
    for pname, mm in memmaps.items():
        chunk = np.array(mm[sample_idx])
        norms = np.linalg.norm(chunk, axis=1)
        rows.append({
            "provider": pname,
            "total_embeddings": n_total,
            "embedding_dim": mm.shape[1],
            "norm_min": f"{norms.min():.6f}",
            "norm_max": f"{norms.max():.6f}",
            "norm_mean": f"{norms.mean():.6f}",
            "norm_std": f"{norms.std():.2e}",
            "nan_count": int(np.isnan(chunk).sum()),
            "inf_count": int(np.isinf(chunk).sum()),
            "zero_rows": int((norms < 1e-6).sum()),
            "dim_mean_range": f"[{chunk.mean(axis=0).min():.4f}, {chunk.mean(axis=0).max():.4f}]",
            "dim_std_range": f"[{chunk.std(axis=0).min():.4f}, {chunk.std(axis=0).max():.4f}]",
        })

    stats_df = pd.DataFrame(rows)

    # Add dataset stats
    n_speakers = index_df["speaker_id"].nunique()
    n_datasets = index_df["dataset_source"].nunique()
    dataset_counts = index_df["dataset_source"].value_counts().to_dict()

    path = os.path.join(OUTPUT_DIR, "1_6a_summary_statistics.csv")
    stats_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Print summary
    print("\n" + "=" * 60)
    print("STEP 1.6a SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total embeddings: {n_total:,}")
    print(f"Unique speakers:  {n_speakers:,}")
    print(f"Datasets:         {n_datasets}")
    for ds, cnt in dataset_counts.items():
        print(f"  {ds}: {cnt:,} ({cnt/n_total*100:.1f}%)")
    print()
    print(stats_df.to_string(index=False))
    print("=" * 60)

    return stats_df


def main():
    print("=" * 60)
    print("Step 1.6a Completion Visualizations")
    print("=" * 60)

    index_df = load_index()
    n_total = len(index_df)

    # Load memmaps
    memmaps = {}
    for pname, (fname, dim) in PROVIDERS.items():
        memmaps[pname] = load_memmap(fname, dim, n_total)
        print(f"  Loaded {pname}: shape=({n_total}, {dim})")

    # Generate all plots
    plot_norm_distributions(memmaps, n_total)
    plot_dimension_stats(memmaps, n_total)
    plot_cosine_similarity(memmaps, index_df, n_total)
    plot_pca_projection(memmaps, index_df, n_total)
    plot_dataset_composition(index_df)
    plot_speaker_utterance_distribution(index_df)
    stats = compute_summary_statistics(memmaps, index_df, n_total)

    print("\nAll visualizations saved to:", OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
