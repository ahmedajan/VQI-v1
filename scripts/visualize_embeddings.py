"""
Visualize embedding data collected so far.
1) Provider cosine similarity distributions (S-norm cohort, 1000 speakers x 5 providers)
2) Embedding space t-SNE (sampled from partial train_pool memmaps, CPU only)

Outputs saved to implementation/reports/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import csv
import random
import os

COHORT_DIR = Path(r"D:\VQI\implementation\data\snorm_cohort")
EMBED_DIR = Path(r"D:\VQI\implementation\data\embeddings")
REPORT_DIR = Path(r"D:\VQI\implementation\reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = [
    ("P1_ECAPA", 192),
    ("P2_RESNET", 256),
    ("P3_ECAPA2", 192),
    ("P4_XVECTOR", 512),
    ("P5_WAVLM", 512),
]

TRAIN_PROVIDERS = [
    ("P1_ECAPA", 192),
    ("P2_RESNET", 256),
    ("P3_ECAPA2", 192),
]


def plot_provider_distributions():
    """Plot pairwise cosine similarity distributions for all 5 providers from S-norm cohort."""
    print("=== Provider Cosine Similarity Distributions ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    all_stats = []

    for i, (name, dim) in enumerate(PROVIDERS):
        emb_path = COHORT_DIR / f"cohort_embeddings_{name}.npy"
        emb = np.load(emb_path)  # (1000, dim)
        print(f"  {name}: loaded {emb.shape}, computing pairwise cosine similarities...")

        # Compute upper-triangle pairwise cosine similarities (499,500 pairs)
        cos_sim = cosine_similarity(emb)
        triu_idx = np.triu_indices(cos_sim.shape[0], k=1)
        scores = cos_sim[triu_idx]

        stats = {
            "provider": name,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "median": np.median(scores),
        }
        all_stats.append(stats)
        print(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")

        ax = axes[i]
        ax.hist(scores, bins=100, alpha=0.7, color=f"C{i}", edgecolor='black', linewidth=0.3)
        ax.set_title(f"{name} (dim={dim})", fontsize=13, fontweight='bold')
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=1.5, label=f"mean={stats['mean']:.3f}")
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Use last subplot for combined overlay
    ax = axes[5]
    for i, (name, dim) in enumerate(PROVIDERS):
        emb = np.load(COHORT_DIR / f"cohort_embeddings_{name}.npy")
        cos_sim = cosine_similarity(emb)
        scores = cos_sim[np.triu_indices(cos_sim.shape[0], k=1)]
        ax.hist(scores, bins=100, alpha=0.4, color=f"C{i}", label=name)
    ax.set_title("All Providers Overlaid", fontsize=13, fontweight='bold')
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Inter-Speaker Cosine Similarity Distributions (S-Norm Cohort, 1000 Speakers)",
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    out_path = REPORT_DIR / "provider_similarity_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")
    plt.close(fig)

    return all_stats


def plot_embedding_tsne():
    """t-SNE visualization of sampled embeddings from partial train_pool memmaps."""
    print("\n=== Embedding Space t-SNE ===")

    # Read checkpoint to know how many rows are valid
    ckpt_path = EMBED_DIR / "train_pool_checkpoint.txt"
    with open(ckpt_path, "r") as f:
        n_valid = int(f.read().strip())
    print(f"  Valid rows in memmap: {n_valid}")

    # Read index to get speaker_id and dataset_source
    print("  Reading index CSV...")
    index_path = EMBED_DIR / "train_pool_index.csv"
    speakers = []
    datasets = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speakers.append(row["speaker_id"])
            datasets.append(row["dataset_source"])

    # Sample strategy: pick ~30 speakers with enough utterances, sample up to 50 per speaker
    # This gives us ~1500 points which is good for t-SNE
    print("  Selecting speakers for visualization...")
    random.seed(42)

    # Count utterances per speaker in the valid range
    spk_indices = {}
    for idx in range(n_valid):
        spk = speakers[idx]
        if spk not in spk_indices:
            spk_indices[spk] = []
        spk_indices[spk].append(idx)

    # Pick speakers with >= 20 utterances, sample 30 of them
    eligible = [(spk, idxs) for spk, idxs in spk_indices.items() if len(idxs) >= 20]
    random.shuffle(eligible)
    selected_spks = eligible[:30]

    sample_indices = []
    sample_labels = []
    sample_datasets = []
    for spk, idxs in selected_spks:
        chosen = random.sample(idxs, min(50, len(idxs)))
        sample_indices.extend(chosen)
        sample_labels.extend([spk] * len(chosen))
        sample_datasets.extend([datasets[i] for i in chosen])

    n_samples = len(sample_indices)
    print(f"  Selected {n_samples} samples from {len(selected_spks)} speakers")

    # Load embeddings for each provider and run t-SNE
    for prov_name, prov_dim in TRAIN_PROVIDERS:
        print(f"\n  Processing {prov_name} (dim={prov_dim})...")
        mmap_path = EMBED_DIR / f"train_pool_{prov_name}.npy"
        mmap = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=(n_valid, prov_dim))

        # Extract sampled rows
        emb_sample = np.array([mmap[i] for i in sample_indices], dtype=np.float32)
        print(f"    Loaded sample: {emb_sample.shape}")

        # Verify no zero/nan rows
        norms = np.linalg.norm(emb_sample, axis=1)
        n_zero = np.sum(norms < 1e-6)
        n_nan = np.sum(np.isnan(norms))
        print(f"    Norms: min={norms.min():.4f}, max={norms.max():.4f}, zero={n_zero}, nan={n_nan}")

        if n_zero > 0 or n_nan > 0:
            print("    WARNING: skipping due to bad embeddings")
            continue

        # t-SNE (CPU, perplexity=30)
        print("    Running t-SNE (CPU)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
        coords = tsne.fit_transform(emb_sample)
        print(f"    t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

        # Plot colored by speaker
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Left: colored by speaker
        unique_spks = list(set(sample_labels))
        cmap = plt.cm.get_cmap('tab20', len(unique_spks))
        spk_to_color = {s: cmap(i) for i, s in enumerate(unique_spks)}
        colors = [spk_to_color[s] for s in sample_labels]

        ax1.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7)
        ax1.set_title(f"{prov_name} - Colored by Speaker ({len(unique_spks)} speakers)", fontsize=13, fontweight='bold')
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.grid(alpha=0.2)

        # Right: colored by dataset source
        unique_ds = list(set(sample_datasets))
        ds_colors_map = {ds: f"C{i}" for i, ds in enumerate(unique_ds)}
        ds_colors = [ds_colors_map[d] for d in sample_datasets]

        for ds in unique_ds:
            mask = [d == ds for d in sample_datasets]
            ax2.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7,
                       label=ds, color=ds_colors_map[ds])
        ax2.set_title(f"{prov_name} - Colored by Dataset", fontsize=13, fontweight='bold')
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend(fontsize=10, markerscale=3)
        ax2.grid(alpha=0.2)

        fig.suptitle(f"Embedding Space Visualization ({prov_name}, {n_samples} samples, t-SNE)",
                     fontsize=15, fontweight='bold', y=1.01)
        fig.tight_layout()
        out_path = REPORT_DIR / f"embedding_tsne_{prov_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    plot_provider_distributions()
    plot_embedding_tsne()
    print("\n=== All visualizations complete ===")
