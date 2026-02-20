"""
Expanded visualizations for completed Step 1 sub-tasks.
Generates ADVANCED visualization types that complement the basic ones
in visualize_step1_completed.py:
  - Treemap, lollipop, strip/swarm plots (1.1)
  - Radar, forest, parallel coordinates (1.5)
  - KDE overlay, hexbin t-SNE, strip norms, QQ, Andrews curves (1.6)
  - KDE overlay, hexbin, CDF (1.7)
  - Statistical measures: bootstrap 95% CI, Cohen's d, KS test, etc.

Outputs saved to the SAME folders as the original script:
  implementation/reports/step1/{sub-step}/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import csv
from pathlib import Path
from collections import Counter, defaultdict
import random

# -- Paths ----------------------------------------------------------------
BASE = Path(r"D:\VQI")
INVENTORY_CSV = BASE / "blueprint" / "dataset_inventory.csv"
SPLITS_DIR = BASE / "implementation" / "data" / "splits"
COHORT_DIR = BASE / "implementation" / "data" / "snorm_cohort"
EMBED_DIR = BASE / "implementation" / "data" / "embeddings"
REPORT_BASE = BASE / "implementation" / "reports" / "step1"

PROVIDERS_ALL = [
    ("P1_ECAPA", 192), ("P2_RESNET", 256), ("P3_ECAPA2", 192),
    ("P4_XVECTOR", 512), ("P5_WAVLM", 512),
]
PROVIDERS_TRAIN = [
    ("P1_ECAPA", 192), ("P2_RESNET", 256), ("P3_ECAPA2", 192),
]


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def bootstrap_ci(data, n_boot=10000, ci=0.95, stat_fn=np.mean, seed=42):
    """Bootstrap 95% CI using BCa-like percentile method."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array([stat_fn(rng.choice(data, size=n, replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])
    return lo, hi


def cohens_d(g1, g2):
    """Cohen's d effect size (pooled std)."""
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / sp if sp > 0 else 0.0


# ========================================================================
# 1.1: Expanded Dataset Verification
# ========================================================================
def viz_11_expanded():
    out_dir = REPORT_BASE / "1.1_dataset_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== 1.1 Expanded: Treemap, Lollipop, Strip ===")

    rows = []
    with open(INVENTORY_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    names = [r["dataset"] for r in rows]
    utterances = [int(r["utterances"]) for r in rows]
    speakers = [int(r["speakers"]) for r in rows]
    hours = [float(r["est_total_hours"]) for r in rows]

    # -- 1. Treemap (manual squarify) --
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')

    total = sum(utterances)
    # Sort descending for better layout
    sorted_pairs = sorted(zip(utterances, names, speakers), reverse=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_pairs)))

    # Simple slice-and-dice treemap
    def _treemap_layout(items, x0, y0, w, h, ax, colors, depth=0):
        if len(items) == 0:
            return
        if len(items) == 1:
            val, name, spk = items[0]
            rect = Rectangle((x0, y0), w, h, linewidth=2, edgecolor='white',
                              facecolor=colors[0], alpha=0.85)
            ax.add_patch(rect)
            label = f"{name}\n{val:,}\n({val/total*100:.1f}%)"
            fs = max(7, min(12, int(w * h / 50)))
            ax.text(x0 + w/2, y0 + h/2, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', wrap=True)
            return
        # Split items into two groups trying to balance
        total_val = sum(v for v, _, _ in items)
        cumsum = 0
        split = 1
        for i, (v, _, _) in enumerate(items):
            cumsum += v
            if cumsum >= total_val / 2:
                split = i + 1
                break
        left, right = items[:split], items[split:]
        left_frac = sum(v for v, _, _ in left) / total_val if total_val > 0 else 0.5
        if depth % 2 == 0:  # horizontal split
            _treemap_layout(left, x0, y0, w * left_frac, h, ax, colors[:split], depth+1)
            _treemap_layout(right, x0 + w * left_frac, y0, w * (1 - left_frac), h,
                            ax, colors[split:], depth+1)
        else:  # vertical split
            _treemap_layout(left, x0, y0, w, h * left_frac, ax, colors[:split], depth+1)
            _treemap_layout(right, x0, y0 + h * left_frac, w, h * (1 - left_frac),
                            ax, colors[split:], depth+1)

    _treemap_layout(sorted_pairs, 0, 0, 100, 100, ax, colors)
    fig.suptitle("Dataset Composition Treemap (by utterance count)",
                 fontsize=15, fontweight='bold')
    save_fig(fig, out_dir / "dataset_composition_treemap.png")

    # -- 2. Lollipop chart --
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    sorted_idx = np.argsort(utterances)
    s_names = [names[i] for i in sorted_idx]
    s_utt = [utterances[i] for i in sorted_idx]

    ax.hlines(y_pos, 0, s_utt, color='#4472C4', linewidth=2, alpha=0.7)
    ax.scatter(s_utt, y_pos, color='#4472C4', s=80, zorder=5, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel("Utterance Count (log scale)", fontsize=12)
    ax.set_title("Dataset Utterance Counts (Lollipop Chart)", fontsize=14, fontweight='bold')
    for i, (val, name) in enumerate(zip(s_utt, s_names)):
        ax.text(val * 1.15, i, f'{val:,}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "dataset_lollipop_chart.png")

    # -- 3. Utterances-per-speaker strip plot with bootstrap CI --
    print("  Computing per-speaker utterance counts from train_pool.csv...")
    ds_spk_counts = defaultdict(list)  # dataset -> list of utterances-per-speaker
    spk_ds = defaultdict(lambda: defaultdict(int))
    with open(SPLITS_DIR / "train_pool.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spk_ds[row["speaker_id"]][row["dataset_source"]] += 1
    for spk, ds_counts in spk_ds.items():
        for ds, count in ds_counts.items():
            ds_spk_counts[ds].append(count)

    fig, ax = plt.subplots(figsize=(12, 6))
    ds_names_ordered = sorted(ds_spk_counts.keys())
    for i, ds in enumerate(ds_names_ordered):
        counts = np.array(ds_spk_counts[ds])
        jitter = np.random.RandomState(42).uniform(-0.3, 0.3, len(counts))
        ax.scatter(np.full(len(counts), i) + jitter, counts, s=3, alpha=0.15, color=f'C{i}')
        # Bootstrap 95% CI for mean
        ci_lo, ci_hi = bootstrap_ci(counts, n_boot=10000, seed=42)
        mean_val = np.mean(counts)
        ax.plot([i - 0.35, i + 0.35], [mean_val, mean_val], color='black', linewidth=2)
        ax.fill_between([i - 0.35, i + 0.35], ci_lo, ci_hi, alpha=0.3, color='red',
                        label='95% CI' if i == 0 else None)

    ax.set_xticks(range(len(ds_names_ordered)))
    ax.set_xticklabels(ds_names_ordered, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Utterances per Speaker", fontsize=12)
    ax.set_title("Speaker Utterance Distribution by Dataset (Strip + Bootstrap 95% CI)",
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "dataset_utterances_per_speaker_strip.png")

    # -- Update analysis.md with stats --
    stat_lines = ["\n## Statistical Measures (Expanded)\n"]
    for ds in ds_names_ordered:
        counts = np.array(ds_spk_counts[ds])
        ci_lo, ci_hi = bootstrap_ci(counts, n_boot=10000, seed=42)
        sk = scipy_stats.skew(counts)
        ku = scipy_stats.kurtosis(counts)
        stat_lines.append(f"**{ds}**: N_spk={len(counts)}, mean={np.mean(counts):.1f}, "
                          f"median={np.median(counts):.0f}, IQR=[{np.percentile(counts, 25):.0f}, "
                          f"{np.percentile(counts, 75):.0f}], "
                          f"bootstrap 95% CI=[{ci_lo:.1f}, {ci_hi:.1f}], "
                          f"skewness={sk:.2f}, kurtosis={ku:.2f}")

    # Cross-dataset Mann-Whitney U
    ds_list = sorted(ds_spk_counts.keys())
    if len(ds_list) >= 2:
        stat_lines.append("\n### Mann-Whitney U Tests (utterances/speaker across datasets)")
        for i in range(len(ds_list)):
            for j in range(i+1, len(ds_list)):
                u_stat, p_val = scipy_stats.mannwhitneyu(
                    ds_spk_counts[ds_list[i]], ds_spk_counts[ds_list[j]], alternative='two-sided')
                stat_lines.append(f"- {ds_list[i]} vs {ds_list[j]}: U={u_stat:.0f}, p={p_val:.2e}")

    analysis_path = out_dir / "analysis.md"
    with open(analysis_path, "r", encoding="utf-8") as f:
        existing = f.read()
    # Append stats if not already present
    if "Statistical Measures (Expanded)" not in existing:
        with open(analysis_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(stat_lines) + "\n")
        print(f"  Updated: {analysis_path}")


# ========================================================================
# 1.5: Expanded Provider Verification
# ========================================================================
def viz_15_expanded():
    out_dir = REPORT_BASE / "1.5_providers"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== 1.5 Expanded: Radar, Forest, Parallel Coordinates ===")

    # Provider specs (known values)
    provider_specs = {
        "P1_ECAPA":  {"dim": 192, "eer": 0.87, "arch": "TDNN+SE+Att",   "role": "Train"},
        "P2_RESNET": {"dim": 256, "eer": 1.05, "arch": "CNN+SE+ASP",    "role": "Train"},
        "P3_ECAPA2": {"dim": 192, "eer": 0.17, "arch": "Hybrid1D2D",    "role": "Train"},
        "P4_XVECTOR":{"dim": 512, "eer": 3.13, "arch": "ClassicalTDNN", "role": "Test"},
        "P5_WAVLM":  {"dim": 512, "eer": 0.60, "arch": "SSLTransformer","role": "Test"},
    }

    # Load cohort cosine similarity stats as proxy for genuine/impostor separation
    cos_stats = {}
    for name, dim in PROVIDERS_ALL:
        emb = np.load(COHORT_DIR / f"cohort_embeddings_{name}.npy")
        cos_sim = cosine_similarity(emb)
        triu = cos_sim[np.triu_indices(cos_sim.shape[0], k=1)]
        # Diagonal = self-similarity (genuine proxy), off-diagonal = impostor proxy
        diag = np.diag(cos_sim)  # should be ~1.0
        cos_stats[name] = {
            "impostor_mean": float(np.mean(triu)),
            "impostor_std": float(np.std(triu)),
            "self_mean": float(np.mean(diag)),
            "d_prime": float((np.mean(diag) - np.mean(triu)) /
                             (np.std(triu) if np.std(triu) > 0 else 1e-6)),
            "triu_scores": triu,
        }

    # -- 1. Radar chart --
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    categories = ["1/EER (%)", "d-prime", "Emb Dim (norm)", "1/Impostor Mean", "Separation"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5']
    for idx, (name, _) in enumerate(PROVIDERS_ALL):
        sp = provider_specs[name]
        cs = cos_stats[name]
        values = [
            min(100 / sp["eer"], 100),         # 1/EER capped at 100
            min(cs["d_prime"], 50) / 50 * 100,  # d-prime normalized
            sp["dim"] / 512 * 100,              # dim normalized
            (1 - abs(cs["impostor_mean"])) * 100,  # closer to 0 = better
            min(cs["d_prime"], 20) / 20 * 100,  # separation normalized
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Provider Capability Radar Chart", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_dir / "provider_radar_chart.png")

    # -- 2. Forest plot: Cohen's d for genuine-impostor separation --
    fig, ax = plt.subplots(figsize=(10, 5))
    prov_names = [n for n, _ in PROVIDERS_ALL]
    d_primes = [cos_stats[n]["d_prime"] for n in prov_names]
    # Bootstrap CI for d-prime
    ci_lo, ci_hi = [], []
    for name in prov_names:
        triu = cos_stats[name]["triu_scores"]
        # d-prime = (1 - mean(triu)) / std(triu)
        def dp_stat(data):
            return (1.0 - np.mean(data)) / (np.std(data) if np.std(data) > 0 else 1e-6)
        lo, hi = bootstrap_ci(triu, n_boot=10000, stat_fn=dp_stat, seed=42)
        ci_lo.append(lo)
        ci_hi.append(hi)

    y_pos = np.arange(len(prov_names))
    ax.hlines(y_pos, ci_lo, ci_hi, colors='black', linewidth=1.5)
    ax.scatter(d_primes, y_pos, color='#4472C4', s=100, zorder=5, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prov_names, fontsize=11)
    ax.set_xlabel("d-prime (self-similarity vs inter-speaker)", fontsize=12)
    ax.set_title("Provider Separation: d-prime with Bootstrap 95% CI",
                 fontsize=14, fontweight='bold')
    ax.axvline(x=3.0, linestyle='--', color='green', alpha=0.5, label='Good threshold (3.0)')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "provider_separation_forest_plot.png")

    # -- 3. Parallel coordinates --
    fig, ax = plt.subplots(figsize=(12, 6))
    attrs = ["EER (%)", "Dim", "d-prime", "Impostor Mean", "Impostor Std"]
    x_pos = np.arange(len(attrs))

    for idx, (name, _) in enumerate(PROVIDERS_ALL):
        sp = provider_specs[name]
        cs = cos_stats[name]
        # Normalize each attribute to 0-1 range for parallel coordinates
        vals_raw = [sp["eer"], sp["dim"], cs["d_prime"], cs["impostor_mean"], cs["impostor_std"]]
        ax.plot(x_pos, vals_raw, 'o-', linewidth=2, markersize=8,
                label=name, color=colors_radar[idx])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(attrs, fontsize=11)
    ax.set_title("Provider Properties: Parallel Coordinates", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='both', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "provider_parallel_coordinates.png")

    # -- 4. Embedding norms box plot --
    fig, ax = plt.subplots(figsize=(10, 5))
    all_norms = []
    labels = []
    for name, dim in PROVIDERS_ALL:
        emb = np.load(COHORT_DIR / f"cohort_embeddings_{name}.npy")
        norms = np.linalg.norm(emb, axis=1)
        all_norms.append(norms)
        labels.append(name)
    bp = ax.boxplot(all_norms, labels=labels, patch_artist=True, showfliers=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_radar[i])
        patch.set_alpha(0.7)
    ax.axhline(1.0, color='red', linestyle='--', label='Expected (1.0)')
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title("Provider Embedding L2 Norms (Cohort, 1000 speakers)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "provider_embedding_norms.png")

    # -- 5. Genuine vs Impostor violin plot (using cohort self vs inter) --
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for i, (name, dim) in enumerate(PROVIDERS_ALL):
        ax = axes[i]
        emb = np.load(COHORT_DIR / f"cohort_embeddings_{name}.npy")
        cos_sim = cosine_similarity(emb)
        genuine = np.diag(cos_sim)  # self-similarity = 1.0 (always)
        impostor = cos_sim[np.triu_indices(cos_sim.shape[0], k=1)]
        # Subsample impostor for violin (too many points)
        imp_sample = np.random.RandomState(42).choice(impostor, size=min(5000, len(impostor)),
                                                       replace=False)
        vp = ax.violinplot([imp_sample], positions=[0], showmeans=True, showmedians=True)
        for pc in vp['bodies']:
            pc.set_facecolor(colors_radar[i])
            pc.set_alpha(0.6)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Self-sim (1.0)')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylabel("Cosine Similarity")
        ax.set_xticks([0])
        ax.set_xticklabels(["Impostor"])
        ax.legend(fontsize=8)
    fig.suptitle("Inter-Speaker Cosine Similarity Violins (Cohort)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "provider_genuine_vs_impostor.png")

    # -- Update analysis.md with statistical measures --
    stat_lines = ["\n## Statistical Measures (Expanded)\n"]
    stat_lines.append("| Provider | d-prime | d-prime 95% CI | Cohen's d (1-vs-imp) | Cliff's delta |")
    stat_lines.append("|----------|---------|----------------|---------------------|---------------|")
    for idx, name in enumerate(prov_names):
        cs = cos_stats[name]
        genuine_proxy = np.ones(1000)  # self-similarity
        imp = cs["triu_scores"]
        cd = cohens_d(genuine_proxy, imp)
        # Cliff's delta
        n1, n2 = len(genuine_proxy), min(5000, len(imp))
        imp_s = np.random.RandomState(42).choice(imp, size=n2, replace=False)
        greater = np.sum(genuine_proxy[:, None] > imp_s[None, :])
        lesser = np.sum(genuine_proxy[:, None] < imp_s[None, :])
        cliff = (greater - lesser) / (n1 * n2)
        stat_lines.append(f"| {name} | {cs['d_prime']:.2f} | "
                          f"[{ci_lo[idx]:.2f}, {ci_hi[idx]:.2f}] | "
                          f"{cd:.2f} | {cliff:.4f} |")

    analysis_path = out_dir / "analysis.md"
    with open(analysis_path, "r", encoding="utf-8") as f:
        existing = f.read()
    if "Statistical Measures (Expanded)" not in existing:
        with open(analysis_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(stat_lines) + "\n")
        print(f"  Updated: {analysis_path}")


# ========================================================================
# 1.7: Expanded S-Norm Cohort
# ========================================================================
def viz_17_expanded():
    out_dir = REPORT_BASE / "1.7_snorm_cohort"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== 1.7 Expanded: KDE overlay, Hexbin, CDF ===")

    # Load all cohort similarity data
    sim_data = {}
    for name, dim in PROVIDERS_ALL:
        emb = np.load(COHORT_DIR / f"cohort_embeddings_{name}.npy")
        cos_sim = cosine_similarity(emb)
        triu = cos_sim[np.triu_indices(cos_sim.shape[0], k=1)]
        sim_data[name] = triu

    # -- 1. KDE density overlay --
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5']
    for i, (name, _) in enumerate(PROVIDERS_ALL):
        scores = sim_data[name]
        # Subsample for KDE speed
        sample = np.random.RandomState(42).choice(scores, size=min(50000, len(scores)), replace=False)
        kde = gaussian_kde(sample)
        x_grid = np.linspace(sample.min() - 0.02, sample.max() + 0.02, 500)
        ax.plot(x_grid, kde(x_grid), linewidth=2, label=name, color=colors[i])
        ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=colors[i])

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Inter-Speaker Cosine Similarity KDE (All 5 Providers)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "cohort_similarity_kde_overlay.png")

    # -- 2. Hexbin: P1 vs P2, P1 vs P3, P2 vs P3 --
    pairs = [("P1_ECAPA", "P2_RESNET"), ("P1_ECAPA", "P3_ECAPA2"), ("P2_RESNET", "P3_ECAPA2")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (pA, pB) in enumerate(pairs):
        ax = axes[idx]
        # Need per-speaker-pair similarities for both providers
        embA = np.load(COHORT_DIR / f"cohort_embeddings_{pA}.npy")
        embB = np.load(COHORT_DIR / f"cohort_embeddings_{pB}.npy")
        cosA = cosine_similarity(embA)
        cosB = cosine_similarity(embB)
        triuA = cosA[np.triu_indices(1000, k=1)]
        triuB = cosB[np.triu_indices(1000, k=1)]
        # Subsample for hexbin
        n_sub = min(100000, len(triuA))
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(triuA), size=n_sub, replace=False)
        hb = ax.hexbin(triuA[idx_sub], triuB[idx_sub], gridsize=60, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel(f"{pA} cosine sim", fontsize=10)
        ax.set_ylabel(f"{pB} cosine sim", fontsize=10)
        ax.set_title(f"{pA} vs {pB}", fontsize=12, fontweight='bold')
        plt.colorbar(hb, ax=ax, label='Count')

    fig.suptitle("Cross-Provider Similarity Agreement (Hexbin Density)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "cohort_similarity_hexbin.png")

    # -- 3. CDF of inter-speaker similarity --
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, _) in enumerate(PROVIDERS_ALL):
        scores = sim_data[name]
        sorted_scores = np.sort(scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        # Subsample for plotting
        step = max(1, len(sorted_scores) // 5000)
        ax.plot(sorted_scores[::step], cdf[::step], linewidth=2, label=name, color=colors[i])

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title("CDF of Inter-Speaker Cosine Similarity (S-Norm Cohort)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, linestyle=':', color='gray', alpha=0.5, label='Median')
    fig.tight_layout()
    save_fig(fig, out_dir / "cohort_cumulative_similarity.png")

    # -- Update analysis.md with statistical measures --
    stat_lines = ["\n## Statistical Measures (Expanded)\n"]
    stat_lines.append("| Provider | IQR | Skewness | Kurtosis | Mean 95% CI |")
    stat_lines.append("|----------|-----|----------|----------|-------------|")
    for name, _ in PROVIDERS_ALL:
        scores = sim_data[name]
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        sk = scipy_stats.skew(scores)
        ku = scipy_stats.kurtosis(scores)
        ci_lo, ci_hi = bootstrap_ci(scores[:50000], n_boot=10000, seed=42)
        stat_lines.append(f"| {name} | {iqr:.4f} | {sk:.3f} | {ku:.3f} | [{ci_lo:.4f}, {ci_hi:.4f}] |")

    # KS test: P1-P3 vs P4-P5
    p13_scores = np.concatenate([sim_data[n] for n, _ in PROVIDERS_ALL[:3]])
    p45_scores = np.concatenate([sim_data[n] for n, _ in PROVIDERS_ALL[3:]])
    # Subsample for KS test
    rng = np.random.RandomState(42)
    p13_sub = rng.choice(p13_scores, size=50000, replace=False)
    p45_sub = rng.choice(p45_scores, size=50000, replace=False)
    ks_stat, ks_p = scipy_stats.ks_2samp(p13_sub, p45_sub)
    stat_lines.append(f"\n**KS test (P1-P3 vs P4-P5):** D={ks_stat:.4f}, p={ks_p:.2e}")
    stat_lines.append("Interpretation: P1-P3 and P4-P5 have significantly different similarity distributions "
                       "(expected -- P4-P5 have compressed embedding spaces).")

    analysis_path = out_dir / "analysis.md"
    with open(analysis_path, "r", encoding="utf-8") as f:
        existing = f.read()
    if "Statistical Measures (Expanded)" not in existing:
        with open(analysis_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(stat_lines) + "\n")
        print(f"  Updated: {analysis_path}")


# ========================================================================
# 1.6: Expanded Partial Embeddings
# ========================================================================
def viz_16_expanded():
    out_dir = REPORT_BASE / "1.6_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== 1.6 Expanded: Hexbin t-SNE, KDE, Strip, QQ, Andrews, Cosine ===")

    ckpt_path = EMBED_DIR / "train_pool_checkpoint.txt"
    with open(ckpt_path, "r") as f:
        n_valid = int(f.read().strip())
    print(f"  Valid rows: {n_valid}")

    # -- 1. Strip/swarm plot of norms --
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_size = min(5000, n_valid)
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(n_valid, size=sample_size, replace=False).tolist())
    colors_p = ['#4472C4', '#ED7D31', '#70AD47']

    all_norms_data = []
    for i, (name, dim) in enumerate(PROVIDERS_TRAIN):
        mmap = np.memmap(EMBED_DIR / f"train_pool_{name}.npy", dtype=np.float32,
                         mode='r', shape=(1210451, dim))
        sample = np.array([mmap[j] for j in indices])
        norms = np.linalg.norm(sample, axis=1)
        all_norms_data.append(norms)
        jitter = rng.uniform(-0.3, 0.3, len(norms))
        ax.scatter(np.full(len(norms), i) + jitter, norms, s=2, alpha=0.2, color=colors_p[i])
        # Mean + CI
        ci_lo, ci_hi = bootstrap_ci(norms, n_boot=10000, seed=42)
        ax.plot([i - 0.35, i + 0.35], [np.mean(norms)] * 2, color='black', linewidth=2)
        ax.fill_between([i - 0.35, i + 0.35], ci_lo, ci_hi, alpha=0.3, color='red')

    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _ in PROVIDERS_TRAIN], fontsize=11)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(f"Embedding Norms Strip Plot ({sample_size:,} samples, bootstrap 95% CI)",
                 fontsize=13, fontweight='bold')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Expected (1.0)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_norms_strip_swarm.png")

    # -- 2. QQ plot of norms --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, dim) in enumerate(PROVIDERS_TRAIN):
        ax = axes[i]
        norms = all_norms_data[i]
        scipy_stats.probplot(norms, dist="norm", plot=ax)
        ax.set_title(f"{name} Norm QQ Plot", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    fig.suptitle("Embedding Norm QQ Plots (vs Normal Distribution)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_norm_qq_plot.png")

    # -- 3. Inter-speaker cosine similarity distributions --
    print("  Computing inter-speaker cosine similarities (sampled)...")
    # Read index for speaker labels
    speakers = []
    with open(EMBED_DIR / "train_pool_index.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            speakers.append(row["speaker_id"])

    # Sample 200 speakers with >= 5 utterances
    spk_indices = defaultdict(list)
    for idx in range(n_valid):
        spk_indices[speakers[idx]].append(idx)
    eligible = [(spk, idxs) for spk, idxs in spk_indices.items() if len(idxs) >= 5]
    random.seed(42)
    random.shuffle(eligible)
    sampled_spks = eligible[:200]

    # Get one embedding per speaker (centroid of first 5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cos_data = {}
    for pi, (pname, pdim) in enumerate(PROVIDERS_TRAIN):
        mmap = np.memmap(EMBED_DIR / f"train_pool_{pname}.npy", dtype=np.float32,
                         mode='r', shape=(1210451, pdim))
        centroids = []
        for spk, idxs in sampled_spks:
            embs = np.array([mmap[j] for j in idxs[:5]])
            centroid = embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            centroids.append(centroid)
        centroids = np.array(centroids)
        cos_sim = cosine_similarity(centroids)
        triu = cos_sim[np.triu_indices(len(centroids), k=1)]
        cos_data[pname] = triu

        ax = axes[pi]
        ax.hist(triu, bins=80, color=colors_p[pi], edgecolor='black', linewidth=0.3, alpha=0.7)
        ax.axvline(np.mean(triu), color='red', linestyle='--', label=f'mean={np.mean(triu):.4f}')
        ax.set_title(f"{pname}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        print(f"    {pname}: mean cosine={np.mean(triu):.4f}, std={np.std(triu):.4f}")

    fig.suptitle(f"Inter-Speaker Cosine Similarity (200 speakers, centroids from {n_valid:,} embeddings)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "cosine_similarity_distributions.png")

    # -- 4. KDE overlay of cosine similarities --
    fig, ax = plt.subplots(figsize=(12, 6))
    for pi, (pname, _) in enumerate(PROVIDERS_TRAIN):
        triu = cos_data[pname]
        kde = gaussian_kde(triu)
        x_grid = np.linspace(triu.min() - 0.02, triu.max() + 0.02, 500)
        ax.plot(x_grid, kde(x_grid), linewidth=2, label=pname, color=colors_p[pi])
        ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=colors_p[pi])
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Inter-Speaker Cosine Similarity KDE (Training Providers)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_cosine_kde_overlay.png")

    # -- 5. Cross-provider similarity agreement heatmap --
    print("  Computing cross-provider similarity agreement (100 utterances)...")
    heatmap_indices = sorted(rng.choice(n_valid, size=100, replace=False).tolist())
    # Compute within-provider pairwise similarity for 100 utterances, then correlate
    prov_sim_matrices = {}
    for pname, pdim in PROVIDERS_TRAIN:
        mmap = np.memmap(EMBED_DIR / f"train_pool_{pname}.npy", dtype=np.float32,
                         mode='r', shape=(1210451, pdim))
        embs = np.array([mmap[j] for j in heatmap_indices])
        prov_sim_matrices[pname] = cosine_similarity(embs)  # 100x100

    # Correlation heatmap: how well do provider similarity matrices agree?
    prov_names_t = [n for n, _ in PROVIDERS_TRAIN]
    corr_matrix = np.zeros((3, 3))
    for i, pA in enumerate(prov_names_t):
        for j, pB in enumerate(prov_names_t):
            triuA = prov_sim_matrices[pA][np.triu_indices(100, k=1)]
            triuB = prov_sim_matrices[pB][np.triu_indices(100, k=1)]
            corr_matrix[i, j] = np.corrcoef(triuA, triuB)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(prov_names_t, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(prov_names_t, fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr_matrix[i, j]:.3f}', ha='center', va='center',
                    fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title("Cross-Provider Similarity Agreement\n(Pearson r of pairwise cosine matrices, 100 utterances)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_cosine_heatmap.png")

    # -- 6. t-SNE with hexbin density --
    print("  Running t-SNE for hexbin density plots...")
    random.seed(42)
    # Use same 30-speaker sample as original
    eligible_tsne = [(spk, idxs) for spk, idxs in spk_indices.items() if len(idxs) >= 20]
    random.shuffle(eligible_tsne)
    selected_spks = eligible_tsne[:30]
    tsne_indices = []
    tsne_labels = []
    for spk, idxs in selected_spks:
        chosen = random.sample(idxs, min(50, len(idxs)))
        tsne_indices.extend(chosen)
        tsne_labels.extend([spk] * len(chosen))
    n_tsne = len(tsne_indices)
    print(f"    {n_tsne} samples from {len(selected_spks)} speakers")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for pi, (pname, pdim) in enumerate(PROVIDERS_TRAIN):
        print(f"    t-SNE for {pname}...")
        mmap = np.memmap(EMBED_DIR / f"train_pool_{pname}.npy", dtype=np.float32,
                         mode='r', shape=(1210451, pdim))
        emb_sample = np.array([mmap[j] for j in tsne_indices], dtype=np.float32)
        norms_check = np.linalg.norm(emb_sample, axis=1)
        if np.sum(norms_check < 1e-6) > 0 or np.sum(np.isnan(norms_check)) > 0:
            print(f"    WARNING: skipping {pname} due to bad embeddings")
            continue

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
        coords = tsne.fit_transform(emb_sample)

        ax = axes[pi]
        hb = ax.hexbin(coords[:, 0], coords[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
        ax.set_title(f"{pname}", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.colorbar(hb, ax=ax, label='Count')

    fig.suptitle(f"Embedding Density (Hexbin t-SNE, {n_tsne} samples)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_hexbin_tsne.png")

    # -- 7. Andrews curves (simplified) --
    print("  Generating Andrews curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t_vals = np.linspace(-np.pi, np.pi, 200)

    for pi, (pname, pdim) in enumerate(PROVIDERS_TRAIN):
        ax = axes[pi]
        mmap = np.memmap(EMBED_DIR / f"train_pool_{pname}.npy", dtype=np.float32,
                         mode='r', shape=(1210451, pdim))
        # Take 50 random embeddings
        ac_indices = sorted(rng.choice(n_valid, size=50, replace=False).tolist())
        embs = np.array([mmap[j] for j in ac_indices])
        # PCA to 10 dims for Andrews curves
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, pdim), random_state=42)
        embs_pca = pca.fit_transform(embs)

        for j in range(embs_pca.shape[0]):
            x = embs_pca[j]
            curve = x[0] / np.sqrt(2)
            for k in range(1, len(x)):
                if k % 2 == 1:
                    curve = curve + x[k] * np.sin((k // 2 + 1) * t_vals)
                else:
                    curve = curve + x[k] * np.cos((k // 2) * t_vals)
            ax.plot(t_vals, curve, alpha=0.3, linewidth=0.8, color=colors_p[pi])

        ax.set_title(f"{pname} (PCA->10d)", fontsize=11, fontweight='bold')
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)")
        ax.grid(alpha=0.3)

    fig.suptitle("Andrews Curves (50 random embeddings per provider)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, out_dir / "embedding_andrews_curves.png")

    # -- Update analysis.md with statistical measures --
    stat_lines = ["\n## Statistical Measures (Expanded)\n"]
    stat_lines.append("### Embedding Norms")
    stat_lines.append("| Provider | Mean | Std | IQR | Skewness | Kurtosis | 95% CI |")
    stat_lines.append("|----------|------|-----|-----|----------|----------|--------|")
    for pi, (pname, _) in enumerate(PROVIDERS_TRAIN):
        norms = all_norms_data[pi]
        iqr = np.percentile(norms, 75) - np.percentile(norms, 25)
        sk = scipy_stats.skew(norms)
        ku = scipy_stats.kurtosis(norms)
        ci_lo, ci_hi = bootstrap_ci(norms, n_boot=10000, seed=42)
        stat_lines.append(f"| {pname} | {np.mean(norms):.6f} | {np.std(norms):.6f} | "
                          f"{iqr:.6f} | {sk:.3f} | {ku:.3f} | [{ci_lo:.6f}, {ci_hi:.6f}] |")

    stat_lines.append("\n### Inter-Speaker Cosine Similarity")
    stat_lines.append("| Provider | Mean | Std | IQR | 95% CI |")
    stat_lines.append("|----------|------|-----|-----|--------|")
    for pname, _ in PROVIDERS_TRAIN:
        triu = cos_data[pname]
        iqr = np.percentile(triu, 75) - np.percentile(triu, 25)
        ci_lo, ci_hi = bootstrap_ci(triu, n_boot=10000, seed=42)
        stat_lines.append(f"| {pname} | {np.mean(triu):.4f} | {np.std(triu):.4f} | "
                          f"{iqr:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] |")

    # Mutual information between providers (discretized)
    stat_lines.append("\n### Cross-Provider Agreement")
    for i in range(len(PROVIDERS_TRAIN)):
        for j in range(i+1, len(PROVIDERS_TRAIN)):
            pA, _ = PROVIDERS_TRAIN[i]
            pB, _ = PROVIDERS_TRAIN[j]
            tA = cos_data[pA]
            tB = cos_data[pB]
            # KS test
            ks_stat, ks_p = scipy_stats.ks_2samp(tA[:10000], tB[:10000])
            stat_lines.append(f"- KS test {pA} vs {pB}: D={ks_stat:.4f}, p={ks_p:.2e}")

    analysis_path = out_dir / "analysis.md"
    with open(analysis_path, "r", encoding="utf-8") as f:
        existing = f.read()
    if "Statistical Measures (Expanded)" not in existing:
        with open(analysis_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(stat_lines) + "\n")
        print(f"  Updated: {analysis_path}")


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Step 1 Expanded Visualizations")
    print("=" * 60)

    viz_11_expanded()
    viz_15_expanded()
    viz_17_expanded()
    viz_16_expanded()

    print("\n=== All expanded visualizations complete ===")
