"""Step 1.6b: Compute provider comparison scores.

Reads memmap embeddings from extract_embeddings.py and cohort embeddings
from build_snorm_cohort.py, then computes:
  1. Speaker centroids (mean embedding per speaker per provider)
  2. Leave-one-out genuine scores (raw cosine similarity)
  3. S-norm normalized genuine scores
  4. Impostor score distributions (centroid cross-comparisons + s-norm)
  5. Output CSVs, impostor .npy files, and summary statistics

Usage:
    python implementation/scripts/compute_scores.py
    python implementation/scripts/compute_scores.py --split train_pool --providers P1_ECAPA P2_RESNET P3_ECAPA2
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import TRAIN_PROVIDERS, PROVIDERS

EMBEDDINGS_DIR = os.path.join("implementation", "data", "step1", "embeddings")
COHORT_DIR = os.path.join("implementation", "data", "step1", "snorm_cohort")
OUTPUT_DIR = os.path.join("implementation", "data", "step1", "provider_scores")

PROVIDER_DIMS = {
    "P1_ECAPA": 192,
    "P2_RESNET": 256,
    "P3_ECAPA2": 192,
    "P4_XVECTOR": 512,
    "P5_WAVLM": 512,
}

PROVIDER_SHORT = {
    "P1_ECAPA": "ecapa",
    "P2_RESNET": "resnet",
    "P3_ECAPA2": "ecapa2",
    "P4_XVECTOR": "xvector",
    "P5_WAVLM": "wavlm",
}

# Process sample-to-cohort scores in chunks to limit RAM
CHUNK_SIZE = 10000


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows (or a single vector)."""
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-12 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def main():
    parser = argparse.ArgumentParser(description="Compute provider comparison scores")
    parser.add_argument(
        "--split", default="train_pool",
        help="Split name (must match extract_embeddings output)"
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Provider names (default: TRAIN_PROVIDERS)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    provider_names = args.providers if args.providers else list(TRAIN_PROVIDERS)
    for pn in provider_names:
        if pn not in PROVIDERS:
            print(f"ERROR: Unknown provider '{pn}'")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load index ---
    index_path = os.path.join(EMBEDDINGS_DIR, f"{args.split}_index.csv")
    print(f"Loading index from {index_path}...")
    row_indices = []
    filenames = []
    speaker_ids = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_indices.append(int(row["row_idx"]))
            filenames.append(row["filename"])
            speaker_ids.append(row["speaker_id"])
    total = len(filenames)
    print(f"  {total} samples")

    # Build speaker -> row indices mapping
    speaker_to_rows = {}
    for i, spk in enumerate(speaker_ids):
        if spk not in speaker_to_rows:
            speaker_to_rows[spk] = []
        speaker_to_rows[spk].append(i)
    unique_speakers = sorted(speaker_to_rows.keys())
    n_speakers = len(unique_speakers)
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
    print(f"  {n_speakers} unique speakers")

    # --- Load cohort ---
    print(f"\nLoading s-norm cohort from {COHORT_DIR}...")
    cohort_speakers_path = os.path.join(COHORT_DIR, "cohort_speakers.txt")
    with open(cohort_speakers_path, "r", encoding="utf-8") as f:
        cohort_speakers = [line.strip() for line in f if line.strip()]
    print(f"  {len(cohort_speakers)} cohort speakers")

    # --- Check for checkpoint ---
    checkpoint_path = os.path.join(OUTPUT_DIR, "_checkpoint.json")
    completed_providers = set()
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        completed_providers = set(ckpt.get("completed_providers", []))
        print(f"  Checkpoint found: {len(completed_providers)} providers already done: {completed_providers}")
    elif not args.resume:
        # Fresh start — remove stale checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # --- Process each provider ---
    all_stats = {}

    # Load stats from already-completed providers
    stats_path = os.path.join(OUTPUT_DIR, "score_statistics.json")
    if completed_providers and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            all_stats = json.load(f)

    for pn in provider_names:
        if pn in completed_providers:
            print(f"\n  Skipping {pn} (already completed, found in checkpoint)")
            continue
        print(f"\n{'=' * 60}")
        print(f"PROVIDER: {pn}")
        print(f"{'=' * 60}")
        t0 = time.time()

        dim = PROVIDER_DIMS[pn]
        short = PROVIDER_SHORT[pn]

        # Load memmap (read-only)
        memmap_path = os.path.join(EMBEDDINGS_DIR, f"{args.split}_{pn}.npy")
        embeddings = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(total, dim))
        print(f"  Loaded embeddings: {memmap_path} shape={embeddings.shape}")

        # Load cohort embeddings
        cohort_path = os.path.join(COHORT_DIR, f"cohort_embeddings_{pn}.npy")
        cohort_embs = np.load(cohort_path)  # (1000, dim)
        print(f"  Loaded cohort: {cohort_path} shape={cohort_embs.shape}")

        # ---- Phase 1: Speaker centroids ----
        print("\n  Phase 1: Computing speaker centroids...")
        centroids = np.zeros((n_speakers, dim), dtype=np.float32)
        speaker_counts = np.zeros(n_speakers, dtype=np.int32)

        for spk, rows in speaker_to_rows.items():
            sidx = speaker_to_idx[spk]
            spk_embs = np.array([embeddings[r] for r in rows])
            # Skip zero rows (errors during extraction)
            valid_mask = np.linalg.norm(spk_embs, axis=1) > 1e-6
            if valid_mask.sum() > 0:
                mean_emb = spk_embs[valid_mask].mean(axis=0)
                centroids[sidx] = l2_normalize(mean_emb)
                speaker_counts[sidx] = int(valid_mask.sum())
            else:
                speaker_counts[sidx] = 0

        print(f"    {n_speakers} centroids computed")
        print(f"    Speakers with 1 utterance: {(speaker_counts == 1).sum()}")
        print(f"    Speakers with 0 valid utts: {(speaker_counts == 0).sum()}")

        # ---- Phase 2: Genuine scores (leave-one-out) ----
        print("\n  Phase 2: Computing leave-one-out genuine scores...")
        genuine_raw = np.full(total, np.nan, dtype=np.float32)

        for spk, rows in speaker_to_rows.items():
            sidx = speaker_to_idx[spk]
            n_s = speaker_counts[sidx]
            if n_s <= 1:
                # Single-utterance speaker: genuine score = NaN
                continue

            centroid_s = centroids[sidx]
            for r in rows:
                emb_i = embeddings[r]
                # Skip zero embeddings
                if np.linalg.norm(emb_i) < 1e-6:
                    continue
                # Leave-one-out centroid
                centroid_excl = (centroid_s * n_s - emb_i) / (n_s - 1)
                centroid_excl = l2_normalize(centroid_excl)
                genuine_raw[r] = float(np.dot(emb_i, centroid_excl))

        valid_genuine = genuine_raw[~np.isnan(genuine_raw)]
        print(f"    Valid genuine scores: {len(valid_genuine)}/{total}")
        print(f"    NaN (single-utt/zero): {np.isnan(genuine_raw).sum()}")
        if len(valid_genuine) > 0:
            print(f"    Genuine raw: mean={valid_genuine.mean():.4f} "
                  f"std={valid_genuine.std():.4f} "
                  f"min={valid_genuine.min():.4f} max={valid_genuine.max():.4f}")

        # ---- Phase 3: S-norm genuine scores ----
        print("\n  Phase 3: Computing s-norm normalized genuine scores...")

        # Precompute centroid-to-cohort scores for each speaker
        # centroids: (n_speakers, dim), cohort_embs: (1000, dim)
        centroid_cohort_scores = centroids @ cohort_embs.T  # (n_speakers, 1000)
        centroid_cohort_mu = centroid_cohort_scores.mean(axis=1)  # (n_speakers,)
        centroid_cohort_sigma = centroid_cohort_scores.std(axis=1)  # (n_speakers,)
        # Avoid division by zero
        centroid_cohort_sigma = np.maximum(centroid_cohort_sigma, 1e-8)

        # Compute sample-to-cohort scores in chunks
        genuine_norm = np.full(total, np.nan, dtype=np.float32)

        n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_i in range(n_chunks):
            start = chunk_i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total)
            chunk_embs = np.array(embeddings[start:end])  # (chunk, dim)
            chunk_cohort = chunk_embs @ cohort_embs.T  # (chunk, 1000)
            chunk_mu_t = chunk_cohort.mean(axis=1)
            chunk_sigma_t = np.maximum(chunk_cohort.std(axis=1), 1e-8)

            for j in range(end - start):
                global_idx = start + j
                raw = genuine_raw[global_idx]
                if np.isnan(raw):
                    continue
                spk = speaker_ids[global_idx]
                sidx = speaker_to_idx[spk]
                mu_t = chunk_mu_t[j]
                sigma_t = chunk_sigma_t[j]
                mu_e = centroid_cohort_mu[sidx]
                sigma_e = centroid_cohort_sigma[sidx]
                genuine_norm[global_idx] = 0.5 * (
                    (raw - mu_t) / sigma_t + (raw - mu_e) / sigma_e
                )

            if (chunk_i + 1) % 10 == 0 or chunk_i == n_chunks - 1:
                print(f"    Chunk {chunk_i + 1}/{n_chunks}")

        valid_norm = genuine_norm[~np.isnan(genuine_norm)]
        print(f"    Valid s-norm genuine scores: {len(valid_norm)}")
        if len(valid_norm) > 0:
            print(f"    Genuine norm: mean={valid_norm.mean():.4f} "
                  f"std={valid_norm.std():.4f} "
                  f"min={valid_norm.min():.4f} max={valid_norm.max():.4f}")

        # ---- Phase 4: Impostor distribution ----
        print("\n  Phase 4: Computing impostor scores...")

        # Full centroid cross-comparison
        sim_matrix = centroids @ centroids.T  # (n_speakers, n_speakers)
        # Upper triangle excluding diagonal
        triu_rows, triu_cols = np.triu_indices(n_speakers, k=1)
        impostor_raw = sim_matrix[triu_rows, triu_cols].astype(np.float32)
        n_impostor = len(impostor_raw)
        print(f"    Impostor pairs: {n_impostor:,}")
        print(f"    Impostor raw: mean={impostor_raw.mean():.4f} "
              f"std={impostor_raw.std():.4f}")

        # S-norm for impostor scores
        print("    Applying s-norm to impostor scores...")
        impostor_norm = np.zeros(n_impostor, dtype=np.float32)
        for k in range(n_impostor):
            i_spk = triu_rows[k]
            j_spk = triu_cols[k]
            raw = impostor_raw[k]
            # Use centroid-to-cohort stats for both sides
            mu_t = centroid_cohort_mu[i_spk]
            sigma_t = centroid_cohort_sigma[i_spk]
            mu_e = centroid_cohort_mu[j_spk]
            sigma_e = centroid_cohort_sigma[j_spk]
            impostor_norm[k] = 0.5 * (
                (raw - mu_t) / sigma_t + (raw - mu_e) / sigma_e
            )

        print(f"    Impostor norm: mean={impostor_norm.mean():.4f} "
              f"std={impostor_norm.std():.4f}")

        # ---- Phase 5: Save outputs ----
        print("\n  Phase 5: Saving outputs...")

        # Score CSV
        csv_path = os.path.join(OUTPUT_DIR, f"scores_{args.split}_{pn}_{short}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_idx", "filename", "speaker_id", "genuine_raw", "genuine_norm"])
            for i in range(total):
                g_raw = "" if np.isnan(genuine_raw[i]) else f"{genuine_raw[i]:.6f}"
                g_norm = "" if np.isnan(genuine_norm[i]) else f"{genuine_norm[i]:.6f}"
                writer.writerow([i, filenames[i], speaker_ids[i], g_raw, g_norm])
        print(f"    {csv_path} ({total} rows)")

        # Impostor arrays
        imp_raw_path = os.path.join(OUTPUT_DIR, f"impostor_raw_{args.split}_{pn}.npy")
        imp_norm_path = os.path.join(OUTPUT_DIR, f"impostor_norm_{args.split}_{pn}.npy")
        np.save(imp_raw_path, impostor_raw)
        np.save(imp_norm_path, impostor_norm)
        print(f"    {imp_raw_path} ({n_impostor:,} values)")
        print(f"    {imp_norm_path} ({n_impostor:,} values)")

        # Collect statistics
        stats = {
            "genuine_raw": {
                "count": int(len(valid_genuine)),
                "nan_count": int(np.isnan(genuine_raw).sum()),
                "mean": float(valid_genuine.mean()) if len(valid_genuine) > 0 else None,
                "std": float(valid_genuine.std()) if len(valid_genuine) > 0 else None,
                "min": float(valid_genuine.min()) if len(valid_genuine) > 0 else None,
                "max": float(valid_genuine.max()) if len(valid_genuine) > 0 else None,
            },
            "genuine_norm": {
                "count": int(len(valid_norm)),
                "mean": float(valid_norm.mean()) if len(valid_norm) > 0 else None,
                "std": float(valid_norm.std()) if len(valid_norm) > 0 else None,
                "min": float(valid_norm.min()) if len(valid_norm) > 0 else None,
                "max": float(valid_norm.max()) if len(valid_norm) > 0 else None,
            },
            "impostor_raw": {
                "count": int(n_impostor),
                "mean": float(impostor_raw.mean()),
                "std": float(impostor_raw.std()),
                "min": float(impostor_raw.min()),
                "max": float(impostor_raw.max()),
            },
            "impostor_norm": {
                "count": int(n_impostor),
                "mean": float(impostor_norm.mean()),
                "std": float(impostor_norm.std()),
                "min": float(impostor_norm.min()),
                "max": float(impostor_norm.max()),
            },
        }
        all_stats[pn] = stats

        t_elapsed = time.time() - t0
        print(f"\n  {pn} completed in {time.strftime('%H:%M:%S', time.gmtime(t_elapsed))}")

        # Save checkpoint + intermediate combined stats after each provider
        completed_providers.add(pn)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_providers": sorted(completed_providers)}, f)
        print(f"  Checkpoint saved: {sorted(completed_providers)} done, stats written to {stats_path}")

    # --- Save combined statistics (final) ---
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStatistics saved: {stats_path}")

    # Remove checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  Checkpoint removed (all providers completed successfully)")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("SCORE SUMMARY")
    print("=" * 80)
    print(f"{'Provider':<12} {'Genuine Raw':>14} {'Genuine Norm':>14} "
          f"{'Impostor Raw':>14} {'Impostor Norm':>14}")
    print(f"{'':12} {'mean +/- std':>14} {'mean +/- std':>14} "
          f"{'mean +/- std':>14} {'mean +/- std':>14}")
    print("-" * 80)
    for pn in provider_names:
        s = all_stats[pn]
        gr = s["genuine_raw"]
        gn = s["genuine_norm"]
        ir = s["impostor_raw"]
        inr = s["impostor_norm"]

        gr_str = f"{gr['mean']:.3f}+/-{gr['std']:.3f}" if gr["mean"] is not None else "N/A"
        gn_str = f"{gn['mean']:.3f}+/-{gn['std']:.3f}" if gn["mean"] is not None else "N/A"
        ir_str = f"{ir['mean']:.3f}+/-{ir['std']:.3f}"
        inr_str = f"{inr['mean']:.3f}+/-{inr['std']:.3f}"

        print(f"  {pn:<10} {gr_str:>14} {gn_str:>14} {ir_str:>14} {inr_str:>14}")

    print(f"\n  Genuine > Impostor check:")
    for pn in provider_names:
        s = all_stats[pn]
        gr_mean = s["genuine_raw"]["mean"]
        ir_mean = s["impostor_raw"]["mean"]
        if gr_mean is not None:
            ok = "PASS" if gr_mean > ir_mean else "FAIL"
            print(f"    {pn}: genuine_raw_mean ({gr_mean:.4f}) > impostor_raw_mean ({ir_mean:.4f}) -> {ok}")

    print(f"\n  S-norm impostor ~zero-mean check:")
    for pn in provider_names:
        s = all_stats[pn]
        inr_mean = s["impostor_norm"]["mean"]
        ok = "PASS" if abs(inr_mean) < 0.5 else "WARN"
        print(f"    {pn}: impostor_norm_mean = {inr_mean:.4f} -> {ok}")

    print("\nDone.")


if __name__ == "__main__":
    main()
