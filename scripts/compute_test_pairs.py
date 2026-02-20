"""Step 8: Compute pairwise comparison scores for test datasets.

For each dataset x provider: compute cosine similarity for all defined pairs.
- VoxCeleb1-test: uses official veri_test2.txt pair list
- VCTK / CN-Celeb: generates pairs (50K genuine + 50K impostor, seed=42)

Usage:
    python scripts/compute_test_pairs.py --dataset voxceleb1 --providers P1_ECAPA P2_RESNET P3_ECAPA2 P4_XVECTOR P5_WAVLM
    python scripts/compute_test_pairs.py --dataset vctk --providers P1_ECAPA P2_RESNET P3_ECAPA2 P4_XVECTOR P5_WAVLM
    python scripts/compute_test_pairs.py --dataset cnceleb --providers P1_ECAPA P2_RESNET P3_ECAPA2 P4_XVECTOR P5_WAVLM
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
COHORT_DIR = os.path.join(DATA_DIR, "snorm_cohort")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR = os.path.join(DATA_DIR, "test_scores")

PROVIDER_DIMS = {
    "P1_ECAPA": 192,
    "P2_RESNET": 256,
    "P3_ECAPA2": 192,
    "P4_XVECTOR": 512,
    "P5_WAVLM": 512,
}

DATASET_TO_SPLIT = {
    "voxceleb1": "test_voxceleb1",
    "vctk": "test_vctk",
    "cnceleb": "test_cnceleb",
}

N_PAIRS_GENERATED = 50000  # per class for VCTK/CN-Celeb


def l2_normalize(x):
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-12 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def load_voxceleb1_pairs(pairs_path, filename_to_idx):
    """Load official VoxCeleb1 verification pairs from veri_test2.txt.

    Format: label path1 path2
    where label is 1 (genuine) or 0 (impostor).
    Paths are like id10270/5r0dWxy17C8/00001.wav
    """
    pairs = []
    labels = []
    skipped = 0

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                skipped += 1
                continue

            label = int(parts[0])
            path1 = parts[1]
            path2 = parts[2]

            # Find matching indices - try multiple path patterns
            idx1 = _find_file_idx(path1, filename_to_idx)
            idx2 = _find_file_idx(path2, filename_to_idx)

            if idx1 is None or idx2 is None:
                skipped += 1
                continue

            pairs.append((idx1, idx2))
            labels.append(label)

    logger.info(f"Loaded {len(pairs)} pairs ({skipped} skipped)")
    return np.array(pairs, dtype=int), np.array(labels, dtype=int)


def _find_file_idx(rel_path, filename_to_idx):
    """Find file index by trying several path matching strategies."""
    # Direct match
    if rel_path in filename_to_idx:
        return filename_to_idx[rel_path]

    # Try with wav/ prefix
    for prefix in ["wav/", ""]:
        candidate = prefix + rel_path
        if candidate in filename_to_idx:
            return filename_to_idx[candidate]

    # Try matching by suffix (last 3 components: spk/vid/utt.wav)
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        suffix = "/".join(parts[-3:])
        for fn, idx in filename_to_idx.items():
            if fn.endswith(suffix):
                return idx

    return None


def generate_pairs(speaker_ids, n_genuine=50000, n_impostor=50000, seed=42):
    """Generate random genuine + impostor pairs.

    Args:
        speaker_ids: list of speaker IDs per sample.
        n_genuine: number of genuine pairs to generate.
        n_impostor: number of impostor pairs to generate.
        seed: random seed.

    Returns:
        (pairs, labels) arrays.
    """
    rng = np.random.RandomState(seed)

    # Build speaker -> indices mapping
    speaker_to_indices = {}
    for i, spk in enumerate(speaker_ids):
        if spk not in speaker_to_indices:
            speaker_to_indices[spk] = []
        speaker_to_indices[spk].append(i)

    # Filter speakers with >= 2 utterances for genuine pairs
    multi_utt_speakers = [spk for spk, idxs in speaker_to_indices.items() if len(idxs) >= 2]
    all_speakers = list(speaker_to_indices.keys())

    # Generate genuine pairs
    genuine_pairs = []
    attempts = 0
    max_attempts = n_genuine * 20
    while len(genuine_pairs) < n_genuine and attempts < max_attempts:
        spk = rng.choice(multi_utt_speakers)
        idxs = speaker_to_indices[spk]
        if len(idxs) >= 2:
            i, j = rng.choice(len(idxs), size=2, replace=False)
            genuine_pairs.append((idxs[i], idxs[j]))
        attempts += 1

    # Generate impostor pairs
    impostor_pairs = []
    attempts = 0
    while len(impostor_pairs) < n_impostor and attempts < max_attempts:
        spk1, spk2 = rng.choice(all_speakers, size=2, replace=False)
        idx1 = rng.choice(speaker_to_indices[spk1])
        idx2 = rng.choice(speaker_to_indices[spk2])
        impostor_pairs.append((idx1, idx2))
        attempts += 1

    pairs = np.array(genuine_pairs + impostor_pairs, dtype=int)
    labels = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs), dtype=int)

    logger.info(f"Generated {len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor pairs")
    return pairs, labels


def compute_pair_scores(embeddings, pairs, cohort_embs=None):
    """Compute cosine similarity for pairs, optionally with S-norm.

    Args:
        embeddings: (N, dim) embedding matrix.
        pairs: (M, 2) index pairs.
        cohort_embs: (C, dim) S-norm cohort (optional).

    Returns:
        cos_sim: (M,) raw cosine similarity.
        cos_sim_snorm: (M,) S-norm cosine similarity (None if no cohort).
    """
    emb1 = l2_normalize(np.array(embeddings[pairs[:, 0]]))
    emb2 = l2_normalize(np.array(embeddings[pairs[:, 1]]))

    # Raw cosine similarity (batch dot product)
    cos_sim = np.sum(emb1 * emb2, axis=1).astype(np.float32)

    cos_sim_snorm = None
    if cohort_embs is not None:
        cohort_norm = l2_normalize(cohort_embs)

        # Compute S-norm: for each pair element, compute mean/std vs cohort
        # Process in chunks to limit memory
        chunk_size = 5000
        cos_sim_snorm = np.zeros(len(pairs), dtype=np.float32)

        for chunk_start in range(0, len(pairs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(pairs))
            chunk_emb1 = emb1[chunk_start:chunk_end]
            chunk_emb2 = emb2[chunk_start:chunk_end]
            chunk_raw = cos_sim[chunk_start:chunk_end]

            # Scores vs cohort
            scores1 = chunk_emb1 @ cohort_norm.T  # (chunk, C)
            scores2 = chunk_emb2 @ cohort_norm.T  # (chunk, C)

            mu1 = scores1.mean(axis=1)
            sigma1 = np.maximum(scores1.std(axis=1), 1e-8)
            mu2 = scores2.mean(axis=1)
            sigma2 = np.maximum(scores2.std(axis=1), 1e-8)

            cos_sim_snorm[chunk_start:chunk_end] = 0.5 * (
                (chunk_raw - mu1) / sigma1 + (chunk_raw - mu2) / sigma2
            )

    return cos_sim, cos_sim_snorm


def main():
    parser = argparse.ArgumentParser(description="Compute test pair scores")
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "vctk", "cnceleb"])
    parser.add_argument("--providers", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    split_name = DATASET_TO_SPLIT[args.dataset]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load index
    index_path = os.path.join(EMBEDDINGS_DIR, f"{split_name}_index.csv")
    if not os.path.exists(index_path):
        logger.error(f"Index not found: {index_path}")
        logger.error("Run extract_embeddings.py first.")
        sys.exit(1)

    filenames = []
    speaker_ids = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row["filename"])
            speaker_ids.append(row["speaker_id"])

    total = len(filenames)
    logger.info(f"Dataset '{args.dataset}' ({split_name}): {total} files")

    # Build filename -> index mapping
    filename_to_idx = {}
    for i, fn in enumerate(filenames):
        filename_to_idx[fn] = i
        # Also index by relative path components for VoxCeleb matching
        parts = fn.replace("\\", "/").split("/")
        for depth in range(1, min(5, len(parts) + 1)):
            suffix = "/".join(parts[-depth:])
            if suffix not in filename_to_idx:
                filename_to_idx[suffix] = i

    # Get pairs
    if args.dataset == "voxceleb1":
        pairs_path = os.path.join(DATA_DIR, "voxceleb1_test_pairs.txt")
        if not os.path.exists(pairs_path):
            logger.error(f"Pairs file not found: {pairs_path}")
            logger.error("Download veri_test2.txt from VoxCeleb website.")
            sys.exit(1)
        pairs, labels = load_voxceleb1_pairs(pairs_path, filename_to_idx)
    else:
        pairs, labels = generate_pairs(speaker_ids, N_PAIRS_GENERATED, N_PAIRS_GENERATED, seed=42)

    n_genuine = int((labels == 1).sum())
    n_impostor = int((labels == 0).sum())
    logger.info(f"Pairs: {len(pairs)} total ({n_genuine} genuine, {n_impostor} impostor)")

    # Save pair definitions
    pairs_def_path = os.path.join(OUTPUT_DIR, f"pair_definitions_{args.dataset}.npz")
    np.savez(pairs_def_path, pairs=pairs, labels=labels,
             filenames=np.array(filenames), speaker_ids=np.array(speaker_ids))
    logger.info(f"Saved pair definitions: {pairs_def_path}")

    # Checkpoint handling
    checkpoint_path = os.path.join(OUTPUT_DIR, f"_checkpoint_pairs_{args.dataset}.json")
    completed_providers = set()
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        completed_providers = set(ckpt.get("completed_providers", []))
        logger.info(f"Resuming: {completed_providers} already done")

    # Load S-norm cohort
    cohort_speakers_path = os.path.join(COHORT_DIR, "cohort_speakers.txt")
    has_cohort = os.path.exists(cohort_speakers_path)
    if has_cohort:
        logger.info("S-norm cohort available")

    # Process each provider
    for pn in args.providers:
        if pn in completed_providers:
            logger.info(f"Skipping {pn} (already completed)")
            continue

        if pn not in PROVIDER_DIMS:
            logger.error(f"Unknown provider: {pn}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROVIDER: {pn}")
        t0 = time.time()

        dim = PROVIDER_DIMS[pn]
        memmap_path = os.path.join(EMBEDDINGS_DIR, f"{split_name}_{pn}.npy")

        if not os.path.exists(memmap_path):
            logger.error(f"Embeddings not found: {memmap_path}")
            continue

        embeddings = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(total, dim))
        logger.info(f"Loaded embeddings: shape={embeddings.shape}")

        # Load cohort
        cohort_embs = None
        if has_cohort:
            cohort_path = os.path.join(COHORT_DIR, f"cohort_embeddings_{pn}.npy")
            if os.path.exists(cohort_path):
                cohort_embs = np.load(cohort_path)
                logger.info(f"Loaded cohort: shape={cohort_embs.shape}")

        # Compute scores
        cos_sim, cos_sim_snorm = compute_pair_scores(embeddings, pairs, cohort_embs)

        # Save results
        out_path = os.path.join(OUTPUT_DIR, f"pair_scores_{args.dataset}_{pn}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["pair_idx", "file1", "file2", "speaker1", "speaker2",
                       "is_genuine", "cos_sim"]
            if cos_sim_snorm is not None:
                header.append("cos_sim_snorm")
            writer.writerow(header)

            for j in range(len(pairs)):
                i1, i2 = pairs[j]
                row = [j, filenames[i1], filenames[i2],
                       speaker_ids[i1], speaker_ids[i2],
                       int(labels[j]), f"{cos_sim[j]:.6f}"]
                if cos_sim_snorm is not None:
                    row.append(f"{cos_sim_snorm[j]:.6f}")
                writer.writerow(row)

        elapsed = time.time() - t0
        logger.info(f"Saved: {out_path} ({len(pairs)} pairs)")
        logger.info(f"  cos_sim: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
        if cos_sim_snorm is not None:
            logger.info(f"  cos_sim_snorm: mean={cos_sim_snorm.mean():.4f}, std={cos_sim_snorm.std():.4f}")
        logger.info(f"  Genuine cos_sim: mean={cos_sim[labels == 1].mean():.4f}")
        logger.info(f"  Impostor cos_sim: mean={cos_sim[labels == 0].mean():.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        # Update checkpoint
        completed_providers.add(pn)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_providers": sorted(completed_providers)}, f)

    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    logger.info("\nAll providers completed.")


if __name__ == "__main__":
    main()
