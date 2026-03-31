"""Step 1.7: Build s-norm cohort from VoxCeleb2 dev speakers.

Samples 1,000 speakers from VoxCeleb2 dev/wav, extracts embeddings from
all 5 providers, computes L2-normalized mean embedding per speaker per
provider, and saves results to implementation/data/step1/snorm_cohort/.

Checkpoints every 50 speakers so interrupted runs can be resumed with --resume.

Usage:
    python implementation/scripts/build_snorm_cohort.py
    python implementation/scripts/build_snorm_cohort.py --resume
"""

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torchaudio

# Allow imports from implementation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import get_provider, PROVIDERS

VOXCELEB2_DEV_WAV = os.path.join("Datasets", "voxCELEB2", "dev", "wav")
OUTPUT_DIR = os.path.join("implementation", "data", "step1", "snorm_cohort")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "_checkpoint.pkl")

NUM_COHORT_SPEAKERS = 1000
MAX_UTTS_PER_SPEAKER = 30
CHECKPOINT_EVERY = 50
RANDOM_SEED = 42

PROVIDER_NAMES = list(PROVIDERS.keys())  # P1-P5
PROVIDER_DIMS = {
    "P1_ECAPA": 192,
    "P2_RESNET": 256,
    "P3_ECAPA2": 192,
    "P4_XVECTOR": 512,
    "P5_WAVLM": 512,
}


def collect_speaker_wavs(speaker_dir: str) -> list[str]:
    """Collect all WAV files under a speaker directory."""
    wavs = []
    for root, _, files in os.walk(speaker_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return sorted(wavs)


def load_providers(device: str) -> dict:
    """Load all 5 providers onto the specified device."""
    providers = {}
    for name in PROVIDER_NAMES:
        print(f"  Loading {name}...", end=" ", flush=True)
        t0 = time.time()
        p = get_provider(name, device=device)
        p.load_model()
        print(f"done ({time.time() - t0:.1f}s)")
        providers[name] = p
    return providers


def extract_mean_embedding(
    provider, wav_files: list[str]
) -> np.ndarray:
    """Extract embeddings for all files and return L2-normalized mean."""
    embeddings = []
    for wav_path in wav_files:
        try:
            waveform, sr = torchaudio.load(wav_path)
            emb = provider.extract_embedding(waveform, sr)
            if not np.any(np.isnan(emb)) and not np.any(np.isinf(emb)):
                embeddings.append(emb)
        except Exception as e:
            print(f"    WARN: {provider.name} failed on {wav_path}: {e}")
    if len(embeddings) == 0:
        return None
    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm < 1e-12:
        return mean_emb
    return mean_emb / norm


def _save_partial_outputs(
    completed_embeddings: dict, completed_speaker_ids: list, n_done: int
):
    """Save partial .npy outputs and speaker list at checkpoint time.

    Files are written with a _partial suffix so they can be inspected
    without waiting for the full run to finish.
    """
    # Partial speaker list
    partial_speakers_path = os.path.join(OUTPUT_DIR, "cohort_speakers_partial.txt")
    with open(partial_speakers_path, "w", encoding="utf-8") as f:
        for spk_id in completed_speaker_ids:
            f.write(spk_id + "\n")

    # Partial per-provider embedding matrices
    for pname in PROVIDER_NAMES:
        dim = PROVIDER_DIMS[pname]
        matrix = np.zeros((n_done, dim), dtype=np.float32)
        for i, spk_id in enumerate(completed_speaker_ids):
            if spk_id in completed_embeddings and pname in completed_embeddings[spk_id]:
                matrix[i] = completed_embeddings[spk_id][pname]
        out_path = os.path.join(OUTPUT_DIR, f"cohort_embeddings_{pname}_partial.npy")
        np.save(out_path, matrix)


def main():
    parser = argparse.ArgumentParser(description="Build s-norm cohort")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Select 1,000 speakers ---
    print("Listing VoxCeleb2 dev speakers...")
    all_speakers = sorted(os.listdir(VOXCELEB2_DEV_WAV))
    all_speakers = [s for s in all_speakers if os.path.isdir(os.path.join(VOXCELEB2_DEV_WAV, s))]
    print(f"  Found {len(all_speakers)} speakers")

    random.seed(RANDOM_SEED)
    cohort_speakers = sorted(random.sample(all_speakers, NUM_COHORT_SPEAKERS))
    print(f"  Sampled {NUM_COHORT_SPEAKERS} cohort speakers (seed={RANDOM_SEED})")

    # --- Step 2: Load checkpoint or start fresh ---
    completed_embeddings = {}  # speaker_id -> {provider_name: embedding}
    start_idx = 0

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        completed_embeddings = ckpt["embeddings"]
        start_idx = ckpt["next_idx"]
        print(f"  Resumed from checkpoint: {start_idx}/{NUM_COHORT_SPEAKERS} speakers done")
    else:
        print("  Starting fresh (no checkpoint)")

    # --- Step 3: Load all 5 providers ---
    print(f"\nLoading all 5 providers on {args.device}...")
    providers = load_providers(args.device)

    # --- Step 4: Process speakers ---
    print(f"\nProcessing {NUM_COHORT_SPEAKERS - start_idx} remaining speakers...\n")
    t_start = time.time()

    for idx in range(start_idx, NUM_COHORT_SPEAKERS):
        spk_id = cohort_speakers[idx]
        spk_dir = os.path.join(VOXCELEB2_DEV_WAV, spk_id)
        wav_files = collect_speaker_wavs(spk_dir)

        # Cap utterances
        if len(wav_files) > MAX_UTTS_PER_SPEAKER:
            random.seed(RANDOM_SEED + hash(spk_id))
            wav_files = sorted(random.sample(wav_files, MAX_UTTS_PER_SPEAKER))

        n_utts = len(wav_files)
        elapsed = time.time() - t_start
        done = idx - start_idx
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        remaining = NUM_COHORT_SPEAKERS - idx
        eta_s = remaining / rate if rate > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))

        print(f"  [{idx + 1}/{NUM_COHORT_SPEAKERS}] {spk_id} ({n_utts} utts) "
              f"| elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
              f"| eta {eta_str}")

        spk_embeddings = {}
        for pname, provider in providers.items():
            emb = extract_mean_embedding(provider, wav_files)
            if emb is not None:
                spk_embeddings[pname] = emb
            else:
                print(f"    WARN: No valid embeddings for {pname} on {spk_id}")

        completed_embeddings[spk_id] = spk_embeddings

        # Checkpoint every N speakers
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt = {"embeddings": completed_embeddings, "next_idx": idx + 1}
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump(ckpt, f)

            # Save partial outputs at checkpoint (so results are always on disk)
            n_done = idx + 1
            _save_partial_outputs(
                completed_embeddings, cohort_speakers[:n_done], n_done
            )

            # Verify checkpoint data
            n_with_all = sum(
                1 for s in completed_embeddings.values()
                if len(s) == len(PROVIDER_NAMES)
            )
            nan_count = sum(
                1 for s in completed_embeddings.values()
                for emb in s.values()
                if np.any(np.isnan(emb))
            )
            sample_spk = list(completed_embeddings.keys())[-1]
            sample_norms = {
                pn: f"{np.linalg.norm(emb):.4f}"
                for pn, emb in completed_embeddings[sample_spk].items()
            }
            print(f"    -> Checkpoint saved at {n_done} speakers | "
                  f"all-5-providers: {n_with_all}/{n_done} | "
                  f"NaN embeddings: {nan_count} | "
                  f"last speaker norms: {sample_norms}")

    # --- Step 5: Save outputs ---
    print("\nSaving outputs...")

    # cohort_speakers.txt
    speakers_path = os.path.join(OUTPUT_DIR, "cohort_speakers.txt")
    with open(speakers_path, "w", encoding="utf-8") as f:
        for spk_id in cohort_speakers:
            f.write(spk_id + "\n")
    print(f"  Saved {speakers_path} ({len(cohort_speakers)} speakers)")

    # Per-provider embedding matrices
    for pname in PROVIDER_NAMES:
        dim = PROVIDER_DIMS[pname]
        matrix = np.zeros((NUM_COHORT_SPEAKERS, dim), dtype=np.float32)
        missing = 0
        for i, spk_id in enumerate(cohort_speakers):
            if spk_id in completed_embeddings and pname in completed_embeddings[spk_id]:
                matrix[i] = completed_embeddings[spk_id][pname]
            else:
                missing += 1
        out_path = os.path.join(OUTPUT_DIR, f"cohort_embeddings_{pname}.npy")
        np.save(out_path, matrix)
        print(f"  Saved {out_path} shape={matrix.shape} missing={missing}")

    # Remove checkpoint and partial files on successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  Checkpoint removed (completed successfully)")
    # Clean up partial outputs (full outputs now exist)
    for cleanup_name in ["cohort_speakers_partial.txt"] + [
        f"cohort_embeddings_{pn}_partial.npy" for pn in PROVIDER_NAMES
    ]:
        cleanup_path = os.path.join(OUTPUT_DIR, cleanup_name)
        if os.path.exists(cleanup_path):
            os.remove(cleanup_path)
    print("  Partial output files removed")

    # --- Step 6: Verification summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for pname in PROVIDER_NAMES:
        dim = PROVIDER_DIMS[pname]
        npy_path = os.path.join(OUTPUT_DIR, f"cohort_embeddings_{pname}.npy")
        mat = np.load(npy_path)
        norms = np.linalg.norm(mat, axis=1)
        pairwise_cos = mat @ mat.T
        # Upper triangle excluding diagonal
        mask = np.triu(np.ones(pairwise_cos.shape, dtype=bool), k=1)
        cos_vals = pairwise_cos[mask]

        has_nan = np.any(np.isnan(mat))
        has_inf = np.any(np.isinf(mat))
        norm_ok = np.allclose(norms, 1.0, atol=1e-4)

        print(f"\n  {pname} ({dim}-dim):")
        print(f"    Shape: {mat.shape}")
        print(f"    NaN: {has_nan} | Inf: {has_inf}")
        print(f"    Norms: min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f} (all ~1.0: {norm_ok})")
        print(f"    Pairwise cosine: mean={cos_vals.mean():.4f} std={cos_vals.std():.4f}")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
    print("Done.")


if __name__ == "__main__":
    main()
