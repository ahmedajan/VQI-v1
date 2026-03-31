"""Step 1.6a: Extract speaker embeddings for training pool.

Reads a split CSV, extracts embeddings from specified providers, and writes
them to numpy memmap files with checkpoint-resume support.

Default: train_pool.csv with P1-P3 (training providers).
Can also extract for val/test splits with different provider subsets.

Features:
  - Memmap output (disk-backed, no RAM limit)
  - Checkpoint every N batches (flush + counter file)
  - --resume to continue from checkpoint
  - Batched GPU inference for all providers
  - Parallel audio I/O with ThreadPoolExecutor
  - Audio truncation (center crop) to limit GPU memory
  - Error logging to embedding_errors.txt
  - --sequential-providers: process one provider at a time (saves VRAM)

Usage:
    python implementation/scripts/extract_embeddings.py
    python implementation/scripts/extract_embeddings.py --resume
    python implementation/scripts/extract_embeddings.py --split val_set --providers P4_XVECTOR P5_WAVLM
    python implementation/scripts/extract_embeddings.py --split test_cnceleb --providers P1_ECAPA P2_RESNET P3_ECAPA2 P4_XVECTOR P5_WAVLM --batch-size 16 --max-seconds 20
    python implementation/scripts/extract_embeddings.py --split x1_expansion_pool --sequential-providers --resume
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import get_provider, TRAIN_PROVIDERS, PROVIDERS

SPLITS_DIR = os.path.join("implementation", "data", "step1", "splits")
OUTPUT_DIR = os.path.join("implementation", "data", "step1", "embeddings")

CHECKPOINT_EVERY = 5000  # files (rounded to batch boundary)

PROVIDER_DIMS = {
    "P1_ECAPA": 192,
    "P2_RESNET": 256,
    "P3_ECAPA2": 192,
    "P4_XVECTOR": 512,
    "P5_WAVLM": 512,
}

PROVIDER_SHORT = {
    "P1_ECAPA": "P1_ECAPA",
    "P2_RESNET": "P2_RESNET",
    "P3_ECAPA2": "P3_ECAPA2",
    "P4_XVECTOR": "P4_XVECTOR",
    "P5_WAVLM": "P5_WAVLM",
}


def load_and_preprocess(filepath, max_samples):
    """Load audio, resample to 16kHz, mono, center-crop to max_samples."""
    try:
        waveform, sr = torchaudio.load(filepath)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.dim() == 2:
            waveform = waveform[0]  # mono: (samples,)
        if max_samples > 0 and waveform.shape[0] > max_samples:
            start = (waveform.shape[0] - max_samples) // 2
            waveform = waveform[start:start + max_samples]
        return waveform, None
    except Exception as e:
        return None, str(e)


def memmap_filename(split_name: str, provider_name: str) -> str:
    short = PROVIDER_SHORT[provider_name]
    return f"{split_name}_{short}.npy"


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings (batched)")
    parser.add_argument(
        "--split", default="train_pool",
        help="Split name (without .csv), e.g. train_pool, val_set, test_voxceleb1"
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Provider names (default: P1_ECAPA P2_RESNET P3_ECAPA2 for train splits)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for GPU inference (default: 16)")
    parser.add_argument("--max-seconds", type=float, default=20.0,
                        help="Max audio duration in seconds, 0=no limit (default: 20)")
    parser.add_argument("--io-workers", type=int, default=4,
                        help="Number of I/O threads for audio loading (default: 4)")
    parser.add_argument("--sequential-providers", action="store_true",
                        help="Process one provider at a time to reduce VRAM usage")
    args = parser.parse_args()

    max_samples = int(args.max_seconds * 16000) if args.max_seconds > 0 else 0

    # Determine providers
    if args.providers:
        provider_names = args.providers
    else:
        provider_names = list(TRAIN_PROVIDERS)  # P1, P2, P3
    for pn in provider_names:
        if pn not in PROVIDERS:
            print(f"ERROR: Unknown provider '{pn}'. Choose from: {list(PROVIDERS)}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Read split CSV ---
    split_csv = os.path.join(SPLITS_DIR, f"{args.split}.csv")
    print(f"Reading {split_csv}...")
    filenames = []
    speaker_ids = []
    dataset_sources = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row["filename"])
            speaker_ids.append(row["speaker_id"])
            dataset_sources.append(row["dataset_source"])
    total = len(filenames)
    print(f"  {total} files to process")
    print(f"  Batch size: {args.batch_size}, Max seconds: {args.max_seconds}, "
          f"I/O workers: {args.io_workers}")

    # --- Step 2: Create index CSV + allocate memmaps ---
    index_path = os.path.join(OUTPUT_DIR, f"{args.split}_index.csv")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{args.split}_checkpoint.txt")
    error_path = os.path.join(OUTPUT_DIR, f"{args.split}_errors.txt")

    with open(index_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "filename", "speaker_id", "dataset_source"])
        for i in range(total):
            writer.writerow([i, filenames[i], speaker_ids[i], dataset_sources[i]])
    print(f"  Index: {index_path}")

    # Create or open memmaps
    memmaps = {}
    for pn in provider_names:
        dim = PROVIDER_DIMS[pn]
        npy_path = os.path.join(OUTPUT_DIR, memmap_filename(args.split, pn))
        if args.resume and os.path.exists(npy_path):
            mm = np.memmap(npy_path, dtype=np.float32, mode="r+", shape=(total, dim))
            print(f"  Opened existing memmap: {npy_path} shape={mm.shape}")
        else:
            mm = np.memmap(npy_path, dtype=np.float32, mode="w+", shape=(total, dim))
            print(f"  Created memmap: {npy_path} shape={mm.shape}")
        memmaps[pn] = mm

    # --- Sequential-providers mode ---
    if args.sequential_providers:
        _run_sequential(args, provider_names, filenames, memmaps, total,
                        checkpoint_path, error_path, max_samples)
        return

    # --- Step 3: Resume from checkpoint ---
    start_idx = 0
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            start_idx = int(f.read().strip())
        print(f"  Resuming from file index {start_idx} ({start_idx}/{total} done)")
    else:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("row_idx,filename,error\n")

    if start_idx >= total:
        print("  All files already processed!")
        _print_summary(memmaps, provider_names, total)
        return

    # --- Step 4: Load providers ---
    print(f"\nLoading {len(provider_names)} providers on {args.device}...")
    providers = {}
    for pn in provider_names:
        print(f"  Loading {pn}...", end=" ", flush=True)
        t0 = time.time()
        p = get_provider(pn, device=args.device)
        p.load_model()
        print(f"done ({time.time() - t0:.1f}s)")
        providers[pn] = p

    # --- Step 5: Batched extraction ---
    batch_size = args.batch_size
    print(f"\nExtracting embeddings ({start_idx} -> {total}), "
          f"batch_size={batch_size}...\n")
    t_start = time.time()
    error_count = 0
    files_processed = 0
    provider_times = {pn: 0.0 for pn in provider_names}

    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)

    try:
        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = list(range(batch_start, batch_end))

            # --- Parallel audio loading ---
            futures = {}
            for idx in batch_indices:
                fut = io_pool.submit(load_and_preprocess, filenames[idx], max_samples)
                futures[fut] = idx

            batch_waveforms = []
            for fut in as_completed(futures):
                idx = futures[fut]
                waveform, err = fut.result()
                if err is not None:
                    error_count += 1
                    with open(error_path, "a", encoding="utf-8") as f:
                        safe_err = err.replace(",", ";").replace("\n", " ")
                        f.write(f"{idx},{filenames[idx]},{safe_err}\n")
                    for pn in provider_names:
                        memmaps[pn][idx] = 0.0
                else:
                    batch_waveforms.append((idx, waveform))

            # Sort by index to maintain deterministic order
            batch_waveforms.sort(key=lambda x: x[0])
            if not batch_waveforms:
                files_processed += len(batch_indices)
                continue

            valid_indices = [item[0] for item in batch_waveforms]
            waveforms = [item[1] for item in batch_waveforms]

            # --- GPU batch inference per provider ---
            for pn, provider in providers.items():
                t_prov = time.time()
                try:
                    embs = provider.extract_embedding_batch(waveforms, 16000)
                    for j, idx in enumerate(valid_indices):
                        memmaps[pn][idx] = embs[j]
                except Exception as e:
                    # Fallback to individual extraction
                    for j, idx in enumerate(valid_indices):
                        try:
                            emb = provider.extract_embedding(
                                waveforms[j].unsqueeze(0), 16000
                            )
                            memmaps[pn][idx] = emb
                        except Exception as e2:
                            error_count += 1
                            memmaps[pn][idx] = 0.0
                            with open(error_path, "a", encoding="utf-8") as f:
                                safe_err = str(e2).replace(",", ";").replace("\n", " ")
                                f.write(f"{idx},{filenames[idx]},{pn}: {safe_err}\n")
                provider_times[pn] += time.time() - t_prov

            files_processed += len(batch_indices)

            # Progress report: first 3 batches, then every 500 files
            show_progress = (files_processed <= batch_size * 3 or
                             files_processed % 500 < batch_size or
                             batch_end == total)
            if show_progress:
                elapsed = time.time() - t_start
                rate = files_processed / elapsed if elapsed > 0 else 0
                remaining = total - batch_end
                eta_s = remaining / rate if rate > 0 else 0
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                pct = batch_end / total * 100
                prov_str = " | ".join(
                    f"{pn.split('_')[0]}:{provider_times[pn]:.0f}s"
                    for pn in provider_names
                )
                print(f"  [{batch_end}/{total}] {pct:.1f}% | elapsed {elapsed_str} "
                      f"| eta {eta_str} | {rate:.1f} files/s | err:{error_count} "
                      f"| {prov_str}")

            # Checkpoint every CHECKPOINT_EVERY files
            if files_processed % CHECKPOINT_EVERY < batch_size:
                for mm in memmaps.values():
                    mm.flush()
                with open(checkpoint_path, "w") as f:
                    f.write(str(batch_end))

                # Spot-check last 100 rows
                check_start = max(0, batch_end - 100)
                for pn in provider_names:
                    chunk = np.array(memmaps[pn][check_start:batch_end])
                    norms = np.linalg.norm(chunk, axis=1)
                    n_nan = int(np.any(np.isnan(chunk), axis=1).sum())
                    n_zero = int((norms < 1e-6).sum())
                    n_ok = len(norms) - n_nan - n_zero
                    print(f"    Checkpoint verify {pn}: last 100 -> "
                          f"ok={n_ok} zero={n_zero} nan={n_nan}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        for mm in memmaps.values():
            mm.flush()
        with open(checkpoint_path, "w") as f:
            f.write(str(batch_start))
        print(f"  Checkpoint saved at index {batch_start}. Use --resume to continue.")
        io_pool.shutdown(wait=False)
        sys.exit(1)

    io_pool.shutdown(wait=True)

    # Final flush
    for mm in memmaps.values():
        mm.flush()

    # Remove checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # --- Step 6: Summary ---
    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
    print(f"Total errors: {error_count}")
    _print_summary(memmaps, provider_names, total)


def _run_sequential(args, provider_names, filenames, memmaps, total,
                    checkpoint_path, error_path, max_samples):
    """Process providers one at a time to minimize VRAM usage.

    Each provider is loaded, processes all remaining files, then unloaded
    before the next provider starts. Uses per-provider checkpoint files.
    """
    batch_size = args.batch_size
    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)

    # Determine optimal batch sizes per provider (P3 doesn't benefit from large batches)
    provider_batch_sizes = {}
    for pn in provider_names:
        if pn == "P3_ECAPA2":
            provider_batch_sizes[pn] = min(batch_size, 8)  # P3 is constant-time per batch
        elif pn == "P2_RESNET":
            provider_batch_sizes[pn] = min(batch_size, 16)  # P2 VRAM grows fast
        else:
            provider_batch_sizes[pn] = batch_size  # P1 is fast, can use full batch

    # Initialize error file if not resuming
    if not args.resume:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("row_idx,filename,error\n")

    for pn in provider_names:
        pn_checkpoint = checkpoint_path.replace("_checkpoint.txt",
                                                f"_{pn}_checkpoint.txt")
        pn_bs = provider_batch_sizes[pn]

        # Per-provider resume
        start_idx = 0
        if args.resume and os.path.exists(pn_checkpoint):
            with open(pn_checkpoint, "r") as f:
                start_idx = int(f.read().strip())
            if start_idx >= total:
                print(f"\n{pn}: Already complete ({total}/{total}). Skipping.")
                continue
            print(f"\n{pn}: Resuming from {start_idx}/{total}")
        else:
            print(f"\n{pn}: Starting from 0/{total}")

        # Load single provider
        print(f"  Loading {pn}...", end=" ", flush=True)
        t0 = time.time()
        provider = get_provider(pn, device=args.device)
        provider.load_model()
        load_time = time.time() - t0
        print(f"done ({load_time:.1f}s)")

        # Report VRAM after loading
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1e6
            print(f"  VRAM after load: {vram:.0f} MB | batch_size={pn_bs}")

        t_start = time.time()
        files_processed = 0
        error_count = 0
        mm = memmaps[pn]

        try:
            for batch_start in range(start_idx, total, pn_bs):
                batch_end = min(batch_start + pn_bs, total)
                batch_indices = list(range(batch_start, batch_end))

                # Parallel audio loading
                futures = {}
                for idx in batch_indices:
                    fut = io_pool.submit(load_and_preprocess, filenames[idx],
                                         max_samples)
                    futures[fut] = idx

                batch_waveforms = []
                for fut in as_completed(futures):
                    idx = futures[fut]
                    waveform, err = fut.result()
                    if err is not None:
                        error_count += 1
                        with open(error_path, "a", encoding="utf-8") as f:
                            safe_err = err.replace(",", ";").replace("\n", " ")
                            f.write(f"{idx},{filenames[idx]},{safe_err}\n")
                        mm[idx] = 0.0
                    else:
                        batch_waveforms.append((idx, waveform))

                batch_waveforms.sort(key=lambda x: x[0])
                if not batch_waveforms:
                    files_processed += len(batch_indices)
                    continue

                valid_indices = [item[0] for item in batch_waveforms]
                waveforms = [item[1] for item in batch_waveforms]

                # GPU inference
                try:
                    embs = provider.extract_embedding_batch(waveforms, 16000)
                    for j, idx in enumerate(valid_indices):
                        mm[idx] = embs[j]
                except Exception as e:
                    for j, idx in enumerate(valid_indices):
                        try:
                            emb = provider.extract_embedding(
                                waveforms[j].unsqueeze(0), 16000)
                            mm[idx] = emb
                        except Exception as e2:
                            error_count += 1
                            mm[idx] = 0.0
                            with open(error_path, "a", encoding="utf-8") as f:
                                safe_err = str(e2).replace(",", ";").replace("\n", " ")
                                f.write(f"{idx},{filenames[idx]},{pn}: {safe_err}\n")

                files_processed += len(batch_indices)

                # Progress report
                show_progress = (files_processed <= pn_bs * 3 or
                                 files_processed % 500 < pn_bs or
                                 batch_end == total)
                if show_progress:
                    elapsed = time.time() - t_start
                    rate = files_processed / elapsed if elapsed > 0 else 0
                    remaining = total - batch_end
                    eta_s = remaining / rate if rate > 0 else 0
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                    pct = batch_end / total * 100
                    print(f"  {pn} [{batch_end}/{total}] {pct:.1f}% | "
                          f"elapsed {elapsed_str} | eta {eta_str} | "
                          f"{rate:.1f} files/s | err:{error_count}")

                # Checkpoint
                if files_processed % CHECKPOINT_EVERY < pn_bs:
                    mm.flush()
                    with open(pn_checkpoint, "w") as f:
                        f.write(str(batch_end))
                    check_start = max(0, batch_end - 100)
                    chunk = np.array(mm[check_start:batch_end])
                    norms = np.linalg.norm(chunk, axis=1)
                    n_nan = int(np.any(np.isnan(chunk), axis=1).sum())
                    n_zero = int((norms < 1e-6).sum())
                    n_ok = len(norms) - n_nan - n_zero
                    print(f"    Checkpoint {batch_end}: ok={n_ok} zero={n_zero} nan={n_nan}")

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving {pn} checkpoint at {batch_start}...")
            mm.flush()
            with open(pn_checkpoint, "w") as f:
                f.write(str(batch_start))
            io_pool.shutdown(wait=False)
            sys.exit(1)

        # Finalize this provider
        mm.flush()
        with open(pn_checkpoint, "w") as f:
            f.write(str(total))
        elapsed_pn = time.time() - t_start
        rate = files_processed / elapsed_pn if elapsed_pn > 0 else 0
        print(f"  {pn} DONE: {files_processed} files in "
              f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_pn))} "
              f"({rate:.1f} files/s, {error_count} errors)")

        # Unload provider to free VRAM
        del provider
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1e6
            print(f"  VRAM after unload: {vram:.0f} MB")

    io_pool.shutdown(wait=True)
    print("\n--- All providers complete ---")
    _print_summary(memmaps, provider_names, total)


def _print_summary(memmaps: dict, provider_names: list, total: int):
    """Print verification summary."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for pn in provider_names:
        mm = memmaps[pn]
        norms = np.linalg.norm(mm, axis=1)
        nonzero_mask = norms > 1e-6
        nonzero_norms = norms[nonzero_mask]
        zero_count = total - nonzero_mask.sum()
        has_nan = np.any(np.isnan(mm))
        has_inf = np.any(np.isinf(mm))

        print(f"\n  {pn} (dim={mm.shape[1]}):")
        print(f"    Shape: {mm.shape}")
        print(f"    NaN: {has_nan} | Inf: {has_inf}")
        print(f"    Zero rows: {zero_count}")
        if len(nonzero_norms) > 0:
            print(f"    Norms (non-zero): min={nonzero_norms.min():.6f} "
                  f"max={nonzero_norms.max():.6f} mean={nonzero_norms.mean():.6f}")
    print()


if __name__ == "__main__":
    main()
