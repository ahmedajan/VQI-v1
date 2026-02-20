"""Step 1.6a: Extract speaker embeddings (optimized version).

Drop-in replacement for extract_embeddings.py with concurrent provider
execution and multi-threaded I/O prefetching. Resumes from existing
checkpoint and writes to the same memmap files.

Key improvements over the original:
- Concurrent provider execution: P1, P2, P3 run in parallel threads
  (CUDA releases the GIL, so GPU work from different threads overlaps)
- Multi-threaded audio prefetching (4 workers, deep queue)
- All audio pre-loaded into memory before GPU processing
- Expected speedup: 2-3x from provider concurrency + I/O overlap

Usage:
    python implementation/scripts/extract_embeddings_batched.py --resume
"""

import argparse
import concurrent.futures
import csv
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import get_provider, TRAIN_PROVIDERS, PROVIDERS

SPLITS_DIR = os.path.join("implementation", "data", "splits")
OUTPUT_DIR = os.path.join("implementation", "data", "embeddings")

CHECKPOINT_EVERY = 10000

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


class MultiThreadPrefetcher:
    """Multi-threaded audio loader with deep queue."""

    def __init__(self, file_list, start_idx, num_workers=4, queue_depth=128):
        self._file_list = file_list
        self._total = len(file_list)
        self._queue = queue.Queue(maxsize=queue_depth)
        self._stop_event = threading.Event()
        self._idx_queue = queue.Queue()

        for idx in range(start_idx, self._total):
            self._idx_queue.put(idx)

        self._workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                idx = self._idx_queue.get(timeout=1.0)
            except queue.Empty:
                break
            filepath = self._file_list[idx]
            try:
                waveform, sr = torchaudio.load(filepath)
                # Preprocess immediately in I/O thread (CPU work)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(
                        waveform, orig_freq=sr, new_freq=16000
                    )
                if waveform.dim() == 2:
                    waveform = waveform[0:1, :]  # keep (1, samples) for providers
                self._queue.put((idx, waveform, 16000, None))
            except Exception as e:
                self._queue.put((idx, None, None, str(e)))

    def get(self, timeout=300.0):
        return self._queue.get(timeout=timeout)

    def stop(self):
        self._stop_event.set()


def _run_provider(pn, provider, waveform, device):
    """Run a single provider on a single waveform. Returns (pn, embedding) or (pn, error)."""
    try:
        waveform_dev = waveform.to(device)
        with torch.no_grad():
            if pn == "P3_ECAPA2":
                emb = provider._model(waveform_dev)
            else:
                emb = provider._model.encode_batch(waveform_dev)
        emb = emb.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        return (pn, emb, None)
    except Exception as e:
        return (pn, None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings (optimized)")
    parser.add_argument(
        "--split", default="train_pool",
        help="Split name (without .csv)"
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Provider names (default: P1_ECAPA P2_RESNET P3_ECAPA2)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="I/O prefetch threads")
    parser.add_argument("--queue-depth", type=int, default=128, help="Prefetch queue depth")
    args = parser.parse_args()

    if args.providers:
        provider_names = args.providers
    else:
        provider_names = list(TRAIN_PROVIDERS)
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

    # --- Step 2: Open memmaps ---
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{args.split}_checkpoint.txt")
    error_path = os.path.join(OUTPUT_DIR, f"{args.split}_errors.txt")

    memmaps = {}
    for pn in provider_names:
        dim = PROVIDER_DIMS[pn]
        npy_path = os.path.join(OUTPUT_DIR, f"{args.split}_{PROVIDER_SHORT[pn]}.npy")
        if args.resume and os.path.exists(npy_path):
            mm = np.memmap(npy_path, dtype=np.float32, mode="r+", shape=(total, dim))
            print(f"  Opened existing memmap: {npy_path} shape={mm.shape}")
        else:
            mm = np.memmap(npy_path, dtype=np.float32, mode="w+", shape=(total, dim))
            print(f"  Created memmap: {npy_path} shape={mm.shape}")
        memmaps[pn] = mm

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

    remaining = total - start_idx

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

    # --- Step 5: Extract embeddings with concurrent providers ---
    print(f"\nExtracting embeddings ({start_idx} -> {total})...")
    print(f"  Concurrent providers: {provider_names}")
    print(f"  Prefetch: {args.num_workers} workers, queue depth {args.queue_depth}")
    print(flush=True)

    t_start = time.time()
    error_count = 0
    processed_count = 0

    prefetcher = MultiThreadPrefetcher(
        filenames, start_idx,
        num_workers=args.num_workers,
        queue_depth=args.queue_depth
    )

    # Thread pool for concurrent provider execution
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(provider_names))

    try:
        for i in range(start_idx, total):
            idx, waveform, sr, load_error = prefetcher.get(timeout=300.0)

            if load_error is not None:
                error_count += 1
                with open(error_path, "a", encoding="utf-8") as f:
                    safe_err = load_error.replace(",", ";").replace("\n", " ")
                    f.write(f"{idx},{filenames[idx]},{safe_err}\n")
                for pn in provider_names:
                    memmaps[pn][idx] = 0.0
                if error_count <= 10:
                    print(f"  ERROR [{idx}] {filenames[idx]}: {load_error}")
                elif error_count == 11:
                    print(f"  (suppressing further error messages, see {error_path})")
                processed_count += 1
                continue

            # Submit all providers concurrently
            futures = {
                executor.submit(_run_provider, pn, providers[pn], waveform, args.device): pn
                for pn in provider_names
            }

            for future in concurrent.futures.as_completed(futures):
                pn, emb, err = future.result()
                if err is not None:
                    error_count += 1
                    memmaps[pn][idx] = 0.0
                    with open(error_path, "a", encoding="utf-8") as f:
                        safe_err = err.replace(",", ";").replace("\n", " ")
                        f.write(f"{idx},{filenames[idx]},{pn}: {safe_err}\n")
                else:
                    memmaps[pn][idx] = emb

            processed_count += 1

            # Progress every 1000 files (or every 100 for first 1000)
            report_every = 100 if processed_count <= 1000 else 1000
            if processed_count % report_every == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining_files = total - i - 1
                eta_s = remaining_files / rate if rate > 0 else 0
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                pct = (i + 1) / total * 100
                print(f"  [{i + 1}/{total}] {pct:.1f}% | elapsed {elapsed_str} "
                      f"| eta {eta_str} | {rate:.2f} files/s | errors: {error_count}",
                      flush=True)

            # Checkpoint every N files
            if processed_count % CHECKPOINT_EVERY == 0:
                for mm in memmaps.values():
                    mm.flush()
                with open(checkpoint_path, "w") as f:
                    f.write(str(i + 1))

                check_start = max(0, i + 1 - 100)
                for pn in provider_names:
                    chunk = np.array(memmaps[pn][check_start:i + 1])
                    norms = np.linalg.norm(chunk, axis=1)
                    n_nan = int(np.any(np.isnan(chunk), axis=1).sum())
                    n_zero = int((norms < 1e-6).sum())
                    n_ok = len(norms) - n_nan - n_zero
                    print(f"    Checkpoint verify {pn}: last 100 rows -> "
                          f"ok={n_ok} zero={n_zero} nan={n_nan}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        for mm in memmaps.values():
            mm.flush()
        with open(checkpoint_path, "w") as f:
            f.write(str(i))
        print(f"  Checkpoint saved at index {i}. Use --resume to continue.")
        prefetcher.stop()
        executor.shutdown(wait=False)
        sys.exit(1)

    # Final flush
    for mm in memmaps.values():
        mm.flush()

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    prefetcher.stop()
    executor.shutdown()

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
    print(f"Total errors: {error_count}")
    _print_summary(memmaps, provider_names, total)


def _print_summary(memmaps, provider_names, total):
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
