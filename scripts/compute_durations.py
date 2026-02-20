#!/usr/bin/env python3
"""
Step 2.1 — Compute Per-Sample Prerequisite Indicators (Speech Duration after VAD)

Reads the training pool manifest (1,210,451 samples), loads each audio file,
runs energy-based VAD, and records speech duration for label selection.

Output: implementation/data/labels/train_pool_durations.csv
Columns: row_idx, filename, speaker_id, dataset_source,
         total_duration_sec, speech_duration_sec, speech_ratio,
         n_frames, n_speech_frames

Checkpoint policy:
  - Saves checkpoint every CHECKPOINT_INTERVAL files
  - Saves partial CSV with _partial suffix
  - --resume flag to restart from checkpoint
  - Verifies outputs at each checkpoint (no NaN, valid ranges)
  - Removes checkpoint on successful completion
"""

import argparse
import csv
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ---- logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- constants ----
TRAIN_POOL_CSV = os.path.join("D:", os.sep, "VQI", "implementation", "data", "splits", "train_pool.csv")
OUTPUT_DIR = os.path.join("D:", os.sep, "VQI", "implementation", "data", "labels")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "train_pool_durations.csv")
PARTIAL_CSV = os.path.join(OUTPUT_DIR, "train_pool_durations_partial.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "_checkpoint_durations.pkl")
CHECKPOINT_INTERVAL = 10_000
CSV_COLUMNS = [
    "row_idx", "filename", "speaker_id", "dataset_source",
    "total_duration_sec", "speech_duration_sec", "speech_ratio",
    "n_frames", "n_speech_frames",
]


def process_single_file(args):
    """Process a single audio file: load, compute VAD, return duration info.

    This function runs in a worker process. It imports torchaudio and the
    VAD module inside the function so each worker has its own imports.

    Parameters
    ----------
    args : tuple
        (row_idx, filename, speaker_id, dataset_source)

    Returns
    -------
    dict with CSV_COLUMNS keys, or dict with duration=-1 on error.
    """
    row_idx, filename, speaker_id, dataset_source = args

    try:
        import torchaudio
        # Import VAD - add project root to path if needed
        sys.path.insert(0, os.path.join("D:", os.sep, "VQI", "implementation"))
        from vqi.preprocessing.vad import energy_vad

        # Load audio
        waveform, sr = torchaudio.load(filename)
        waveform_np = waveform.numpy().squeeze()  # to 1-D numpy

        # Resample to 16kHz if needed (safety for non-16kHz files)
        if sr != 16000:
            waveform_torch = torchaudio.functional.resample(waveform, sr, 16000)
            waveform_np = waveform_torch.numpy().squeeze()
            sr = 16000

        total_duration_sec = len(waveform_np) / sr

        # Run VAD
        vad_mask, speech_duration_sec, speech_ratio = energy_vad(
            waveform_np, sample_rate=sr
        )

        n_frames = len(vad_mask)
        n_speech_frames = int(np.sum(vad_mask)) if len(vad_mask) > 0 else 0

        return {
            "row_idx": row_idx,
            "filename": filename,
            "speaker_id": speaker_id,
            "dataset_source": dataset_source,
            "total_duration_sec": round(total_duration_sec, 6),
            "speech_duration_sec": round(speech_duration_sec, 6),
            "speech_ratio": round(speech_ratio, 6),
            "n_frames": n_frames,
            "n_speech_frames": n_speech_frames,
        }

    except Exception as e:
        return {
            "row_idx": row_idx,
            "filename": filename,
            "speaker_id": speaker_id,
            "dataset_source": dataset_source,
            "total_duration_sec": -1.0,
            "speech_duration_sec": -1.0,
            "speech_ratio": -1.0,
            "n_frames": -1,
            "n_speech_frames": -1,
        }


def verify_results(results, label="checkpoint"):
    """Verify a batch of results for data quality."""
    n = len(results)
    if n == 0:
        return True

    errors = []
    n_failed = sum(1 for r in results if r["total_duration_sec"] < 0)
    n_nan = 0
    n_negative_duration = 0
    n_ratio_oob = 0
    n_speech_gt_total = 0

    for r in results:
        if r["total_duration_sec"] < 0:
            continue  # failed files already counted
        if np.isnan(r["speech_duration_sec"]) or np.isnan(r["speech_ratio"]):
            n_nan += 1
        if r["speech_duration_sec"] < 0:
            n_negative_duration += 1
        if r["speech_ratio"] < 0 or r["speech_ratio"] > 1.0001:
            n_ratio_oob += 1
        if r["speech_duration_sec"] > r["total_duration_sec"] + 0.001:
            n_speech_gt_total += 1

    if n_nan > 0:
        errors.append(f"NaN values: {n_nan}")
    if n_negative_duration > 0:
        errors.append(f"Negative durations: {n_negative_duration}")
    if n_ratio_oob > 0:
        errors.append(f"Ratio out of bounds: {n_ratio_oob}")
    if n_speech_gt_total > 0:
        errors.append(f"Speech > total duration: {n_speech_gt_total}")

    logger.info(
        f"[Verify {label}] {n} results: {n_failed} load failures, "
        f"{len(errors)} issues: {errors if errors else 'CLEAN'}"
    )
    return len(errors) == 0


def save_partial_csv(results, path):
    """Save results list as a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def save_checkpoint(results, completed_indices, path):
    """Save checkpoint: results so far + set of completed row indices."""
    data = {
        "results": results,
        "completed_indices": completed_indices,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(results),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    """Load checkpoint if it exists."""
    if not os.path.exists(path):
        return None, set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        f"Loaded checkpoint: {data['count']} results from {data['timestamp']}"
    )
    return data["results"], data["completed_indices"]


def main():
    parser = argparse.ArgumentParser(description="Step 2.1: Compute speech durations after VAD")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Save checkpoint every N files")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Step 2.1: Compute Per-Sample Speech Durations (VAD)")
    logger.info("=" * 70)

    # ---- Load training pool manifest ----
    logger.info(f"Loading training pool manifest: {TRAIN_POOL_CSV}")
    rows = []
    with open(TRAIN_POOL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append((i, row["filename"], row["speaker_id"], row["dataset_source"]))
    total = len(rows)
    logger.info(f"Total samples: {total:,}")

    # ---- Resume from checkpoint if requested ----
    results = []
    completed_indices = set()
    if args.resume:
        results, completed_indices = load_checkpoint(CHECKPOINT_FILE)
        if results:
            logger.info(f"Resuming from {len(results):,} completed results")
        else:
            logger.info("No checkpoint found, starting fresh")

    # Filter out already-completed rows
    remaining = [(idx, fn, spk, ds) for idx, fn, spk, ds in rows if idx not in completed_indices]
    logger.info(f"Remaining to process: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info("All files already processed!")
    else:
        # ---- Process with multiprocessing ----
        n_workers = min(args.workers, os.cpu_count() or 4)
        logger.info(f"Using {n_workers} worker processes")

        batch_results = []
        n_done_since_ckpt = 0
        t_start = time.time()
        n_completed_total = len(results)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_args = {}
            for item in remaining:
                future = executor.submit(process_single_file, item)
                future_to_args[future] = item

            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                completed_indices.add(result["row_idx"])
                batch_results.append(result)
                n_completed_total += 1
                n_done_since_ckpt += 1

                # Progress logging every 1000 files
                if n_completed_total % 1000 == 0:
                    elapsed = time.time() - t_start
                    rate = (n_completed_total - len(results) + len(remaining)) / max(elapsed, 1)
                    # More accurate: files done in this run
                    done_this_run = n_completed_total - (len(results) - len(batch_results))
                    if done_this_run > 0:
                        rate = done_this_run / elapsed
                    remaining_count = total - n_completed_total
                    eta_sec = remaining_count / max(rate, 0.01)
                    eta_h = eta_sec / 3600
                    logger.info(
                        f"Progress: {n_completed_total:,}/{total:,} "
                        f"({100*n_completed_total/total:.1f}%) | "
                        f"Rate: {rate:.1f} files/s | "
                        f"ETA: {eta_h:.1f}h"
                    )

                # Checkpoint
                if n_done_since_ckpt >= args.checkpoint_interval:
                    logger.info(f"--- Checkpoint at {n_completed_total:,} ---")
                    verify_results(batch_results, label=f"batch_{n_completed_total}")
                    save_checkpoint(results, completed_indices, CHECKPOINT_FILE)
                    save_partial_csv(results, PARTIAL_CSV)
                    logger.info(f"Saved checkpoint ({n_completed_total:,} total) and partial CSV")
                    batch_results = []
                    n_done_since_ckpt = 0

        # Final batch verification
        if batch_results:
            verify_results(batch_results, label="final_batch")

    # ---- Sort by row_idx for deterministic output ----
    results.sort(key=lambda r: r["row_idx"])

    # ---- Final verification ----
    logger.info("=" * 50)
    logger.info("Final verification:")
    verify_results(results, label="FINAL")

    n_failed = sum(1 for r in results if r["total_duration_sec"] < 0)
    durations = [r["speech_duration_sec"] for r in results if r["speech_duration_sec"] >= 0]
    ratios = [r["speech_ratio"] for r in results if r["speech_ratio"] >= 0]

    if durations:
        logger.info(f"  Total results: {len(results):,} (expected {total:,})")
        logger.info(f"  Load failures: {n_failed:,}")
        logger.info(f"  Speech duration stats:")
        logger.info(f"    Mean: {np.mean(durations):.3f}s")
        logger.info(f"    Median: {np.median(durations):.3f}s")
        logger.info(f"    P5: {np.percentile(durations, 5):.3f}s")
        logger.info(f"    P95: {np.percentile(durations, 95):.3f}s")
        logger.info(f"    Min: {np.min(durations):.3f}s, Max: {np.max(durations):.3f}s")
        logger.info(f"  Speech ratio stats:")
        logger.info(f"    Mean: {np.mean(ratios):.4f}")
        logger.info(f"    Median: {np.median(ratios):.4f}")

        # Class eligibility preview
        n_class1_eligible = sum(1 for d in durations if d >= 3.0)
        n_class0_forced = sum(1 for d in durations if d < 1.5)
        n_ambiguous = sum(1 for d in durations if 1.5 <= d < 3.0)
        logger.info(f"  Class eligibility preview:")
        logger.info(f"    >= 3.0s (Class 1 eligible): {n_class1_eligible:,} ({100*n_class1_eligible/len(durations):.1f}%)")
        logger.info(f"    < 1.5s (Class 0 forced):    {n_class0_forced:,} ({100*n_class0_forced/len(durations):.1f}%)")
        logger.info(f"    1.5-3.0s (ambiguous zone):   {n_ambiguous:,} ({100*n_ambiguous/len(durations):.1f}%)")

    # ---- Save final output ----
    save_partial_csv(results, OUTPUT_CSV)
    logger.info(f"Saved final output: {OUTPUT_CSV} ({len(results):,} rows)")

    # ---- Clean up checkpoint and partial ----
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Removed checkpoint file")
    if os.path.exists(PARTIAL_CSV):
        os.remove(PARTIAL_CSV)
        logger.info("Removed partial CSV")

    # ---- Verify row count ----
    if len(results) != total:
        logger.warning(f"ROW COUNT MISMATCH: {len(results)} results vs {total} expected!")
    else:
        logger.info(f"Row count verified: {len(results):,} == {total:,}")

    logger.info("=" * 70)
    logger.info("Step 2.1 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
