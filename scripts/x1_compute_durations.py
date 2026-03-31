"""X1.9c: Compute speech durations (VAD) for expansion pool.

Adapted from compute_durations.py (Step 2.1) for the x1_expansion_pool split.
Uses energy-based VAD to measure speech duration per file.

Output: implementation/data/step2/labels/x1_expansion_pool_durations.csv
"""

import argparse
import csv
import logging
import os
import pickle
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "step1", "splits",
                         "x1_expansion_pool.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "step2", "labels")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "x1_expansion_pool_durations.csv")
PARTIAL_CSV = os.path.join(OUTPUT_DIR, "x1_expansion_pool_durations_partial.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "_checkpoint_x1_durations.pkl")
CHECKPOINT_INTERVAL = 50_000

CSV_COLUMNS = [
    "row_idx", "filename", "speaker_id", "dataset_source",
    "total_duration_sec", "speech_duration_sec", "speech_ratio",
    "n_frames", "n_speech_frames",
]


def process_single_file(args):
    """Process a single audio file: load, compute VAD, return duration info."""
    row_idx, filename, speaker_id, dataset_source = args
    try:
        import torchaudio
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from vqi.preprocessing.vad import energy_vad

        waveform, sr = torchaudio.load(filename)
        waveform_np = waveform.numpy().squeeze()
        if sr != 16000:
            waveform_torch = torchaudio.functional.resample(waveform, sr, 16000)
            waveform_np = waveform_torch.numpy().squeeze()
            sr = 16000

        total_duration_sec = len(waveform_np) / sr
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
    except Exception:
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


def save_partial_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def save_checkpoint(results, completed_indices, path):
    data = {
        "results": results,
        "completed_indices": completed_indices,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(results),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None, set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded checkpoint: {data['count']} results from {data['timestamp']}")
    return data["results"], data["completed_indices"]


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="X1.9: Compute speech durations for expansion pool")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("X1.9c: Compute Speech Durations for Expansion Pool")
    logger.info("=" * 70)

    # Load expansion pool manifest
    logger.info(f"Loading manifest: {INPUT_CSV}")
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append((i, row["filename"], row["speaker_id"],
                         row["dataset_source"]))
    total = len(rows)
    logger.info(f"Total samples: {total:,}")

    # Resume
    results = []
    completed_indices = set()
    if args.resume:
        results, completed_indices = load_checkpoint(CHECKPOINT_FILE)
        if results:
            logger.info(f"Resuming from {len(results):,} completed results")
        else:
            results = []
            logger.info("No checkpoint found, starting fresh")

    remaining = [(idx, fn, spk, ds) for idx, fn, spk, ds in rows
                 if idx not in completed_indices]
    logger.info(f"Remaining: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info("All files already processed!")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        n_workers = min(args.workers, os.cpu_count() or 4)
        logger.info(f"Using {n_workers} workers")

        batch_results = []
        n_done_since_ckpt = 0
        t_start = time.time()
        n_completed_total = len(results)

        # Process in chunks to avoid submitting 1.39M futures at once
        SUBMIT_CHUNK = 5000
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for chunk_start in range(0, len(remaining), SUBMIT_CHUNK):
                chunk = remaining[chunk_start:chunk_start + SUBMIT_CHUNK]
                futures = {executor.submit(process_single_file, item): item
                           for item in chunk}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed_indices.add(result["row_idx"])
                    batch_results.append(result)
                    n_completed_total += 1
                    n_done_since_ckpt += 1

                elapsed = time.time() - t_start
                done_this_run = n_completed_total - len(completed_indices) + len(remaining)
                rate = n_done_since_ckpt / max(elapsed, 1)
                remaining_count = total - n_completed_total
                eta_h = (remaining_count / max(rate, 0.01)) / 3600
                logger.info(
                    f"Progress: {n_completed_total:,}/{total:,} "
                    f"({100*n_completed_total/total:.1f}%) | "
                    f"Rate: {rate:.1f} files/s | "
                    f"ETA: {eta_h:.1f}h"
                )

                if n_done_since_ckpt >= CHECKPOINT_INTERVAL:
                    logger.info(f"--- Checkpoint at {n_completed_total:,} ---")
                    save_checkpoint(results, completed_indices, CHECKPOINT_FILE)
                    save_partial_csv(results, PARTIAL_CSV)
                    logger.info(f"Saved checkpoint ({n_completed_total:,} total)")
                    batch_results = []
                    n_done_since_ckpt = 0

    # Sort and save
    results.sort(key=lambda r: r["row_idx"])

    n_failed = sum(1 for r in results if r["total_duration_sec"] < 0)
    durations = [r["speech_duration_sec"] for r in results
                 if r["speech_duration_sec"] >= 0]

    if durations:
        logger.info(f"Total: {len(results):,}, Failures: {n_failed:,}")
        logger.info(f"Speech duration: mean={np.mean(durations):.3f}s, "
                     f"median={np.median(durations):.3f}s")
        logger.info(f"  P5={np.percentile(durations, 5):.3f}s, "
                     f"P95={np.percentile(durations, 95):.3f}s")
        n_ge3 = sum(1 for d in durations if d >= 3.0)
        n_lt15 = sum(1 for d in durations if d < 1.5)
        logger.info(f"Class 1 eligible (>=3.0s): {n_ge3:,} ({100*n_ge3/len(durations):.1f}%)")
        logger.info(f"Class 0 forced (<1.5s): {n_lt15:,} ({100*n_lt15/len(durations):.1f}%)")

    save_partial_csv(results, OUTPUT_CSV)
    logger.info(f"Saved: {OUTPUT_CSV} ({len(results):,} rows)")

    # Clean up
    for f in [CHECKPOINT_FILE, PARTIAL_CSV]:
        if os.path.exists(f):
            os.remove(f)

    if len(results) != total:
        logger.warning(f"ROW MISMATCH: {len(results)} vs {total} expected!")
    else:
        logger.info(f"Row count verified: {len(results):,}")

    logger.info("X1.9c duration computation COMPLETE")


if __name__ == "__main__":
    main()
