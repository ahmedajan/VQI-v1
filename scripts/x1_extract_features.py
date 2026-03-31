"""X1.9e: Extract VQI-S (544) and VQI-V (161) features for expansion pool labeled samples.

Input:  data/step2/labels/x1_expansion_pool_labels.csv (61,328 labeled rows)
Output: data/step4/features/features_s_x1_expansion.npy (61328 x 544)
        data/step4/features/features_v_x1_expansion.npy (61328 x 161)
        data/step4/features/extraction_log_x1_expansion.csv

Uses ProcessPoolExecutor with checkpoint recovery (batch_size=500).
"""

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from vqi.core.feature_orchestrator import N_TOTAL_S
from vqi.core.feature_orchestrator_v import N_TOTAL_V

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "step4", "features")
LABELS_CSV = os.path.join(DATA_DIR, "step2", "labels",
                           "x1_expansion_pool_labels.csv")

OUTPUT_S = os.path.join(FEATURES_DIR, "features_s_x1_expansion.npy")
OUTPUT_V = os.path.join(FEATURES_DIR, "features_v_x1_expansion.npy")
OUTPUT_LOG = os.path.join(FEATURES_DIR, "extraction_log_x1_expansion.csv")
CHECKPOINT = os.path.join(FEATURES_DIR, "_checkpoint_x1_expansion.pkl")
PREFIX = os.path.join(FEATURES_DIR, "x1_expansion")


def extract_one_file(filepath):
    """Extract features for a single audio file."""
    import warnings
    warnings.filterwarnings("ignore")

    impl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)

    t0 = time.time()
    try:
        from vqi.preprocessing.audio_loader import load_audio
        from vqi.preprocessing.normalize import dc_remove_and_normalize
        from vqi.preprocessing.vad import energy_vad
        from vqi.core.feature_orchestrator import compute_all_features, N_TOTAL_S
        from vqi.core.feature_orchestrator_v import compute_all_features_v, N_TOTAL_V

        raw_waveform = load_audio(filepath)
        waveform = dc_remove_and_normalize(raw_waveform)
        vad_mask, _, _ = energy_vad(waveform)

        _, feat_arr_s, intermediates = compute_all_features(
            waveform, 16000, vad_mask, raw_waveform=raw_waveform
        )
        _, feat_arr_v = compute_all_features_v(
            waveform, 16000, vad_mask, intermediates
        )

        elapsed = (time.time() - t0) * 1000
        nan_count = int(np.sum(~np.isfinite(feat_arr_s))) + \
                    int(np.sum(~np.isfinite(feat_arr_v)))
        return feat_arr_s, feat_arr_v, elapsed, "", nan_count
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return (np.full(N_TOTAL_S, np.nan), np.full(N_TOTAL_V, np.nan),
                elapsed, str(e), -1)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="X1.9e: Extract features for expansion pool")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("X1.9e: Feature Extraction for Expansion Pool")
    logger.info("=" * 70)

    # Load labeled files
    logger.info(f"Loading labels: {LABELS_CSV}")
    labels_df = pd.read_csv(LABELS_CSV)
    filepaths = labels_df["filename"].tolist()
    n_files = len(filepaths)
    logger.info(f"Total files to extract: {n_files:,}")
    logger.info(f"  Class 1: {(labels_df['label'] == 1).sum():,}, "
                f"Class 0: {(labels_df['label'] == 0).sum():,}")
    logger.info(f"  Feature dimensions: S={N_TOTAL_S}, V={N_TOTAL_V}")

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Initialize arrays
    features_s = np.zeros((n_files, N_TOTAL_S), dtype=np.float32)
    features_v = np.zeros((n_files, N_TOTAL_V), dtype=np.float32)
    log_entries = []
    start_idx = 0

    # Resume from checkpoint
    if args.resume and os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt["processed"]
        logger.info(f"Resuming from index {start_idx:,}")
        if os.path.exists(f"{PREFIX}_features_s_partial.npy"):
            partial_s = np.load(f"{PREFIX}_features_s_partial.npy")
            partial_v = np.load(f"{PREFIX}_features_v_partial.npy")
            features_s[:start_idx] = partial_s[:start_idx]
            features_v[:start_idx] = partial_v[:start_idx]
        if os.path.exists(f"{PREFIX}_extraction_log_partial.csv"):
            log_df = pd.read_csv(f"{PREFIX}_extraction_log_partial.csv")
            log_entries = log_df.to_dict("records")

    n_workers = min(args.workers, os.cpu_count() or 4)
    batch_size = args.batch_size
    logger.info(f"Workers: {n_workers}, Batch size: {batch_size}")

    overall_start = time.time()
    completed_count = 0
    total_to_do = n_files - start_idx

    for batch_lo in range(start_idx, n_files, batch_size):
        batch_hi = min(batch_lo + batch_size, n_files)
        batch_files = filepaths[batch_lo:batch_hi]
        batch_indices = list(range(batch_lo, batch_hi))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(extract_one_file, fp): idx
                       for fp, idx in zip(batch_files, batch_indices)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    feat_s, feat_v, elapsed_ms, error, nan_count = \
                        future.result()
                except Exception as e:
                    feat_s = np.full(N_TOTAL_S, np.nan, dtype=np.float32)
                    feat_v = np.full(N_TOTAL_V, np.nan, dtype=np.float32)
                    elapsed_ms, error, nan_count = 0.0, str(e), -1

                features_s[idx] = feat_s.astype(np.float32)
                features_v[idx] = feat_v.astype(np.float32)
                log_entries.append({
                    "index": idx,
                    "filepath": filepaths[idx],
                    "time_ms": round(elapsed_ms, 1),
                    "error": error,
                    "nan_count": nan_count,
                })
                completed_count += 1

        # Checkpoint after each batch
        processed = batch_hi
        elapsed_total = time.time() - overall_start
        rate = completed_count / elapsed_total if elapsed_total > 0 else 0
        remaining = n_files - processed
        eta_min = remaining / rate / 60 if rate > 0 else 0

        batch_nan = int(np.sum(~np.isfinite(features_s[batch_lo:batch_hi])))
        batch_errors = sum(
            1 for e in log_entries[-(batch_hi - batch_lo):] if e["error"])

        logger.info(
            f"[{processed:,}/{n_files:,}] ({100*processed/n_files:.1f}%) "
            f"{rate:.1f} files/s, ETA {eta_min:.0f}min, "
            f"batch NaN={batch_nan}, errors={batch_errors}"
        )

        # Save partial outputs
        np.save(f"{PREFIX}_features_s_partial.npy", features_s)
        np.save(f"{PREFIX}_features_v_partial.npy", features_v)
        pd.DataFrame(log_entries).to_csv(
            f"{PREFIX}_extraction_log_partial.csv", index=False
        )
        with open(CHECKPOINT, "wb") as f:
            pickle.dump({"processed": processed}, f)

    # Save final outputs
    np.save(OUTPUT_S, features_s)
    np.save(OUTPUT_V, features_v)
    pd.DataFrame(log_entries).to_csv(OUTPUT_LOG, index=False)

    # Cleanup partial files
    for f in [CHECKPOINT,
              f"{PREFIX}_features_s_partial.npy",
              f"{PREFIX}_features_v_partial.npy",
              f"{PREFIX}_extraction_log_partial.csv"]:
        if os.path.exists(f):
            os.remove(f)

    # Summary
    total_errors = sum(1 for e in log_entries if e["error"])
    total_nan_files = int(np.sum(np.any(~np.isfinite(features_s), axis=1)))
    elapsed_total = time.time() - overall_start

    logger.info("=" * 70)
    logger.info("X1.9e Feature Extraction COMPLETE")
    logger.info(f"  Files: {n_files:,}, Errors: {total_errors:,}, "
                f"NaN files: {total_nan_files:,}")
    logger.info(f"  Time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.2f} hrs)")
    logger.info(f"  Rate: {n_files/elapsed_total:.1f} files/s")
    logger.info(f"  Output S: {OUTPUT_S} {features_s.shape}")
    logger.info(f"  Output V: {OUTPUT_V} {features_v.shape}")
    logger.info(f"  S value range: [{np.nanmin(features_s):.4f}, "
                f"{np.nanmax(features_s):.4f}]")
    logger.info(f"  V value range: [{np.nanmin(features_v):.4f}, "
                f"{np.nanmax(features_v):.4f}]")


if __name__ == "__main__":
    main()
