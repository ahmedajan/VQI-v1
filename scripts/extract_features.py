"""Step 4: Feature extraction script.

Extracts 544 VQI-S + 161 VQI-V features for training and validation sets.
Supports --resume for checkpoint recovery.

Uses ProcessPoolExecutor with proper Windows spawn support.

Usage:
    python extract_features.py [--resume] [--workers 8] [--batch-size 500]
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Setup path BEFORE any VQI imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from vqi.core.feature_orchestrator import get_feature_names_s, N_TOTAL_S
from vqi.core.feature_orchestrator_v import get_feature_names_v, N_TOTAL_V

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = os.path.join(_IMPL_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "step4", "features")
TRAIN_CSV = os.path.join(DATA_DIR, "step2", "training_set_final.csv")
VAL_CSV = os.path.join(DATA_DIR, "step1", "splits", "val_set.csv")


def extract_one_file(filepath):
    """Extract features for a single audio file. Top-level for ProcessPoolExecutor."""
    import warnings
    warnings.filterwarnings("ignore")

    # Ensure VQI modules are importable in spawned worker
    impl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)

    t0 = time.time()
    try:
        from vqi.preprocessing.audio_loader import load_audio
        from vqi.preprocessing.normalize import dc_remove_and_normalize
        from vqi.preprocessing.vad import energy_vad
        from vqi.core.feature_orchestrator import compute_all_features
        from vqi.core.feature_orchestrator_v import compute_all_features_v

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
        nan_count = int(np.sum(~np.isfinite(feat_arr_s))) + int(np.sum(~np.isfinite(feat_arr_v)))
        return feat_arr_s, feat_arr_v, elapsed, "", nan_count
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return np.full(N_TOTAL_S, np.nan), np.full(N_TOTAL_V, np.nan), elapsed, str(e), -1


def run_extraction(filepaths, output_prefix, checkpoint_path, n_workers=8, batch_size=500, start_idx=0):
    """Extract features with ProcessPoolExecutor and batch checkpointing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_files = len(filepaths)
    features_s = np.zeros((n_files, N_TOTAL_S), dtype=np.float32)
    features_v = np.zeros((n_files, N_TOTAL_V), dtype=np.float32)
    log_entries = []

    # Load checkpoint if resuming
    if start_idx > 0 and os.path.exists(f"{output_prefix}_features_s_partial.npy"):
        logger.info(f"Resuming from index {start_idx}")
        partial_s = np.load(f"{output_prefix}_features_s_partial.npy")
        partial_v = np.load(f"{output_prefix}_features_v_partial.npy")
        features_s[:start_idx] = partial_s[:start_idx]
        features_v[:start_idx] = partial_v[:start_idx]
        if os.path.exists(f"{output_prefix}_extraction_log_partial.csv"):
            log_df = pd.read_csv(f"{output_prefix}_extraction_log_partial.csv")
            log_entries = log_df.to_dict("records")

    overall_start = time.time()
    completed_count = 0
    total_to_do = n_files - start_idx

    # Process in batches using ProcessPoolExecutor
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
                    feat_s, feat_v, elapsed_ms, error, nan_count = future.result()
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
        eta = remaining / rate if rate > 0 else 0

        # Batch NaN check
        batch_nan = int(np.sum(~np.isfinite(features_s[batch_lo:batch_hi])))
        batch_errors = sum(1 for e in log_entries[-(batch_hi - batch_lo):] if e["error"])

        logger.info(
            f"[{processed}/{n_files}] "
            f"{rate:.1f} files/s, ETA {eta/60:.1f}min, "
            f"batch NaN/Inf={batch_nan}, batch errors={batch_errors}"
        )

        # Save partial outputs
        np.save(f"{output_prefix}_features_s_partial.npy", features_s)
        np.save(f"{output_prefix}_features_v_partial.npy", features_v)
        pd.DataFrame(log_entries).to_csv(
            f"{output_prefix}_extraction_log_partial.csv", index=False
        )
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"processed": processed}, f)

    return features_s, features_v, log_entries


def main():
    parser = argparse.ArgumentParser(description="Extract VQI features")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load file lists
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    logger.info(f"Training files: {len(train_df)}, Validation files: {len(val_df)}")

    # Save feature names
    names_s = get_feature_names_s()
    names_v = get_feature_names_v()
    with open(os.path.join(FEATURES_DIR, "feature_names_s.json"), "w", encoding="utf-8") as f:
        json.dump(names_s, f, indent=2)
    with open(os.path.join(FEATURES_DIR, "feature_names_v.json"), "w", encoding="utf-8") as f:
        json.dump(names_v, f, indent=2)
    logger.info(f"Feature names saved: {len(names_s)} VQI-S, {len(names_v)} VQI-V")

    # ===== Training Set =====
    train_checkpoint = os.path.join(FEATURES_DIR, "_checkpoint_train.pkl")
    train_prefix = os.path.join(FEATURES_DIR, "train")
    train_start = 0

    if args.resume and os.path.exists(train_checkpoint):
        with open(train_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        train_start = ckpt["processed"]
        logger.info(f"Resuming training extraction from {train_start}")

    train_files = train_df["filename"].tolist()
    logger.info(f"Extracting training features ({train_start} -> {len(train_files)})...")
    feat_s_train, feat_v_train, log_train = run_extraction(
        train_files, train_prefix, train_checkpoint,
        n_workers=args.workers, batch_size=args.batch_size, start_idx=train_start,
    )

    # Save final training outputs
    np.save(os.path.join(FEATURES_DIR, "features_s_train.npy"), feat_s_train)
    np.save(os.path.join(FEATURES_DIR, "features_v_train.npy"), feat_v_train)
    pd.DataFrame(log_train).to_csv(
        os.path.join(FEATURES_DIR, "extraction_log_train.csv"), index=False
    )
    logger.info(f"Training features saved: S={feat_s_train.shape}, V={feat_v_train.shape}")

    # Cleanup train checkpoint
    for f in [train_checkpoint,
              f"{train_prefix}_features_s_partial.npy",
              f"{train_prefix}_features_v_partial.npy",
              f"{train_prefix}_extraction_log_partial.csv"]:
        if os.path.exists(f):
            os.remove(f)

    # ===== Validation Set =====
    val_checkpoint = os.path.join(FEATURES_DIR, "_checkpoint_val.pkl")
    val_prefix = os.path.join(FEATURES_DIR, "val")
    val_start = 0

    if args.resume and os.path.exists(val_checkpoint):
        with open(val_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        val_start = ckpt["processed"]
        logger.info(f"Resuming validation extraction from {val_start}")

    val_files = val_df["filename"].tolist()
    logger.info(f"Extracting validation features ({val_start} -> {len(val_files)})...")
    feat_s_val, feat_v_val, log_val = run_extraction(
        val_files, val_prefix, val_checkpoint,
        n_workers=args.workers, batch_size=args.batch_size, start_idx=val_start,
    )

    # Save final validation outputs
    np.save(os.path.join(FEATURES_DIR, "features_s_val.npy"), feat_s_val)
    np.save(os.path.join(FEATURES_DIR, "features_v_val.npy"), feat_v_val)
    pd.DataFrame(log_val).to_csv(
        os.path.join(FEATURES_DIR, "extraction_log_val.csv"), index=False
    )
    logger.info(f"Validation features saved: S={feat_s_val.shape}, V={feat_v_val.shape}")

    # Cleanup val checkpoint
    for f in [val_checkpoint,
              f"{val_prefix}_features_s_partial.npy",
              f"{val_prefix}_features_v_partial.npy",
              f"{val_prefix}_extraction_log_partial.csv"]:
        if os.path.exists(f):
            os.remove(f)

    # ===== Final Summary =====
    total_errors_train = sum(1 for e in log_train if e["error"])
    total_errors_val = sum(1 for e in log_val if e["error"])
    logger.info("=" * 60)
    logger.info("Feature extraction complete!")
    logger.info(f"  Training:   {feat_s_train.shape[0]} files, {total_errors_train} errors")
    logger.info(f"  Validation: {feat_s_val.shape[0]} files, {total_errors_val} errors")
    logger.info(f"  VQI-S: {N_TOTAL_S} features, VQI-V: {N_TOTAL_V} features")
    logger.info(f"  Output: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
