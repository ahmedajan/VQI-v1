"""Step 8: Extract VQI features for test splits.

Extends the Step 4 pattern for test datasets (VoxCeleb1-test, VCTK, CN-Celeb).
Reuses extract_one_file() worker from extract_features.py.

Usage:
    python scripts/extract_features_test.py --split test_voxceleb1 [--workers 8] [--batch-size 500] [--resume]
    python scripts/extract_features_test.py --split test_vctk --workers 8 --resume
    python scripts/extract_features_test.py --split test_cnceleb --workers 8 --resume
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from vqi.core.feature_orchestrator import get_feature_names_s, N_TOTAL_S
from vqi.core.feature_orchestrator_v import get_feature_names_v, N_TOTAL_V

# Reuse the worker function from extract_features.py
# Add scripts dir to path for direct import
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from extract_features import extract_one_file, run_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
FEATURES_DIR = os.path.join(DATA_DIR, "features")


def main():
    parser = argparse.ArgumentParser(description="Extract VQI features for test splits")
    parser.add_argument("--split", required=True,
                        help="Split name (e.g., test_voxceleb1, test_vctk, test_cnceleb)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for checkpointing")
    args = parser.parse_args()

    split_csv = os.path.join(SPLITS_DIR, f"{args.split}.csv")
    if not os.path.exists(split_csv):
        logger.error(f"Split file not found: {split_csv}")
        sys.exit(1)

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load file list
    df = pd.read_csv(split_csv)
    filepaths = df["filename"].tolist()
    logger.info(f"Split '{args.split}': {len(filepaths)} files")

    # Save feature names (if not already saved)
    names_s_path = os.path.join(FEATURES_DIR, "feature_names_s.json")
    names_v_path = os.path.join(FEATURES_DIR, "feature_names_v.json")
    if not os.path.exists(names_s_path):
        names_s = get_feature_names_s()
        with open(names_s_path, "w", encoding="utf-8") as f:
            json.dump(names_s, f, indent=2)
    if not os.path.exists(names_v_path):
        names_v = get_feature_names_v()
        with open(names_v_path, "w", encoding="utf-8") as f:
            json.dump(names_v, f, indent=2)

    # Checkpoint path
    checkpoint_path = os.path.join(FEATURES_DIR, f"_checkpoint_{args.split}.pkl")
    output_prefix = os.path.join(FEATURES_DIR, args.split)
    start_idx = 0

    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        start_idx = ckpt["processed"]
        logger.info(f"Resuming from index {start_idx}")

    # Run extraction
    t0 = time.time()
    feat_s, feat_v, log_entries = run_extraction(
        filepaths, output_prefix, checkpoint_path,
        n_workers=args.workers, batch_size=args.batch_size, start_idx=start_idx,
    )
    elapsed = time.time() - t0

    # Save final outputs
    out_s = os.path.join(FEATURES_DIR, f"features_s_{args.split}.npy")
    out_v = os.path.join(FEATURES_DIR, f"features_v_{args.split}.npy")
    out_log = os.path.join(FEATURES_DIR, f"extraction_log_{args.split}.csv")

    np.save(out_s, feat_s)
    np.save(out_v, feat_v)
    pd.DataFrame(log_entries).to_csv(out_log, index=False)

    # Cleanup checkpoint + partial files
    for f_path in [
        checkpoint_path,
        f"{output_prefix}_features_s_partial.npy",
        f"{output_prefix}_features_v_partial.npy",
        f"{output_prefix}_extraction_log_partial.csv",
    ]:
        if os.path.exists(f_path):
            os.remove(f_path)

    # Verify outputs
    total_errors = sum(1 for e in log_entries if e["error"])
    total_nan = int(np.sum(~np.isfinite(feat_s))) + int(np.sum(~np.isfinite(feat_v)))

    logger.info("=" * 60)
    logger.info(f"Feature extraction complete for '{args.split}'")
    logger.info(f"  Files: {len(filepaths)}, Errors: {total_errors}")
    logger.info(f"  VQI-S shape: {feat_s.shape}, VQI-V shape: {feat_v.shape}")
    logger.info(f"  NaN/Inf total: {total_nan}")
    logger.info(f"  S range: [{np.nanmin(feat_s):.2f}, {np.nanmax(feat_s):.2f}]")
    logger.info(f"  V range: [{np.nanmin(feat_v):.2f}, {np.nanmax(feat_v):.2f}]")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"  Outputs: {out_s}, {out_v}")


if __name__ == "__main__":
    main()
