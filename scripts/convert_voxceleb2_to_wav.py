"""
VoxCeleb2 M4A -> WAV Conversion Script (Sub-task 1.2a)

Converts all .m4a files under D:/VQI/Datasets/voxCELEB2/ to 16kHz 16-bit
mono WAV using ffmpeg. Creates parallel `wav/` directories alongside `aac/`.

Directory mapping:
    dev/aac/{spk}/{vid}/{utt}.m4a  ->  dev/wav/{spk}/{vid}/{utt}.wav
    aac/{spk}/{vid}/{utt}.m4a      ->  wav/{spk}/{vid}/{utt}.wav

Usage:
    python implementation/scripts/convert_voxceleb2_to_wav.py
    python implementation/scripts/convert_voxceleb2_to_wav.py --workers 8
    python implementation/scripts/convert_voxceleb2_to_wav.py --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

VOXCELEB2_ROOT = Path("D:/VQI/Datasets/voxCELEB2")

# Source directories containing .m4a files
SOURCE_DIRS = [
    VOXCELEB2_ROOT / "dev" / "aac",   # dev set
    VOXCELEB2_ROOT / "aac",            # test set
]

# ffmpeg conversion settings
FFMPEG_ARGS = [
    "-y",                  # overwrite output without asking
    "-ac", "1",            # mono
    "-ar", "16000",        # 16kHz sample rate
    "-acodec", "pcm_s16le",  # 16-bit signed little-endian PCM
]

LOG_FILE = VOXCELEB2_ROOT / "conversion_log.txt"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def find_m4a_files():
    """Discover all .m4a files under the source directories."""
    m4a_files = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            logging.warning("Source directory not found: %s", src_dir)
            continue
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                if fname.lower().endswith(".m4a"):
                    m4a_files.append(Path(root) / fname)
    return m4a_files


def m4a_to_wav_path(m4a_path: Path) -> Path:
    """
    Map an .m4a path to its corresponding .wav path.
    Replaces the 'aac' directory component with 'wav'.

    dev/aac/id00012/vid/00001.m4a -> dev/wav/id00012/vid/00001.wav
    aac/id00017/vid/00001.m4a     -> wav/id00017/vid/00001.wav
    """
    parts = list(m4a_path.parts)
    # Replace the 'aac' component closest to the speaker directories
    # Walk from the end to find the 'aac' directory
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "aac":
            parts[i] = "wav"
            break
    wav_path = Path(*parts).with_suffix(".wav")
    return wav_path


def convert_one(m4a_path: Path) -> tuple:
    """
    Convert a single .m4a file to .wav using ffmpeg.
    Returns (m4a_path, wav_path, success, error_msg).
    """
    wav_path = m4a_to_wav_path(m4a_path)

    # Skip if already converted
    if wav_path.exists():
        return (str(m4a_path), str(wav_path), True, "skipped (exists)")

    # Ensure output directory exists
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(m4a_path),
        *FFMPEG_ARGS,
        str(wav_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return (str(m4a_path), str(wav_path), True, "")
        else:
            # Clean up partial output
            if wav_path.exists():
                wav_path.unlink()
            return (str(m4a_path), str(wav_path), False, result.stderr[-200:])
    except subprocess.TimeoutExpired:
        if wav_path.exists():
            wav_path.unlink()
        return (str(m4a_path), str(wav_path), False, "timeout")
    except Exception as e:
        if wav_path.exists():
            wav_path.unlink()
        return (str(m4a_path), str(wav_path), False, str(e))


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert VoxCeleb2 .m4a files to 16kHz 16-bit mono WAV"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, cpu_count() - 2),
        help="Number of parallel worker processes (default: cpu_count - 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only count files, don't convert"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        ],
    )

    logging.info("=" * 60)
    logging.info("VoxCeleb2 M4A -> WAV Conversion")
    logging.info("=" * 60)
    logging.info("Root: %s", VOXCELEB2_ROOT)
    logging.info("Workers: %d", args.workers)

    # Discover files
    logging.info("Scanning for .m4a files...")
    m4a_files = find_m4a_files()
    total = len(m4a_files)
    logging.info("Found %d .m4a files", total)

    if total == 0:
        logging.warning("No .m4a files found. Nothing to do.")
        return

    # Check how many are already converted
    already_done = sum(1 for f in m4a_files if m4a_to_wav_path(f).exists())
    remaining = total - already_done
    logging.info("Already converted: %d", already_done)
    logging.info("Remaining: %d", remaining)

    if args.dry_run:
        logging.info("Dry run — exiting without conversion.")
        # Show a few example mappings
        for f in m4a_files[:5]:
            logging.info("  %s -> %s", f, m4a_to_wav_path(f))
        return

    # Convert
    logging.info("Starting conversion...")
    t0 = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_files = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_one, m4a_files, chunksize=64)):
            m4a_str, wav_str, ok, msg = result
            if ok:
                if "skipped" in msg:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                failed_files.append((m4a_str, msg))

            done = i + 1
            if done % 10000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logging.info(
                    "Progress: %d/%d (%.1f%%) | OK=%d Skip=%d Fail=%d | "
                    "%.0f files/s | ETA %.0fs",
                    done, total, 100 * done / total,
                    success_count, skip_count, fail_count,
                    rate, eta,
                )

    elapsed = time.time() - t0
    logging.info("=" * 60)
    logging.info("CONVERSION COMPLETE")
    logging.info("=" * 60)
    logging.info("Total files:    %d", total)
    logging.info("Converted:      %d", success_count)
    logging.info("Skipped:        %d", skip_count)
    logging.info("Failed:         %d", fail_count)
    logging.info("Time:           %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    if failed_files:
        logging.warning("Failed files:")
        for path, err in failed_files[:50]:
            logging.warning("  %s — %s", path, err)
        if len(failed_files) > 50:
            logging.warning("  ... and %d more", len(failed_files) - 50)

    logging.info("Log saved to: %s", LOG_FILE)


if __name__ == "__main__":
    main()
