"""X1.9 Step 1: Build expansion pool CSV for new datasets.

Scans Common Voice (English) and LibriSpeech train sets, creates a unified
split CSV in the same format as train_pool.csv for the embedding pipeline.

Common Voice: MP3 files, speaker IDs from train.tsv (client_id column).
LibriSpeech train: FLAC files, speaker IDs from directory structure.

Filters:
  - Common Voice: Only speakers with >= 5 clips (need genuine pairs)
  - LibriSpeech: All speakers included (avg ~120 clips/speaker)

Output: implementation/data/step1/splits/x1_expansion_pool.csv
"""

import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Datasets"))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "step1", "splits")

# Common Voice paths
CV_BASE = os.path.join(DATASETS_DIR, "Common Voice",
                       "cv-corpus-24.0-2025-12-05-en",
                       "cv-corpus-24.0-2025-12-05", "en")
CV_CLIPS = os.path.join(CV_BASE, "clips")
CV_TRAIN_TSV = os.path.join(CV_BASE, "train.tsv")

# LibriSpeech train paths
LS_BASE = os.path.join(DATASETS_DIR, "librispeech", "LibriSpeech")
LS_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]


def build_common_voice_entries():
    """Parse Common Voice train.tsv, filter speakers with >= 5 clips."""
    print("Scanning Common Voice train.tsv...")
    speaker_files = {}  # client_id -> list of filenames

    with open(CV_TRAIN_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row["client_id"]
            path = row["path"]
            if cid not in speaker_files:
                speaker_files[cid] = []
            speaker_files[cid].append(path)

    # Filter: >= 5 clips per speaker
    total_speakers = len(speaker_files)
    speaker_files = {k: v for k, v in speaker_files.items() if len(v) >= 5}
    filtered_speakers = len(speaker_files)

    entries = []
    for cid, files in speaker_files.items():
        # Use short hash of client_id as speaker_id
        spk_id = f"cv_{cid[:12]}"
        for fname in files:
            full_path = os.path.join(CV_CLIPS, fname)
            entries.append((full_path, spk_id, "common_voice_en"))

    print(f"  Common Voice: {total_speakers} total speakers, "
          f"{filtered_speakers} with >= 5 clips")
    print(f"  Files: {len(entries)}")
    return entries


def build_librispeech_entries():
    """Scan LibriSpeech train directories for FLAC files."""
    print("Scanning LibriSpeech train splits...")
    entries = []
    speakers = set()

    for split_name in LS_SPLITS:
        # Handle nested structure: LibriSpeech/train-X/LibriSpeech/train-X/
        split_dir = os.path.join(LS_BASE, split_name, "LibriSpeech", split_name)
        if not os.path.isdir(split_dir):
            # Try direct path
            split_dir = os.path.join(LS_BASE, split_name)

        if not os.path.isdir(split_dir):
            print(f"  WARNING: {split_dir} not found, skipping")
            continue

        for speaker_id in os.listdir(split_dir):
            speaker_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            speakers.add(speaker_id)
            spk_id = f"ls_{speaker_id}"
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if not os.path.isdir(chapter_dir):
                    continue
                for fname in os.listdir(chapter_dir):
                    if fname.endswith(".flac"):
                        full_path = os.path.join(chapter_dir, fname)
                        entries.append((full_path, spk_id,
                                       f"librispeech_{split_name}"))

    print(f"  LibriSpeech: {len(speakers)} speakers, {len(entries)} files")
    return entries


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Build entries from both datasets
    cv_entries = build_common_voice_entries()
    ls_entries = build_librispeech_entries()

    all_entries = cv_entries + ls_entries
    print(f"\nTotal expansion pool: {len(all_entries)} files")

    # Write split CSV
    out_path = os.path.join(SPLITS_DIR, "x1_expansion_pool.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "dataset_source"])
        for filepath, spk_id, source in all_entries:
            writer.writerow([filepath, spk_id, source])

    print(f"Saved: {out_path}")

    # Summary stats
    sources = {}
    spk_set = set()
    for _, spk_id, source in all_entries:
        sources[source] = sources.get(source, 0) + 1
        spk_set.add(spk_id)

    print(f"\nSummary:")
    print(f"  Total speakers: {len(spk_set)}")
    print(f"  Total files: {len(all_entries)}")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")


if __name__ == "__main__":
    main()
