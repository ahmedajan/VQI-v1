"""
Step 1.4 -- Create Train/Val/Test Split Manifests

Collects file lists from all 8 datasets and produces 7 CSV manifests:
  - train_pool.csv     (~1.21M rows: VoxCeleb1-dev + VoxCeleb2-dev remainder + VOiCES)
  - val_set.csv        (50,000 rows: sampled from VoxCeleb1-dev + VoxCeleb2-dev)
  - test_voxceleb1.csv (40 held-out speakers from VoxCeleb1)
  - test_vctk.csv      (mic1 only)
  - test_librispeech_clean.csv
  - test_librispeech_other.csv
  - test_cnceleb.csv
"""

import csv
import random
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"D:\VQI")
DATASETS = BASE / "Datasets"
SPLITS_DIR = BASE / "implementation" / "data" / "splits"
TEST_SPEAKERS_FILE = BASE / "implementation" / "data" / "voxceleb1_test_speakers.txt"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load VoxCeleb1 test speaker IDs ────────────────────────────────────────
test_speaker_ids = set()
with open(TEST_SPEAKERS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sid = line.strip()
        if sid:
            test_speaker_ids.add(sid)

print(f"Loaded {len(test_speaker_ids)} VoxCeleb1 test speaker IDs")
assert len(test_speaker_ids) == 40, f"Expected 40 test speakers, got {len(test_speaker_ids)}"


def collect_files(root: Path, pattern: str) -> list[Path]:
    """Collect files matching glob pattern under root."""
    files = sorted(root.glob(pattern))
    return files


def write_csv(filepath: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write CSV with header: filename,speaker_id,dataset_source."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "dataset_source"])
        writer.writerows(rows)


# ── 1. VoxCeleb1 ──────────────────────────────────────────────────────────
print("Scanning VoxCeleb1...")
vc1_root = DATASETS / "voxCELEB1" / "wav"
vc1_all = collect_files(vc1_root, "**/*.wav")

vc1_dev_rows = []
vc1_test_rows = []
for p in vc1_all:
    spk = p.parent.parent.name  # wav/{spk}/{vid}/{utt}.wav
    row = (str(p), spk)
    if spk in test_speaker_ids:
        vc1_test_rows.append((*row, "voxceleb1_test"))
    else:
        vc1_dev_rows.append((*row, "voxceleb1_dev"))

print(f"  VoxCeleb1-dev:  {len(vc1_dev_rows):>10,}")
print(f"  VoxCeleb1-test: {len(vc1_test_rows):>10,}")

# ── 2. VoxCeleb2-dev ──────────────────────────────────────────────────────
print("Scanning VoxCeleb2-dev...")
vc2_root = DATASETS / "voxCELEB2" / "dev" / "wav"
vc2_all = collect_files(vc2_root, "**/*.wav")

vc2_dev_rows = []
for p in vc2_all:
    spk = p.parent.parent.name  # dev/wav/{spk}/{vid}/{utt}.wav
    vc2_dev_rows.append((str(p), spk, "voxceleb2_dev"))

print(f"  VoxCeleb2-dev:  {len(vc2_dev_rows):>10,}")

# ── 3. VOiCES ─────────────────────────────────────────────────────────────
print("Scanning VOiCES...")
voices_root = DATASETS / "VOiCES" / "VOiCES_devkit"

# Source files: source-16k/{split}/{spk}/*.wav
voices_source = collect_files(voices_root / "source-16k", "**/*.wav")
voices_source_rows = []
for p in voices_source:
    spk = p.parent.name  # source-16k/{split}/{spk}/{file}.wav
    voices_source_rows.append((str(p), spk, "voices"))

# Distant speech files: distant-16k/speech/{split}/{room}/{noise}/{spk}/*.wav
voices_distant = collect_files(voices_root / "distant-16k" / "speech", "**/*.wav")
voices_distant_rows = []
for p in voices_distant:
    # Extract spXXXX from filename
    match = re.search(r"(sp\d{4})", p.name)
    if match:
        spk = match.group(1)
        voices_distant_rows.append((str(p), spk, "voices"))

voices_rows = voices_source_rows + voices_distant_rows
print(f"  VOiCES source:  {len(voices_source_rows):>10,}")
print(f"  VOiCES distant: {len(voices_distant_rows):>10,}")
print(f"  VOiCES total:   {len(voices_rows):>10,}")

# ── 4. VCTK (mic1 only) ──────────────────────────────────────────────────
print("Scanning VCTK...")
vctk_root = DATASETS / "VCTK" / "wav48_silence_trimmed"
vctk_all = collect_files(vctk_root, "**/*_mic1.flac")

vctk_rows = []
for p in vctk_all:
    spk = p.parent.name  # wav48_silence_trimmed/{spk}/{file}.flac
    vctk_rows.append((str(p), spk, "vctk"))

print(f"  VCTK mic1:      {len(vctk_rows):>10,}")

# ── 5. LibriSpeech test-clean ─────────────────────────────────────────────
print("Scanning LibriSpeech test-clean...")
ls_clean_root = DATASETS / "librispeech" / "LibriSpeech" / "test-clean"
ls_clean_all = collect_files(ls_clean_root, "**/*.flac")

ls_clean_rows = []
for p in ls_clean_all:
    spk = p.parent.parent.name  # test-clean/{spk}/{book}/{file}.flac
    ls_clean_rows.append((str(p), spk, "librispeech_test_clean"))

print(f"  LS test-clean:  {len(ls_clean_rows):>10,}")

# ── 6. LibriSpeech test-other ─────────────────────────────────────────────
print("Scanning LibriSpeech test-other...")
ls_other_root = DATASETS / "librispeech" / "LibriSpeech" / "test-other"
ls_other_all = collect_files(ls_other_root, "**/*.flac")

ls_other_rows = []
for p in ls_other_all:
    spk = p.parent.parent.name  # test-other/{spk}/{book}/{file}.flac
    ls_other_rows.append((str(p), spk, "librispeech_test_other"))

print(f"  LS test-other:  {len(ls_other_rows):>10,}")

# ── 7. CN-Celeb ───────────────────────────────────────────────────────────
print("Scanning CN-Celeb...")
cn_root = DATASETS / "CN-Celeb" / "CN-Celeb_flac" / "data"
cn_all = collect_files(cn_root, "**/*.flac")

cn_rows = []
for p in cn_all:
    spk = p.parent.name  # data/{spk}/{file}.flac
    cn_rows.append((str(p), spk, "cnceleb"))

print(f"  CN-Celeb:       {len(cn_rows):>10,}")

# ── Create validation set (50K from VoxCeleb1-dev + VoxCeleb2-dev) ────────
print("\nCreating validation set...")
dev_pool = vc1_dev_rows + vc2_dev_rows
print(f"  Dev pool size:  {len(dev_pool):>10,}")

random.seed(42)
val_rows = random.sample(dev_pool, 50_000)
val_filenames = {row[0] for row in val_rows}

print(f"  Val set size:   {len(val_rows):>10,}")

# ── Create training pool (dev remainder + VOiCES) ─────────────────────────
print("Creating training pool...")
train_rows = [row for row in dev_pool if row[0] not in val_filenames]
train_rows.extend(voices_rows)
print(f"  Train pool:     {len(train_rows):>10,}")

# ── Verification ──────────────────────────────────────────────────────────
print("\n--- Verification ---")

# No overlap between train and val
train_filenames = {row[0] for row in train_rows}
overlap_train_val = train_filenames & val_filenames
print(f"  Train-Val overlap: {len(overlap_train_val)}")
assert len(overlap_train_val) == 0, "FAIL: Train-Val overlap detected!"

# Test speaker IDs should not appear in train/val
train_val_speakers = {row[1] for row in train_rows} | {row[1] for row in val_rows}
leaked_test_speakers = test_speaker_ids & train_val_speakers
print(f"  Test speaker leaks into train/val: {len(leaked_test_speakers)}")
assert len(leaked_test_speakers) == 0, f"FAIL: Test speakers found in train/val: {leaked_test_speakers}"

# Test files should not appear in train/val
test_all_filenames = (
    {row[0] for row in vc1_test_rows}
    | {row[0] for row in vctk_rows}
    | {row[0] for row in ls_clean_rows}
    | {row[0] for row in ls_other_rows}
    | {row[0] for row in cn_rows}
)
overlap_trainval_test = (train_filenames | val_filenames) & test_all_filenames
print(f"  Train/Val-Test file overlap: {len(overlap_trainval_test)}")
assert len(overlap_trainval_test) == 0, "FAIL: Train/Val-Test file overlap detected!"

print("  All checks PASSED")

# ── Write CSVs ────────────────────────────────────────────────────────────
print("\nWriting CSVs...")

csv_files = {
    "val_set.csv": val_rows,
    "train_pool.csv": train_rows,
    "test_voxceleb1.csv": vc1_test_rows,
    "test_vctk.csv": vctk_rows,
    "test_librispeech_clean.csv": ls_clean_rows,
    "test_librispeech_other.csv": ls_other_rows,
    "test_cnceleb.csv": cn_rows,
}

for name, rows in csv_files.items():
    path = SPLITS_DIR / name
    write_csv(path, rows)
    print(f"  {name:<30s} -> {len(rows):>10,} rows")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
total = sum(len(r) for r in csv_files.values())
print(f"  Total files across all splits: {total:,}")
print(f"  Train pool:              {len(train_rows):>10,}")
print(f"  Validation set:          {len(val_rows):>10,}")
print(f"  Test VoxCeleb1:          {len(vc1_test_rows):>10,}")
print(f"  Test VCTK:               {len(vctk_rows):>10,}")
print(f"  Test LibriSpeech clean:  {len(ls_clean_rows):>10,}")
print(f"  Test LibriSpeech other:  {len(ls_other_rows):>10,}")
print(f"  Test CN-Celeb:           {len(cn_rows):>10,}")
print("\nDone.")
