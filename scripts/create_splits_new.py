"""Create test split CSVs for VPQAD and VSEA DC datasets.

Outputs:
  - data/step1/splits/test_vpqad.csv   (332 rows: Lab + Cafeteria, TD + TID)
  - data/step1/splits/test_vseadc.csv  (336 rows: WAV only, no M4A, no df_* deepfakes)

Usage:
    python scripts/create_splits_new.py
"""

import csv
from pathlib import Path

BASE = Path(r"D:\VQI")
DATASETS = BASE / "Datasets"
SPLITS_DIR = BASE / "implementation" / "data" / "step1" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(filepath: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write CSV with header: filename,speaker_id,dataset_source."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "dataset_source"])
        writer.writerows(rows)


# ── VPQAD ────────────────────────────────────────────────────────────────
print("Scanning VPQAD...")
vpqad_root = DATASETS / "VPQAD"

vpqad_rows = []
for session_dir in ["Lab_Session_Data", "Cafeteria_Data"]:
    for task_dir in ["TD", "TID"]:
        folder = vpqad_root / session_dir / task_dir
        if not folder.exists():
            print(f"  WARNING: {folder} not found, skipping")
            continue
        wavs = sorted(folder.glob("*.wav"))
        for p in wavs:
            # Filename like sub001_1_td.wav -> speaker_id = sub001
            speaker_id = p.stem.split("_")[0]  # e.g. "sub001"
            vpqad_rows.append((str(p), speaker_id, "vpqad"))
        print(f"  {session_dir}/{task_dir}: {len(wavs)} files")

print(f"  VPQAD total: {len(vpqad_rows)}")

# Verify speaker count
vpqad_speakers = set(row[1] for row in vpqad_rows)
print(f"  VPQAD speakers: {len(vpqad_speakers)}")

# ── VSEA DC ──────────────────────────────────────────────────────────────
print("\nScanning VSEA DC...")
vseadc_root = DATASETS / "VSEA DC"

vseadc_rows = []
# Collect all sub* and Sub* directories
sub_dirs = sorted(
    [d for d in vseadc_root.iterdir() if d.is_dir() and d.name.lower().startswith("sub")],
    key=lambda d: d.name.lower(),
)

for sub_dir in sub_dirs:
    # Normalize speaker_id: Sub023 -> sub023
    speaker_id = sub_dir.name.lower()

    # Collect only .wav files, skip df_* deepfakes and .m4a
    wavs = sorted(sub_dir.glob("*.wav"))
    genuine_wavs = [w for w in wavs if not w.name.startswith("df_")]
    for p in genuine_wavs:
        vseadc_rows.append((str(p), speaker_id, "vseadc"))

print(f"  VSEA DC total: {len(vseadc_rows)}")
vseadc_speakers = set(row[1] for row in vseadc_rows)
print(f"  VSEA DC speakers: {len(vseadc_speakers)}")

# ── Write CSVs ────────────────────────────────────────────────────────────
print("\nWriting CSVs...")

csv_files = {
    "test_vpqad.csv": vpqad_rows,
    "test_vseadc.csv": vseadc_rows,
}

for name, rows in csv_files.items():
    path = SPLITS_DIR / name
    write_csv(path, rows)
    print(f"  {name:<25s} -> {len(rows):>6,} rows, {len(set(r[1] for r in rows))} speakers")

# ── Verification ──────────────────────────────────────────────────────────
print("\n--- Verification ---")

# Check all files exist
for name, rows in csv_files.items():
    missing = [r[0] for r in rows if not Path(r[0]).exists()]
    if missing:
        print(f"  WARNING: {name} has {len(missing)} missing files!")
        for m in missing[:5]:
            print(f"    {m}")
    else:
        print(f"  {name}: all {len(rows)} files exist")

# Check speaker IDs are lowercase
for name, rows in csv_files.items():
    upper = [r[1] for r in rows if r[1] != r[1].lower()]
    if upper:
        print(f"  WARNING: {name} has {len(upper)} non-lowercase speaker IDs!")
    else:
        print(f"  {name}: all speaker IDs lowercase")

# Check each speaker has >= 2 files (needed for genuine pairs)
for name, rows in csv_files.items():
    spk_counts = {}
    for r in rows:
        spk_counts[r[1]] = spk_counts.get(r[1], 0) + 1
    single_utt = [spk for spk, cnt in spk_counts.items() if cnt < 2]
    if single_utt:
        print(f"  WARNING: {name} has {len(single_utt)} speakers with < 2 utterances: {single_utt[:5]}")
    else:
        min_utt = min(spk_counts.values())
        max_utt = max(spk_counts.values())
        print(f"  {name}: all speakers have >= 2 utterances (range: {min_utt}-{max_utt})")

print("\nDone.")
