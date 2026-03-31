"""Select 200 conformance test files from VoxCeleb1-test covering full VQI-S score range.

Selects ~20 files per decile (0-10, 11-20, ..., 91-100) for conformance testing.
Copies selected files to conformance/test_files/ and writes a file list.

Outputs:
  conformance/test_files/*.wav          (200 WAV files)
  conformance/conformance_file_list.txt (200 lines: filename)
"""

import csv
import os
import random
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def main():
    random.seed(42)

    scores_path = os.path.join(BASE_DIR, "data", "step8", "full_feature", "test_scores", "vqi_scores_test_voxceleb1.csv")
    conf_dir = os.path.join(BASE_DIR, "conformance")
    files_dir = os.path.join(conf_dir, "test_files")
    os.makedirs(files_dir, exist_ok=True)

    # Load all scores
    rows = []
    with open(scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["filename"]
            vqi_s = int(row["vqi_s"])
            vqi_v = int(row["vqi_v"])
            if os.path.exists(filepath):
                rows.append({"filepath": filepath, "vqi_s": vqi_s, "vqi_v": vqi_v})

    print(f"Loaded {len(rows)} scored files from VoxCeleb1-test")

    # Bin by decile (0-10, 11-20, ..., 91-100)
    decile_bins = {i: [] for i in range(10)}
    for row in rows:
        s = row["vqi_s"]
        decile = min(s // 10, 9)  # 0-9
        decile_bins[decile].append(row)

    print("Decile distribution:")
    for d in range(10):
        lo, hi = d * 10, (d + 1) * 10 if d < 9 else 100
        print(f"  {lo:>3}-{hi:>3}: {len(decile_bins[d])} files")

    # Select ~20 per decile
    selected = []
    for d in range(10):
        candidates = decile_bins[d]
        n_select = min(20, len(candidates))
        if n_select > 0:
            chosen = random.sample(candidates, n_select)
            selected.extend(chosen)

    # Pad to 200 if needed (take extras from largest bins)
    while len(selected) < 200:
        for d in range(9, -1, -1):
            remaining = [r for r in decile_bins[d] if r not in selected]
            if remaining:
                extra = random.choice(remaining)
                selected.append(extra)
                if len(selected) >= 200:
                    break

    # Trim to exactly 200
    selected = selected[:200]
    print(f"\nSelected {len(selected)} files for conformance set")

    # Copy files and write file list
    file_list = []
    for i, row in enumerate(selected):
        src = row["filepath"]
        basename = f"conf_{i:03d}.wav"
        dst = os.path.join(files_dir, basename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        file_list.append({
            "conf_name": basename,
            "original": src,
            "vqi_s": row["vqi_s"],
            "vqi_v": row["vqi_v"],
        })

    # Write file list
    list_path = os.path.join(conf_dir, "conformance_file_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for item in file_list:
            f.write(f"{item['conf_name']}\t{item['original']}\t{item['vqi_s']}\t{item['vqi_v']}\n")
    print(f"Wrote {list_path}")

    # Summary by decile
    sel_deciles = {}
    for item in file_list:
        d = min(item["vqi_s"] // 10, 9)
        sel_deciles[d] = sel_deciles.get(d, 0) + 1
    print("\nSelected per decile:")
    for d in range(10):
        lo, hi = d * 10, (d + 1) * 10 if d < 9 else 100
        print(f"  {lo:>3}-{hi:>3}: {sel_deciles.get(d, 0)} files")

    print(f"\nTotal files copied to {files_dir}: {len(file_list)}")


if __name__ == "__main__":
    main()
