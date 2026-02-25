"""Generate expected conformance output for VQI v2.0 (PCA-90% models).

Runs VQIEngine v2.0 on all 200 conformance files.

Outputs:
  conformance/conformance_expected_output_v2.0.csv
  conformance/conformance_expected_output_v2.0_features.csv

Supports --resume via checkpoint file.
"""

import csv
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    conf_dir = os.path.join(BASE_DIR, "conformance")
    files_dir = os.path.join(conf_dir, "test_files")
    checkpoint_path = os.path.join(conf_dir, "_checkpoint_v2.json")

    output_csv = os.path.join(conf_dir, "conformance_expected_output_v2.0.csv")
    output_features_csv = os.path.join(conf_dir, "conformance_expected_output_v2.0_features.csv")

    # Load file list
    file_list_path = os.path.join(conf_dir, "conformance_file_list.txt")
    files = []
    with open(file_list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                files.append({"conf_name": parts[0], "original": parts[1]})

    print(f"Conformance files: {len(files)}")

    # Check for resume
    results = []
    done_set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        results = checkpoint.get("results", [])
        done_set = {r["filename"] for r in results}
        print(f"Resuming from checkpoint: {len(done_set)} done")

    # Load engine (v2.0 with PCA-90% models)
    from vqi.engine import VQIEngine
    print("Loading VQI Engine v2.0 (PCA-90%)...")
    engine = VQIEngine()
    print("Engine loaded.")

    total = len(files)
    t0 = time.time()

    for i, finfo in enumerate(files):
        conf_name = finfo["conf_name"]
        if conf_name in done_set:
            continue

        filepath = os.path.join(files_dir, conf_name)
        if not os.path.exists(filepath):
            print(f"  SKIP {conf_name}: file not found")
            continue

        try:
            result = engine.score_file(filepath)
            entry = {
                "filename": conf_name,
                "vqi_s": result.score_s,
                "vqi_v": result.score_v,
                "categories_s": ";".join(sorted(result.category_scores_s.keys())),
                "categories_v": ";".join(sorted(result.category_scores_v.keys())),
                "processing_time_ms": result.processing_time_ms,
                "features_s": {k: float(v) for k, v in result.features_s.items()},
                "features_v": {k: float(v) for k, v in result.features_v.items()},
            }
            results.append(entry)
            done_set.add(conf_name)

            elapsed = time.time() - t0
            rate = len(done_set) / elapsed if elapsed > 0 else 0
            print(f"  [{len(done_set)}/{total}] {conf_name}: S={result.score_s}, V={result.score_v} ({rate:.1f} files/s)")

            # Checkpoint every 20 files
            if len(results) % 20 == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"results": results}, f)

        except Exception as e:
            print(f"  ERROR {conf_name}: {e}")

    # Write outputs
    print(f"\nWriting {len(results)} results...")

    # Main CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "vqi_s", "vqi_v", "categories_s", "categories_v",
                         "processing_time_ms"])
        for r in results:
            writer.writerow([r["filename"], r["vqi_s"], r["vqi_v"],
                             r["categories_s"], r["categories_v"],
                             f"{r['processing_time_ms']:.1f}"])
    print(f"Wrote {output_csv}")

    # Features CSV
    if results:
        s_keys = sorted(results[0]["features_s"].keys())
        v_keys = sorted(results[0]["features_v"].keys())
        with open(output_features_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["filename"] + [f"S_{k}" for k in s_keys] + [f"V_{k}" for k in v_keys]
            writer.writerow(header)
            for r in results:
                row = [r["filename"]]
                row += [r["features_s"].get(k, 0.0) for k in s_keys]
                row += [r["features_v"].get(k, 0.0) for k in v_keys]
                writer.writerow(row)
        print(f"Wrote {output_features_csv}")

    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed.")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(results)} files)")


if __name__ == "__main__":
    main()
