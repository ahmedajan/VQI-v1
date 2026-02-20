"""
Monitor Step 1.6a embedding extraction progress every 6 hours.
Updates all 4 blueprint docs: Phase_Step1, PROGRESS_LOG, MEMORY, VQI_COMPLETE_BLUEPRINT.
"""
import re
import time
import os
import sys
from datetime import datetime

EXTRACTION_OUTPUT = r"C:\Users\Ajan\AppData\Local\Temp\claude\D--VQI\tasks\b83f21b.output"
PHASE_STEP1 = r"D:\VQI\blueprint\Phase_Step1_Data_Collection.md"
PROGRESS_LOG = r"D:\VQI\blueprint\PROGRESS_LOG.md"
MEMORY_MD = r"C:\Users\Ajan\.claude\projects\D--VQI\memory\MEMORY.md"
COMPLETE_BLUEPRINT = r"D:\VQI\blueprint\VQI_COMPLETE_BLUEPRINT.md"

CHECK_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours


def get_latest_progress(output_file):
    """Read the last progress line from extraction output."""
    if not os.path.exists(output_file):
        return None
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # Find last line matching progress pattern
    pattern = re.compile(r"\[(\d+)/(\d+)\]\s+([\d.]+)%.*?(\d+\.\d+)\s+files/s.*?errors:\s+(\d+)")
    last_match = None
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            last_match = m
            break
    if not last_match:
        return None
    return {
        "done": int(last_match.group(1)),
        "total": int(last_match.group(2)),
        "pct": float(last_match.group(3)),
        "rate": float(last_match.group(4)),
        "errors": int(last_match.group(5)),
    }


def update_file(filepath, old_pattern, new_text):
    """Replace a regex pattern in a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    new_content, count = re.subn(old_pattern, new_text, content)
    if count > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def update_all_docs(p):
    """Update all 4 blueprint docs with current progress."""
    done_k = p["done"] // 1000
    pct = f"{p['pct']:.1f}"
    rate = f"{p['rate']:.2f}"
    errors = p["errors"]
    now = datetime.now().strftime("%Y-%m-%d")

    # 1. Phase_Step1_Data_Collection.md
    update_file(
        PHASE_STEP1,
        r"\*\*Status:\*\* \[ \] 1\.6a IN PROGRESS \([^)]+\) \| 1\.6b pending",
        f"**Status:** [ ] 1.6a IN PROGRESS ({p['done']:,}/1,210,451 -- {pct}%, ~{rate} f/s, {errors} errors, ETA ~Feb 13-14) | 1.6b pending",
    )

    # 2. PROGRESS_LOG.md
    update_file(
        PROGRESS_LOG,
        r"- \*\*Progress:\*\* ~[\d,]+ / 1,210,451 \([\d.]+%\)",
        f"- **Progress:** ~{p['done']:,} / 1,210,451 ({pct}%)",
    )
    update_file(
        PROGRESS_LOG,
        r"- \*\*Rate:\*\* ~[\d.]+ files/s \(stable\)",
        f"- **Rate:** ~{rate} files/s (stable)",
    )

    # 3. MEMORY.md
    update_file(
        MEMORY_MD,
        r"- Step 1\.6a: at \d+K/1\.21M \(~[\d.]+%\) as of [^,]+, ~[\d.]+ f/s, \d+ errors, ETA ~Feb 13-14",
        f"- Step 1.6a: at {done_k}K/1.21M (~{pct}%) as of {now}, ~{rate} f/s, {errors} errors, ETA ~Feb 13-14",
    )

    # 4. VQI_COMPLETE_BLUEPRINT.md
    update_file(
        COMPLETE_BLUEPRINT,
        r"\| 1\.6a Embedding extraction \| \*\*IN PROGRESS\*\* \|[^|]+\|",
        f"| 1.6a Embedding extraction | **IN PROGRESS** | {done_k}K/1.21M ({pct}%), ~{rate} f/s, {errors} errors, ETA ~Feb 13-14. Batching/concurrency attempted -- no significant speedup (models too small for GPU saturation, bottleneck is Python/IO). Reverted to original script. |",
    )


def check_completion(p):
    """Check if extraction is done."""
    return p["done"] >= p["total"]


def main():
    print(f"[monitor_1_6a] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[monitor_1_6a] Checking every {CHECK_INTERVAL_SECONDS // 3600} hours")
    print(f"[monitor_1_6a] Reading from: {EXTRACTION_OUTPUT}")
    print(f"[monitor_1_6a] Updating: Phase_Step1, PROGRESS_LOG, MEMORY, VQI_COMPLETE_BLUEPRINT")
    sys.stdout.flush()

    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)

        p = get_latest_progress(EXTRACTION_OUTPUT)
        if p is None:
            print(f"[monitor_1_6a] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- Could not read progress (file missing or no progress lines)")
            sys.stdout.flush()
            continue

        print(f"[monitor_1_6a] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- {p['done']:,}/{p['total']:,} ({p['pct']:.1f}%) | {p['rate']:.2f} f/s | errors: {p['errors']}")
        sys.stdout.flush()

        update_all_docs(p)
        print(f"[monitor_1_6a] Updated all 4 blueprint docs")
        sys.stdout.flush()

        if check_completion(p):
            print(f"[monitor_1_6a] EXTRACTION COMPLETE! {p['done']:,}/{p['total']:,}")
            print(f"[monitor_1_6a] Next: run Step 1.6b (compute_scores.py)")
            sys.stdout.flush()
            break

    print(f"[monitor_1_6a] Monitor stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
