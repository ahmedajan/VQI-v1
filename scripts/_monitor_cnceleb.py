"""
Monitor CN-Celeb background tasks (embeddings + features) every 1 hour.
Updates PROGRESS_LOG.md, Phase_Step8, MEMORY.md, and VQI_COMPLETE_BLUEPRINT.md.
"""
import re
import time
import os
import sys
from datetime import datetime

# Paths
EMB_OUTPUT = r"C:\Users\Ajan\AppData\Local\Temp\claude\D--VQI\tasks\b0e8a9b.output"
FEAT_OUTPUT = r"C:\Users\Ajan\AppData\Local\Temp\claude\D--VQI\tasks\bfc4db8.output"
PROGRESS_LOG = r"D:\VQI\blueprint\PROGRESS_LOG.md"
PHASE_STEP8 = r"D:\VQI\blueprint\Phase_Step8_Evaluation.md"
MEMORY_FILE = r"C:\Users\Ajan\.claude\projects\D--VQI\memory\MEMORY.md"
BLUEPRINT = r"D:\VQI\blueprint\VQI_COMPLETE_BLUEPRINT.md"

TOTAL_FILES = 126532
INTERVAL_SECONDS = 3600  # 1 hour


def parse_last_progress(filepath):
    """Parse the last progress line from a task output file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    # Find last progress line like [4000/126532] 3.2% | elapsed ...
    last = None
    for line in reversed(lines):
        m = re.search(r'\[(\d+)/(\d+)\]\s+([\d.]+)%\s*\|\s*elapsed\s+([\d:]+)\s*\|\s*eta\s+([\d:hm.]+)', line)
        if m:
            last = {
                "processed": int(m.group(1)),
                "total": int(m.group(2)),
                "pct": float(m.group(3)),
                "elapsed": m.group(4),
                "eta": m.group(5),
            }
            break
        # Features format: [9500/126532] 1.6 files/s, ETA 1205.3min
        m2 = re.search(r'\[(\d+)/(\d+)\]\s+([\d.]+)\s*files/s,\s*ETA\s+([\d.]+)min', line)
        if m2:
            processed = int(m2.group(1))
            total = int(m2.group(2))
            last = {
                "processed": processed,
                "total": total,
                "pct": round(100.0 * processed / total, 1),
                "rate": float(m2.group(3)),
                "eta_min": float(m2.group(4)),
            }
            break
    # Check for completion
    for line in reversed(lines[-20:]):
        if "COMPLETE" in line.upper() or "Done!" in line or "Successfully" in line.lower():
            if last:
                last["completed"] = True
            else:
                last = {"completed": True, "processed": TOTAL_FILES, "total": TOTAL_FILES, "pct": 100.0}
            break
    # Check for errors count
    for line in reversed(lines):
        m3 = re.search(r'err(?:ors?)?[=:]?\s*(\d+)', line)
        if m3:
            if last:
                last["errors"] = int(m3.group(1))
            break
    return last


def format_status(name, info):
    """Format a status string."""
    if info is None:
        return f"  {name}: No output yet"
    if info.get("completed"):
        return f"  {name}: COMPLETE ({info['total']} files)"
    pct = info.get("pct", 0)
    proc = info.get("processed", 0)
    total = info.get("total", TOTAL_FILES)
    parts = [f"{name}: {proc}/{total} ({pct}%)"]
    if "elapsed" in info:
        parts.append(f"elapsed {info['elapsed']}")
    if "eta" in info:
        parts.append(f"ETA {info['eta']}")
    if "eta_min" in info:
        hours = info["eta_min"] / 60
        parts.append(f"ETA {hours:.1f}h")
    if "rate" in info:
        parts.append(f"{info['rate']} files/s")
    if "errors" in info:
        parts.append(f"errors={info['errors']}")
    return "  " + " | ".join(parts)


def update_memory(emb_info, feat_info, timestamp):
    """Update the CN-Celeb line in MEMORY.md."""
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Build new CN-Celeb status line
    emb_status = "COMPLETE" if (emb_info and emb_info.get("completed")) else (
        f"{emb_info['pct']}%" if emb_info else "unknown"
    )
    feat_status = "COMPLETE" if (feat_info and feat_info.get("completed")) else (
        f"{feat_info['pct']}%" if feat_info else "unknown"
    )

    if emb_info and emb_info.get("completed") and feat_info and feat_info.get("completed"):
        new_line = f"  - CN-Celeb: COMPLETE (126,532 files, embeddings + features done)"
    else:
        emb_detail = ""
        feat_detail = ""
        if emb_info and not emb_info.get("completed"):
            rate = ""
            if "eta" in emb_info:
                rate = f", ETA {emb_info['eta']}"
            emb_detail = f"embeddings {emb_status}{rate}"
        elif emb_info and emb_info.get("completed"):
            emb_detail = "embeddings DONE"
        if feat_info and not feat_info.get("completed"):
            eta = ""
            if "eta_min" in feat_info:
                eta = f", ETA {feat_info['eta_min']/60:.1f}h"
            feat_detail = f"features {feat_status}{eta}"
        elif feat_info and feat_info.get("completed"):
            feat_detail = "features DONE"
        parts = [p for p in [emb_detail, feat_detail] if p]
        new_line = f"  - CN-Celeb: IN PROGRESS ({' + '.join(parts)}, 126,532 files) [updated {timestamp}]"

    # Replace existing CN-Celeb line
    pattern = r"  - CN-Celeb:.*(?:\n    -.*)*"
    if re.search(pattern, content):
        content = re.sub(pattern, new_line, content)
    else:
        # Fallback: add after VCTK line
        content = content.replace(
            "  - 2 multi-dataset plots pending",
            new_line + "\n  - 2 multi-dataset plots pending"
        )

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def update_progress_log(emb_info, feat_info, timestamp):
    """Append a checkpoint entry to PROGRESS_LOG.md."""
    if not os.path.exists(PROGRESS_LOG):
        return
    with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
        content = f.read()

    emb_pct = emb_info.get("pct", 0) if emb_info else 0
    feat_pct = feat_info.get("pct", 0) if feat_info else 0
    emb_done = emb_info and emb_info.get("completed", False)
    feat_done = feat_info and feat_info.get("completed", False)

    entry = f"\n### CN-Celeb Progress Check ({timestamp})\n"
    entry += f"- Embeddings: {'COMPLETE' if emb_done else f'{emb_pct}%'}"
    if emb_info and "errors" in emb_info:
        entry += f" (errors: {emb_info['errors']})"
    entry += "\n"
    entry += f"- Features: {'COMPLETE' if feat_done else f'{feat_pct}%'}"
    if feat_info and "errors" in feat_info:
        entry += f" (errors: {feat_info['errors']})"
    entry += "\n"

    if emb_done and feat_done:
        entry += "- **Both tasks COMPLETE -- ready for CN-Celeb evaluation pipeline**\n"

    content += entry
    with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    check_num = 0
    print(f"CN-Celeb monitor started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking every {INTERVAL_SECONDS}s ({INTERVAL_SECONDS/3600:.1f}h)")
    print(f"Embeddings output: {EMB_OUTPUT}")
    print(f"Features output: {FEAT_OUTPUT}")
    print()

    while True:
        check_num += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"=== Check #{check_num} at {timestamp} ===")

        emb_info = parse_last_progress(EMB_OUTPUT)
        feat_info = parse_last_progress(FEAT_OUTPUT)

        print(format_status("Embeddings", emb_info))
        print(format_status("Features", feat_info))

        # Update documents
        try:
            update_memory(emb_info, feat_info, timestamp)
            print("  Updated MEMORY.md")
        except Exception as e:
            print(f"  MEMORY.md update failed: {e}")

        try:
            update_progress_log(emb_info, feat_info, timestamp)
            print("  Updated PROGRESS_LOG.md")
        except Exception as e:
            print(f"  PROGRESS_LOG.md update failed: {e}")

        # Check if both complete
        emb_done = emb_info and emb_info.get("completed", False)
        feat_done = feat_info and feat_info.get("completed", False)
        if emb_done and feat_done:
            print("\n*** BOTH TASKS COMPLETE ***")
            print("CN-Celeb embeddings and features are ready.")
            print("Next: Run CN-Celeb evaluation pipeline.")
            break

        print(f"  Next check in {INTERVAL_SECONDS/3600:.1f}h")
        print()
        sys.stdout.flush()
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
