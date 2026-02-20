"""Step 8.12: Generate VQI-V + dual-score evaluation report.

Compiles VQI-V ERC, DET, combined ERC, and quadrant analysis results
into a comprehensive markdown report.

Usage:
    python scripts/generate_eval_report_v.py --dataset voxceleb1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
REPORTS_DIR = os.path.join(_IMPL_DIR, "reports")

ALL_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]
PROVIDER_SHORT = {
    "P1_ECAPA": "ECAPA-TDNN",
    "P2_RESNET": "ResNetSE34V2",
    "P3_ECAPA2": "ECAPA2",
    "P4_XVECTOR": "x-vector",
    "P5_WAVLM": "WavLM-SV",
}


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate VQI-V evaluation report")
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "vctk", "cnceleb"])
    args = parser.parse_args()

    dataset = args.dataset
    eval_base = os.path.join(DATA_DIR, "step8_eval", dataset)

    eval_v = load_json(os.path.join(eval_base, "vqi_v", f"vqi_v_evaluation_{dataset}.json"))
    cross_v = load_json(os.path.join(eval_base, "cross_system", f"cross_system_vqi_v_{dataset}.json"))
    combined = load_json(os.path.join(eval_base, "dual_score", f"combined_erc_{dataset}.json"))
    quadrant = load_json(os.path.join(eval_base, "dual_score", f"quadrant_analysis_{dataset}.json"))

    report_dir = os.path.join(REPORTS_DIR, "evaluation_v")
    os.makedirs(report_dir, exist_ok=True)

    lines = [
        f"# VQI-V + Dual-Score Evaluation Report: {dataset}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Dataset:** {dataset}",
        "",
        "---",
        "",
        "## 1. VQI-V Error vs. Reject Curves",
        "",
        "| Provider | Reject 10% | Reject 20% | Reject 30% |",
        "|----------|------------|------------|------------|",
    ]

    for pn in ALL_PROVIDERS:
        if pn not in eval_v:
            continue
        reds = eval_v[pn].get("fnmr_10pct", {}).get("reductions", {})
        r10 = reds.get("0.1", {}).get("fnmr_reduction_pct", 0)
        r20 = reds.get("0.2", {}).get("fnmr_reduction_pct", 0)
        r30 = reds.get("0.3", {}).get("fnmr_reduction_pct", 0)
        lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | {r10:.1f}% | {r20:.1f}% | {r30:.1f}% |")

    # VQI-V DET
    lines.extend(["", "## 2. VQI-V Ranked DET", "",
                   "| Provider | Bottom EER | Top EER | Separation |",
                   "|----------|-----------|---------|------------|"])
    for pn in ALL_PROVIDERS:
        if pn not in eval_v:
            continue
        det = eval_v[pn].get("ranked_det", {})
        eer_b = det.get("bottom", {}).get("eer")
        eer_t = det.get("top", {}).get("eer")
        sep = det.get("eer_separation")
        if all(x is not None for x in [eer_b, eer_t, sep]):
            lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | {eer_b:.4f} | {eer_t:.4f} | {sep:.2f}x |")

    # Combined ERC
    if combined:
        lines.extend(["", "## 3. Combined ERC (Dual-Score Rejection)", "",
                       "FNMR reduction at 20% rejection by strategy:", "",
                       "| Provider | S-only | V-only | Union | Intersection |",
                       "|----------|--------|--------|-------|--------------|"])
        for pn in ALL_PROVIDERS:
            if pn not in combined:
                continue
            summary = combined[pn].get("summary", {})
            vals = []
            for strat in ["s_only", "v_only", "union", "intersection"]:
                red = summary.get(strat, {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
                vals.append(f"{red:.1f}%")
            lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | {' | '.join(vals)} |")

    # Quadrant analysis
    if quadrant:
        lines.extend(["", "## 4. Per-Quadrant Performance", "",
                       "| Provider | Q1 EER | Q2 EER | Q3 EER | Q4 EER | Q1 < Q3 |",
                       "|----------|--------|--------|--------|--------|---------|"])
        for pn in ALL_PROVIDERS:
            if pn not in quadrant:
                continue
            eers = []
            for qn in ["Q1", "Q2", "Q3", "Q4"]:
                eer = quadrant[pn].get(qn, {}).get("eer")
                eers.append(f"{eer:.4f}" if eer is not None else "N/A")
            q1_lt_q3 = quadrant[pn].get("q1_eer_lt_q3_eer")
            lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | {' | '.join(eers)} | "
                        f"{'Yes' if q1_lt_q3 else 'No'} |")

        lines.extend([
            "",
            "### Quadrant Interpretation",
            "- Q1 (high S, high V): Best quality -> lowest EER",
            "- Q2 (low S, high V): Signal issues, voice OK -> moderate EER",
            "- Q3 (low S, low V): Worst quality -> highest EER",
            "- Q4 (high S, low V): Signal OK, voice indistinct -> moderate EER",
        ])

    # Cross-system V
    if cross_v:
        verdict = cross_v.get("_verdict", {})
        lines.extend([
            "",
            "## 5. VQI-V Cross-System Generalization",
            f"",
            f"**Verdict: {'PASS' if verdict.get('passed') else 'FAIL'}**",
            f"- Mean test reduction@20%: {verdict.get('mean_test_reduction_20pct', 0):.1f}%",
        ])

    lines.extend(["", "---", "", "*Report generated by VQI Step 8 evaluation pipeline.*"])

    report_path = os.path.join(report_dir, f"evaluation_report_v_{dataset}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
