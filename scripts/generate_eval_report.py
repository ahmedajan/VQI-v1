"""Step 8.6: Generate VQI-S evaluation report.

Compiles ERC, DET, cross-system, and benchmark results into a
comprehensive markdown report.

Usage:
    python scripts/generate_eval_report.py --dataset voxceleb1
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
    parser = argparse.ArgumentParser(description="Generate VQI-S evaluation report")
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "vctk", "cnceleb", "vpqad", "vseadc"])
    args = parser.parse_args()

    dataset = args.dataset
    eval_base = os.path.join(DATA_DIR, "step8", "full_feature", "step8_eval", dataset)

    eval_s = load_json(os.path.join(eval_base, "vqi_s", f"vqi_s_evaluation_{dataset}.json"))
    cross_s = load_json(os.path.join(eval_base, "cross_system", f"cross_system_vqi_s_{dataset}.json"))
    bench = load_json(os.path.join(DATA_DIR, "step8", "full_feature", "test_scores", "benchmark_results.json"))

    report_dir = os.path.join(REPORTS_DIR, "step8", "full_feature")
    os.makedirs(report_dir, exist_ok=True)

    lines = [
        f"# VQI-S Evaluation Report: {dataset}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Dataset:** {dataset}",
        f"**Providers:** {', '.join(PROVIDER_SHORT.get(p, p) for p in ALL_PROVIDERS)}",
        "",
        "---",
        "",
        "## 1. Error vs. Reject Curves (ERC)",
        "",
        "ERC measures how much FNMR decreases when low-quality pairs are rejected.",
        "Quality metric: min(VQI-S score of sample 1, VQI-S score of sample 2).",
        "",
        "### FNMR Reduction Summary",
        "",
        "| Provider | Init FNMR | Reject 10% | Reject 20% | Reject 30% |",
        "|----------|-----------|------------|------------|------------|",
    ]

    for pn in ALL_PROVIDERS:
        if pn not in eval_s:
            continue
        for fnmr_key, init_label in [("fnmr_10pct", "10%")]:
            reds = eval_s[pn].get(fnmr_key, {}).get("reductions", {})
            r10 = reds.get("0.1", {}).get("fnmr_reduction_pct", 0)
            r20 = reds.get("0.2", {}).get("fnmr_reduction_pct", 0)
            r30 = reds.get("0.3", {}).get("fnmr_reduction_pct", 0)
            lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | {init_label} | "
                        f"{r10:.1f}% | {r20:.1f}% | {r30:.1f}% |")

    lines.extend(["", "### Target Check", ""])
    target_pass = True
    for pn in ALL_PROVIDERS:
        if pn not in eval_s:
            continue
        reds = eval_s[pn].get("fnmr_10pct", {}).get("reductions", {})
        r20 = reds.get("0.2", {}).get("fnmr_reduction_pct", 0)
        status = "PASS" if r20 > 50 else "FAIL"
        if r20 <= 50:
            target_pass = False
        lines.append(f"- {PROVIDER_SHORT.get(pn, pn)}: {r20:.1f}% reduction at 20% reject "
                    f"(target: >50%) [{status}]")

    lines.extend([
        "",
        "## 2. Ranked DET Curves",
        "",
        "| Provider | Bottom 15% EER | Middle 70% EER | Top 15% EER | Separation |",
        "|----------|---------------|----------------|-------------|------------|",
    ])

    for pn in ALL_PROVIDERS:
        if pn not in eval_s:
            continue
        det = eval_s[pn].get("ranked_det", {})
        eer_b = det.get("bottom", {}).get("eer")
        eer_m = det.get("middle", {}).get("eer")
        eer_t = det.get("top", {}).get("eer")
        sep = det.get("eer_separation")
        lines.append(f"| {PROVIDER_SHORT.get(pn, pn)} | "
                    f"{eer_b:.4f} | {eer_m:.4f} | {eer_t:.4f} | {sep:.2f}x |"
                    if all(x is not None for x in [eer_b, eer_m, eer_t, sep])
                    else f"| {PROVIDER_SHORT.get(pn, pn)} | N/A | N/A | N/A | N/A |")

    # Cross-system
    if cross_s:
        verdict = cross_s.get("_verdict", {})
        lines.extend([
            "",
            "## 3. Cross-System Generalization",
            "",
            f"**Verdict: {'PASS' if verdict.get('passed') else 'FAIL'}**",
            "",
            f"- All test providers monotonic: {verdict.get('all_monotonic')}",
            f"- Mean train FNMR reduction@20%: {verdict.get('mean_train_reduction_20pct', 0):.1f}%",
            f"- Mean test FNMR reduction@20%: {verdict.get('mean_test_reduction_20pct', 0):.1f}%",
        ])

    # Benchmarks
    if bench:
        lines.extend([
            "",
            "## 4. Speed Benchmarks",
            "",
            "| Duration | Mean (ms) | Std (ms) | Target (ms) | Status |",
            "|----------|-----------|----------|-------------|--------|",
        ])
        for dur_key in ["3s", "10s", "60s"]:
            if dur_key in bench:
                r = bench[dur_key]
                status = "PASS" if r.get("passed") else "FAIL"
                lines.append(f"| {dur_key} | {r['total']['mean_ms']:.1f} | "
                            f"{r['total']['std_ms']:.1f} | <{r['target_ms']} | {status} |")

    lines.extend(["", "---", "", "*Report generated by VQI Step 8 evaluation pipeline.*"])

    report_path = os.path.join(report_dir, f"evaluation_report_{dataset}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
