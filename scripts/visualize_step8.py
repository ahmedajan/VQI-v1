"""Step 8: Visualization — 25 outputs per blueprint M.8.

VQI-S plots (1-15):
  1.  erc_fnmr1_all_providers.png
  2.  erc_fnmr10_all_providers.png
  3.  erc_by_dataset.png
  4.  ranked_det_all_providers.png
  5.  cross_system_erc.png
  6.  cross_system_ranked_det.png
  7.  erc_summary_table.png
  8.  det_separation_table.png
  9.  timing_benchmark.png
  10. erc_ridgeline_across_datasets.png
  11. provider_performance_radar.png
  12. fnmr_reduction_forest_plot.png
  13. erc_parallel_coordinates.png
  14. det_separation_strip.png
  15. analysis.md

VQI-V + Dual-score plots (16-25):
  16. erc_v_all_providers.png
  17. ranked_det_v_all_providers.png
  18. cross_system_v_erc.png
  19. combined_erc_comparison.png
  20. combined_erc_summary_table.png
  21. quadrant_eer_comparison.png
  22. quadrant_score_distributions.png
  23. quadrant_performance_table.png
  24. dual_score_value_assessment.png
  25. analysis_v.md

Usage:
    python scripts/visualize_step8.py --dataset voxceleb1
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(_IMPL_DIR, "data")
REPORTS_DIR = os.path.join(_IMPL_DIR, "reports")

ALL_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]
PROVIDER_COLORS = {
    "P1_ECAPA": "#1f77b4",
    "P2_RESNET": "#ff7f0e",
    "P3_ECAPA2": "#2ca02c",
    "P4_XVECTOR": "#d62728",
    "P5_WAVLM": "#9467bd",
}
PROVIDER_SHORT = {
    "P1_ECAPA": "ECAPA-TDNN",
    "P2_RESNET": "ResNetSE34V2",
    "P3_ECAPA2": "ECAPA2",
    "P4_XVECTOR": "x-vector",
    "P5_WAVLM": "WavLM-SV",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ============================================================
# VQI-S Plots (1-14)
# ============================================================

def plot_erc_all_providers(eval_data, fnmr_key, output_path, title_suffix):
    """Plot 1 or 2: ERC at FNMR=1% or 10% for all providers."""
    fig, axes = plt.subplots(1, len(eval_data), figsize=(4 * len(eval_data), 4), squeeze=False)

    for i, (pn, pdata) in enumerate(eval_data.items()):
        if pn.startswith("_"):
            continue
        ax = axes[0, i]
        if fnmr_key not in pdata:
            continue

        erc = pdata[fnmr_key]
        rf = np.array(erc["erc_reject_fracs"])
        fnmr = np.array(erc["erc_fnmr"])
        random = np.array(erc["random_baseline"]) if "random_baseline" in erc else None

        ax.plot(rf, fnmr, color=PROVIDER_COLORS.get(pn, "blue"), linewidth=2, label="VQI-S")
        if random is not None:
            ax.plot(rf, random, "--", color="gray", linewidth=1, label="Random")

        ax.set_xlabel("Rejection Fraction")
        ax.set_ylabel("FNMR")
        ax.set_title(PROVIDER_SHORT.get(pn, pn))
        ax.set_xlim(0, 0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"ERC Curves ({title_suffix})", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_ranked_det_all_providers(eval_data, dataset, data_dir, prefix="", save_dir=None):
    """Plot 4 or 17: Ranked DET for all providers."""
    providers = [p for p in eval_data if not p.startswith("_")]
    fig, axes = plt.subplots(1, len(providers), figsize=(4 * len(providers), 4), squeeze=False)

    group_colors = {"bottom": "#d62728", "middle": "#1f77b4", "top": "#2ca02c"}

    for i, pn in enumerate(providers):
        ax = axes[0, i]
        det_info = eval_data[pn].get("ranked_det", {})

        for gname in ["bottom", "middle", "top"]:
            det_file = os.path.join(data_dir, f"det{prefix}_{dataset}_{pn}_{gname}.npz")
            if os.path.exists(det_file):
                data = np.load(det_file)
                fmr = data["fmr"]
                fnmr = data["fnmr"]
                # Filter out zeros for log scale
                mask = (fmr > 0) & (fnmr > 0)
                if mask.sum() > 10:
                    ax.plot(fmr[mask], fnmr[mask], color=group_colors[gname],
                            linewidth=1.5, label=gname.capitalize())

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FMR")
        ax.set_ylabel("FNMR")
        ax.set_title(PROVIDER_SHORT.get(pn, pn))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)

    fig.suptitle(f"Ranked DET Curves {'(VQI-V) ' if prefix else ''}-- {dataset}", fontsize=14, y=1.02)
    fig.tight_layout()
    out_dir = save_dir or data_dir
    output_path = os.path.join(out_dir, f"ranked_det{prefix}_all_providers.png")
    save_fig(fig, output_path)


def plot_cross_system_erc(cross_data, output_path, score_label="S"):
    """Plot 5 or 18: Cross-system ERC for P4/P5."""
    test_providers = ["P4_XVECTOR", "P5_WAVLM"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, pn in enumerate(test_providers):
        ax = axes[i]
        if pn not in cross_data:
            continue
        pdata = cross_data[pn]
        for fnmr_key in ["fnmr_10pct"]:
            erc_info = pdata.get("erc", {}).get(fnmr_key, {})
            reductions = erc_info.get("reductions", {})
            red_20 = reductions.get("0.2", {}).get("fnmr_reduction_pct", 0)
            mono = erc_info.get("monotonic", "?")
            ax.text(0.5, 0.9, f"FNMR red@20%: {red_20:.1f}%\nMonotonic: {mono}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(facecolor="lightyellow", alpha=0.8))

        ax.set_title(f"{PROVIDER_SHORT.get(pn, pn)} (unseen)")
        ax.set_xlabel("Rejection Fraction")
        ax.set_ylabel("FNMR")
        ax.grid(True, alpha=0.3)

    verdict = cross_data.get("_verdict", {})
    passed = verdict.get("passed", "?")
    fig.suptitle(f"Cross-System Generalization (VQI-{score_label}) — {'PASS' if passed else 'FAIL'}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_erc_summary_table(eval_data, output_path):
    """Plot 7: FNMR reduction summary table as image."""
    rows = []
    for pn in ALL_PROVIDERS:
        if pn not in eval_data:
            continue
        pdata = eval_data[pn]
        for fnmr_key, label in [("fnmr_1pct", "1%"), ("fnmr_10pct", "10%")]:
            if fnmr_key not in pdata:
                continue
            reductions = pdata[fnmr_key].get("reductions", {})
            for rf_key in ["0.1", "0.2", "0.3"]:
                red = reductions.get(rf_key, {}).get("fnmr_reduction_pct", 0)
                rows.append((PROVIDER_SHORT.get(pn, pn), label, f"{float(rf_key)*100:.0f}%", f"{red:.1f}%"))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.3 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Provider", "Init FNMR", "Reject %", "FNMR Reduction"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    fig.suptitle("ERC FNMR Reduction Summary (VQI-S)", fontsize=13)
    save_fig(fig, output_path)


def plot_det_separation_table(eval_data, output_path):
    """Plot 8: DET EER separation table."""
    rows = []
    for pn in ALL_PROVIDERS:
        if pn not in eval_data:
            continue
        det = eval_data[pn].get("ranked_det", {})
        for gname in ["bottom", "middle", "top"]:
            eer = det.get(gname, {}).get("eer")
            if eer is not None:
                rows.append((PROVIDER_SHORT.get(pn, pn), gname.capitalize(), f"{eer:.4f}"))
        sep = det.get("eer_separation")
        if sep is not None:
            rows.append((PROVIDER_SHORT.get(pn, pn), "Separation", f"{sep:.2f}x"))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.25 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Provider", "Group", "EER"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    fig.suptitle("Ranked DET EER by Quality Group (VQI-S)", fontsize=13)
    save_fig(fig, output_path)


def plot_timing_benchmark(output_path):
    """Plot 9: Speed benchmark stacked bar chart."""
    bench_path = os.path.join(DATA_DIR, "test_scores", "benchmark_results.json")
    if not os.path.exists(bench_path):
        logger.warning("  Benchmark results not found, skipping timing plot")
        return

    bench = load_json(bench_path)

    durations = []
    components = ["preprocess", "vad", "frame_features", "vqi_v_features", "rf_predict"]
    values = {c: [] for c in components}
    totals = []
    targets = []

    for dur_key in ["3s", "10s", "60s"]:
        if dur_key not in bench:
            continue
        durations.append(dur_key)
        totals.append(bench[dur_key]["total"]["mean_ms"])
        targets.append(bench[dur_key]["target_ms"])
        for c in components:
            values[c].append(bench[dur_key].get(c, {}).get("mean_ms", 0))

    if not durations:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(durations))
    width = 0.5

    bottom = np.zeros(len(durations))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    for i, c in enumerate(components):
        ax.bar(x, values[c], width, bottom=bottom, label=c, color=colors[i % len(colors)])
        bottom += np.array(values[c])

    # Target line
    for i, (t, d) in enumerate(zip(targets, durations)):
        ax.plot([i - 0.3, i + 0.3], [t, t], "r--", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(durations)
    ax.set_xlabel("Audio Duration")
    ax.set_ylabel("Time (ms)")
    ax.set_title("VQI Scoring Speed Benchmark")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, output_path)


def plot_combined_erc_comparison(combined_data, output_path):
    """Plot 19: Combined ERC comparison — 4 strategies overlaid."""
    providers = [p for p in combined_data if not p.startswith("_")]
    fig, axes = plt.subplots(1, len(providers), figsize=(4 * len(providers), 4), squeeze=False)

    strategy_colors = {
        "s_only": "#1f77b4",
        "v_only": "#ff7f0e",
        "union": "#2ca02c",
        "intersection": "#d62728",
    }
    strategy_labels = {
        "s_only": "VQI-S only",
        "v_only": "VQI-V only",
        "union": "Union (either)",
        "intersection": "Intersection (both)",
    }

    for i, pn in enumerate(providers):
        ax = axes[0, i]
        strategies = combined_data[pn].get("strategies", {})
        for strat, sdata in strategies.items():
            rf = np.array(sdata["reject_fracs"])
            fnmr = np.array(sdata["fnmr_values"])
            ax.plot(rf, fnmr, color=strategy_colors.get(strat, "gray"),
                    linewidth=1.5, label=strategy_labels.get(strat, strat))

        ax.set_xlabel("Rejection Fraction")
        ax.set_ylabel("FNMR")
        ax.set_title(PROVIDER_SHORT.get(pn, pn))
        ax.set_xlim(0, 0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Combined ERC Comparison — Dual-Score Rejection Strategies", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_quadrant_eer_comparison(quadrant_data, output_path):
    """Plot 21: Per-quadrant EER bar chart."""
    providers = [p for p in quadrant_data if not p.startswith("_")]
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(providers))
    width = 0.2
    quad_colors = {"Q1": "#2ca02c", "Q2": "#ff7f0e", "Q3": "#d62728", "Q4": "#9467bd"}

    for j, qname in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        eers = []
        for pn in providers:
            eer = quadrant_data[pn].get(qname, {}).get("eer")
            eers.append(eer if eer is not None else 0)
        ax.bar(x + j * width, eers, width, label=qname, color=quad_colors[qname])

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([PROVIDER_SHORT.get(p, p) for p in providers], rotation=15)
    ax.set_ylabel("EER")
    ax.set_title("Per-Quadrant EER Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, output_path)


def plot_quadrant_performance_table(quadrant_data, output_path):
    """Plot 23: Per-quadrant metrics table."""
    rows = []
    for pn in ALL_PROVIDERS:
        if pn not in quadrant_data:
            continue
        for qname in ["Q1", "Q2", "Q3", "Q4"]:
            qd = quadrant_data[pn].get(qname, {})
            eer = qd.get("eer")
            fnmr_001 = qd.get("fnmr_at_fmr_001")
            n_gen = qd.get("n_genuine", 0)
            n_imp = qd.get("n_impostor", 0)
            rows.append((
                PROVIDER_SHORT.get(pn, pn), qname,
                f"{eer:.4f}" if eer else "N/A",
                f"{fnmr_001:.4f}" if fnmr_001 else "N/A",
                str(n_gen), str(n_imp),
            ))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.25 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Provider", "Quadrant", "EER", "FNMR@FMR=1%", "N_gen", "N_imp"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    fig.suptitle("Per-Quadrant Performance Table", fontsize=13)
    save_fig(fig, output_path)


def plot_dual_score_value_assessment(combined_data, quadrant_data, output_path):
    """Plot 24: VQI-S alone vs dual assessment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: FNMR reduction comparison at 20% reject
    ax = axes[0]
    providers = [p for p in combined_data if not p.startswith("_")]
    strats = ["s_only", "v_only", "union", "intersection"]
    x = np.arange(len(providers))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for j, strat in enumerate(strats):
        reds = []
        for pn in providers:
            summary = combined_data[pn].get("summary", {})
            red = summary.get(strat, {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
            reds.append(red)
        ax.bar(x + j * width, reds, width, label=strat.replace("_", " ").title(), color=colors[j])

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([PROVIDER_SHORT.get(p, p) for p in providers], rotation=15)
    ax.set_ylabel("FNMR Reduction (%)")
    ax.set_title("FNMR Reduction at 20% Rejection")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Q1 vs Q3 EER comparison
    ax = axes[1]
    providers_q = [p for p in quadrant_data if not p.startswith("_")]
    q1_eers = [quadrant_data[p].get("Q1", {}).get("eer", 0) or 0 for p in providers_q]
    q3_eers = [quadrant_data[p].get("Q3", {}).get("eer", 0) or 0 for p in providers_q]

    x = np.arange(len(providers_q))
    ax.bar(x - 0.15, q1_eers, 0.3, label="Q1 (high S, high V)", color="#2ca02c")
    ax.bar(x + 0.15, q3_eers, 0.3, label="Q3 (low S, low V)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([PROVIDER_SHORT.get(p, p) for p in providers_q], rotation=15)
    ax.set_ylabel("EER")
    ax.set_title("Q1 vs Q3 EER")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Dual-Score Value Assessment", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def write_analysis_s(eval_data, cross_data, bench_data, dataset, output_path):
    """Write analysis.md for VQI-S (output 15)."""
    lines = [
        f"# Step 8: VQI-S Evaluation Analysis - {dataset}",
        "",
        "## ERC Results",
        "",
    ]

    for pn in ALL_PROVIDERS:
        if pn not in eval_data:
            continue
        lines.append(f"### {PROVIDER_SHORT.get(pn, pn)}")
        for fnmr_key, label in [("fnmr_1pct", "FNMR=1%"), ("fnmr_10pct", "FNMR=10%")]:
            if fnmr_key not in eval_data[pn]:
                continue
            reductions = eval_data[pn][fnmr_key].get("reductions", {})
            lines.append(f"\n**{label}:**")
            for rf_key in ["0.1", "0.2", "0.3"]:
                red = reductions.get(rf_key, {})
                lines.append(f"- Reject {float(rf_key)*100:.0f}%: FNMR reduction = "
                            f"{red.get('fnmr_reduction_pct', 0):.1f}%")

        det = eval_data[pn].get("ranked_det", {})
        lines.append(f"\n**Ranked DET:**")
        for gname in ["bottom", "middle", "top"]:
            eer = det.get(gname, {}).get("eer")
            if eer is not None:
                lines.append(f"- {gname}: EER = {eer:.4f}")
        sep = det.get("eer_separation")
        if sep:
            lines.append(f"- EER separation (bottom/top) = {sep:.2f}x")
        lines.append("")

    # Cross-system
    if cross_data:
        lines.append("## Cross-System Generalization")
        verdict = cross_data.get("_verdict", {})
        lines.append(f"\n- **Verdict:** {'PASS' if verdict.get('passed') else 'FAIL'}")
        lines.append(f"- All monotonic: {verdict.get('all_monotonic')}")
        lines.append(f"- Mean train reduction@20%: {verdict.get('mean_train_reduction_20pct', 0):.1f}%")
        lines.append(f"- Mean test reduction@20%: {verdict.get('mean_test_reduction_20pct', 0):.1f}%")
        lines.append("")

    # Benchmarks
    if bench_data:
        lines.append("## Speed Benchmarks")
        for dur_key in ["3s", "10s", "60s"]:
            if dur_key in bench_data:
                r = bench_data[dur_key]
                status = "PASS" if r.get("passed") else "FAIL"
                lines.append(f"- {dur_key}: {r['total']['mean_ms']:.1f}ms "
                            f"(target: <{r['target_ms']}ms) [{status}]")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {output_path}")


def write_analysis_v(eval_v_data, combined_data, quadrant_data, dataset, output_path):
    """Write analysis_v.md for VQI-V + dual-score (output 25)."""
    lines = [
        f"# Step 8: VQI-V + Dual-Score Evaluation Analysis - {dataset}",
        "",
        "## VQI-V ERC Results",
        "",
    ]

    for pn in ALL_PROVIDERS:
        if pn not in eval_v_data:
            continue
        lines.append(f"### {PROVIDER_SHORT.get(pn, pn)}")
        for fnmr_key, label in [("fnmr_10pct", "FNMR=10%")]:
            if fnmr_key not in eval_v_data[pn]:
                continue
            reductions = eval_v_data[pn][fnmr_key].get("reductions", {})
            lines.append(f"\n**{label}:**")
            for rf_key in ["0.1", "0.2", "0.3"]:
                red = reductions.get(rf_key, {})
                lines.append(f"- Reject {float(rf_key)*100:.0f}%: FNMR reduction = "
                            f"{red.get('fnmr_reduction_pct', 0):.1f}%")
        lines.append("")

    # Combined ERC
    lines.append("## Combined ERC (Dual-Score)")
    for pn in ALL_PROVIDERS:
        if pn not in combined_data:
            continue
        lines.append(f"\n### {PROVIDER_SHORT.get(pn, pn)}")
        summary = combined_data[pn].get("summary", {})
        for strat in ["s_only", "v_only", "union", "intersection"]:
            red = summary.get(strat, {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
            lines.append(f"- {strat}: FNMR reduction@20% = {red:.1f}%")
    lines.append("")

    # Quadrant analysis
    lines.append("## Quadrant Analysis")
    for pn in ALL_PROVIDERS:
        if pn not in quadrant_data:
            continue
        lines.append(f"\n### {PROVIDER_SHORT.get(pn, pn)}")
        q1_lt_q3 = quadrant_data[pn].get("q1_eer_lt_q3_eer")
        lines.append(f"- Q1 EER < Q3 EER: {q1_lt_q3}")
        for qname in ["Q1", "Q2", "Q3", "Q4"]:
            qd = quadrant_data[pn].get(qname, {})
            eer = qd.get("eer")
            lines.append(f"- {qname}: EER = {eer:.4f}" if eer else f"- {qname}: EER = N/A")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {output_path}")


def plot_cross_system_ranked_det(eval_data, dataset, data_dir, output_path, prefix=""):
    """Plot 6: Cross-system ranked DET for P4/P5 only."""
    test_providers = ["P4_XVECTOR", "P5_WAVLM"]
    providers = [p for p in test_providers if p in eval_data]
    if not providers:
        logger.warning("  No cross-system providers in eval data, skipping")
        return

    fig, axes = plt.subplots(1, len(providers), figsize=(5 * len(providers), 4), squeeze=False)
    group_colors = {"bottom": "#d62728", "middle": "#1f77b4", "top": "#2ca02c"}

    for i, pn in enumerate(providers):
        ax = axes[0, i]
        det_info = eval_data[pn].get("ranked_det", {})

        for gname in ["bottom", "middle", "top"]:
            det_file = os.path.join(data_dir, f"det{prefix}_{dataset}_{pn}_{gname}.npz")
            if os.path.exists(det_file):
                data = np.load(det_file)
                fmr = data["fmr"]
                fnmr = data["fnmr"]
                mask = (fmr > 0) & (fnmr > 0)
                if mask.sum() > 10:
                    eer = det_info.get(gname, {}).get("eer", 0)
                    ax.plot(fmr[mask], fnmr[mask], color=group_colors[gname],
                            linewidth=1.5, label=f"{gname.capitalize()} (EER={eer:.3f})")

        sep = det_info.get("eer_separation", 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FMR")
        ax.set_ylabel("FNMR")
        ax.set_title(f"{PROVIDER_SHORT.get(pn, pn)} (unseen, sep={sep:.2f}x)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)

    score_label = "VQI-V" if prefix else "VQI-S"
    fig.suptitle(f"Cross-System Ranked DET ({score_label}) -- {dataset}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_provider_performance_radar(eval_data, output_path):
    """Plot 11: Radar chart of provider performance metrics."""
    providers = [p for p in ALL_PROVIDERS if p in eval_data]
    if not providers:
        return

    metrics = ["ERC Red@20%\n(FNMR=10%)", "DET\nSeparation", "1 - EER\n(Top 15%)", "1 - EER\n(Bottom 15%)"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for pn in providers:
        pdata = eval_data[pn]
        red_20 = pdata.get("fnmr_10pct", {}).get("reductions", {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
        det = pdata.get("ranked_det", {})
        det_sep = det.get("eer_separation", 1.0)
        eer_top = det.get("top", {}).get("eer", 0.5)
        eer_bottom = det.get("bottom", {}).get("eer", 0.5)

        values = [
            min(red_20 / 30.0, 1.0),
            min(det_sep / 3.0, 1.0),
            1.0 - eer_top,
            1.0 - eer_bottom,
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=PROVIDER_SHORT.get(pn, pn),
                color=PROVIDER_COLORS.get(pn, "gray"))
        ax.fill(angles, values, alpha=0.1, color=PROVIDER_COLORS.get(pn, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Provider Performance Radar (VQI-S)", fontsize=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save_fig(fig, output_path)


def plot_fnmr_reduction_forest(eval_data, output_path):
    """Plot 12: Forest plot of FNMR reduction at different rejection rates."""
    providers = [p for p in ALL_PROVIDERS if p in eval_data]
    if not providers:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(providers) * 1.2)))

    reject_fracs = ["0.1", "0.2", "0.3"]
    markers = ["o", "s", "D"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for j, rf in enumerate(reject_fracs):
        reductions = []
        for pn in providers:
            red = eval_data[pn].get("fnmr_10pct", {}).get("reductions", {}).get(rf, {}).get("fnmr_reduction_pct", 0)
            reductions.append(red)
        ax.scatter(reductions, np.arange(len(providers)) + (j - 1) * 0.15,
                   marker=markers[j], s=80, color=colors[j], zorder=3,
                   label=f"Reject {float(rf)*100:.0f}%")

    for i, pn in enumerate(providers):
        reds = []
        for rf in reject_fracs:
            red = eval_data[pn].get("fnmr_10pct", {}).get("reductions", {}).get(rf, {}).get("fnmr_reduction_pct", 0)
            reds.append(red)
        ax.plot(reds, [i - 0.15, i, i + 0.15], color="gray", linewidth=0.8, alpha=0.5)

    ax.set_yticks(np.arange(len(providers)))
    ax.set_yticklabels([PROVIDER_SHORT.get(p, p) for p in providers])
    ax.set_xlabel("FNMR Reduction (%)")
    ax.set_title("FNMR Reduction Forest Plot (FNMR=10%)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    save_fig(fig, output_path)


def plot_erc_parallel_coordinates(eval_data, output_path):
    """Plot 13: Parallel coordinates of FNMR reduction at 10%, 20%, 30%."""
    providers = [p for p in ALL_PROVIDERS if p in eval_data]
    if not providers:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = [0, 1, 2]
    x_labels = ["Reject 10%", "Reject 20%", "Reject 30%"]

    for pn in providers:
        values = []
        for rf in ["0.1", "0.2", "0.3"]:
            red = eval_data[pn].get("fnmr_10pct", {}).get("reductions", {}).get(rf, {}).get("fnmr_reduction_pct", 0)
            values.append(red)
        ax.plot(x, values, marker="o", linewidth=2, markersize=8,
                color=PROVIDER_COLORS.get(pn, "gray"),
                label=PROVIDER_SHORT.get(pn, pn))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("FNMR Reduction (%)")
    ax.set_title("ERC Parallel Coordinates (FNMR=10%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


def plot_det_separation_strip(eval_data, output_path):
    """Plot 14: Strip plot of EER per quality group per provider."""
    providers = [p for p in ALL_PROVIDERS if p in eval_data]
    if not providers:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    group_colors = {"bottom": "#d62728", "middle": "#1f77b4", "top": "#2ca02c"}
    offsets = {"bottom": -0.2, "middle": 0, "top": 0.2}

    x_positions = np.arange(len(providers))

    for gname in ["bottom", "middle", "top"]:
        eers = []
        for pn in providers:
            eer = eval_data[pn].get("ranked_det", {}).get(gname, {}).get("eer", 0)
            eers.append(eer)
        ax.scatter(x_positions + offsets[gname], eers, s=100,
                   color=group_colors[gname], label=gname.capitalize(),
                   edgecolors="black", linewidths=0.5, zorder=3)

    for i, pn in enumerate(providers):
        det = eval_data[pn].get("ranked_det", {})
        y_vals = [det.get(g, {}).get("eer", 0) for g in ["bottom", "middle", "top"]]
        x_vals = [i + offsets[g] for g in ["bottom", "middle", "top"]]
        ax.plot(x_vals, y_vals, color="gray", linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([PROVIDER_SHORT.get(p, p) for p in providers], rotation=15)
    ax.set_ylabel("EER")
    ax.set_title("DET EER by Quality Group (VQI-S)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, output_path)


def plot_combined_erc_summary_table(combined_data, output_path):
    """Plot 20: Combined ERC summary table as image."""
    providers = [p for p in ALL_PROVIDERS if p in combined_data]
    if not providers:
        return

    rows = []
    for pn in providers:
        summary = combined_data[pn].get("summary", {})
        row = [PROVIDER_SHORT.get(pn, pn)]
        for strat in ["s_only", "v_only", "union", "intersection"]:
            red = summary.get(strat, {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
            row.append(f"{red:.1f}%")
        rows.append(row)

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Provider", "S-only", "V-only", "Union", "Intersection"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    fig.suptitle("Combined ERC: FNMR Reduction at 20% Rejection", fontsize=13)
    save_fig(fig, output_path)


def plot_quadrant_score_distributions(vqi_scores_path, pair_scores_path, dataset, output_path):
    """Plot 22: 4-panel violin plots of cosine similarity per quadrant."""
    import pandas as pd

    if not os.path.exists(vqi_scores_path) or not os.path.exists(pair_scores_path):
        logger.warning("  Missing VQI/pair scores for quadrant distributions, skipping")
        return

    vqi_df = pd.read_csv(vqi_scores_path)
    pair_df = pd.read_csv(pair_scores_path)

    fname_to_s = dict(zip(vqi_df["filename"], vqi_df["vqi_s"]))
    fname_to_v = dict(zip(vqi_df["filename"], vqi_df["vqi_v"]))

    min_s = np.array([min(fname_to_s.get(f1, 50), fname_to_s.get(f2, 50))
                      for f1, f2 in zip(pair_df["file1"], pair_df["file2"])])
    min_v = np.array([min(fname_to_v.get(f1, 50), fname_to_v.get(f2, 50))
                      for f1, f2 in zip(pair_df["file1"], pair_df["file2"])])

    thr_s = np.median(min_s)
    thr_v = np.median(min_v)

    high_s = min_s >= thr_s
    high_v = min_v >= thr_v
    quadrants = np.where(high_s & high_v, "Q1",
                np.where(~high_s & high_v, "Q2",
                np.where(~high_s & ~high_v, "Q3", "Q4")))

    cos_sim = pair_df["cos_sim_snorm"].values
    is_genuine = pair_df["is_genuine"].values.astype(bool)

    quad_labels = ["Q1", "Q2", "Q3", "Q4"]
    quad_titles = [
        "Q1: High S, High V",
        "Q2: Low S, High V",
        "Q3: Low S, Low V",
        "Q4: High S, Low V",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (ql, qt) in enumerate(zip(quad_labels, quad_titles)):
        ax = axes[idx // 2, idx % 2]
        mask = quadrants == ql
        gen_scores = cos_sim[mask & is_genuine]
        imp_scores = cos_sim[mask & ~is_genuine]

        if len(gen_scores) > 0 and len(imp_scores) > 0:
            parts_g = ax.violinplot([gen_scores], positions=[0], showmedians=True, showextrema=False)
            parts_i = ax.violinplot([imp_scores], positions=[1], showmedians=True, showextrema=False)

            for pc in parts_g["bodies"]:
                pc.set_facecolor("#2ca02c")
                pc.set_alpha(0.6)
            parts_g["cmedians"].set_color("#2ca02c")

            for pc in parts_i["bodies"]:
                pc.set_facecolor("#d62728")
                pc.set_alpha(0.6)
            parts_i["cmedians"].set_color("#d62728")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Genuine", "Impostor"])
            ax.set_ylabel("S-norm Score")
            ax.set_title(f"{qt}\n(n_gen={len(gen_scores)}, n_imp={len(imp_scores)})")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(qt)

        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Score Distributions by Quality Quadrant -- {dataset}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


DATASET_NAMES = {"voxceleb1": "VoxCeleb1", "vctk": "VCTK", "cnceleb": "CN-Celeb"}


def plot_erc_by_dataset(datasets, output_path):
    """Plot 3: ERC curves grouped by dataset for each provider."""
    dataset_colors = {"voxceleb1": "#1f77b4", "vctk": "#ff7f0e", "cnceleb": "#2ca02c"}

    fig, axes = plt.subplots(1, len(ALL_PROVIDERS), figsize=(4 * len(ALL_PROVIDERS), 4), squeeze=False)

    for i, pn in enumerate(ALL_PROVIDERS):
        ax = axes[0, i]
        for ds_name, eval_data in datasets.items():
            if pn not in eval_data:
                continue
            erc = eval_data[pn].get("fnmr_10pct", {})
            rf = np.array(erc.get("erc_reject_fracs", []))
            fnmr = np.array(erc.get("erc_fnmr", []))
            if len(rf) > 0:
                ax.plot(rf, fnmr, color=dataset_colors.get(ds_name, "gray"),
                        linewidth=1.5, label=DATASET_NAMES.get(ds_name, ds_name))

        ax.set_xlabel("Rejection Fraction")
        ax.set_ylabel("FNMR")
        ax.set_title(PROVIDER_SHORT.get(pn, pn))
        ax.set_xlim(0, 0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ERC by Dataset (FNMR=10%)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_erc_ridgeline_across_datasets(datasets, output_path):
    """Plot 10: FNMR reduction at 20% rejection across datasets per provider."""
    dataset_colors = {"voxceleb1": "#1f77b4", "vctk": "#ff7f0e", "cnceleb": "#2ca02c"}
    ds_list = list(datasets.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ALL_PROVIDERS))
    width = 0.25

    for j, ds_name in enumerate(ds_list):
        eval_data = datasets[ds_name]
        reds = []
        for pn in ALL_PROVIDERS:
            red = eval_data.get(pn, {}).get("fnmr_10pct", {}).get("reductions", {}).get("0.2", {}).get("fnmr_reduction_pct", 0)
            reds.append(red)
        offset = (j - (len(ds_list) - 1) / 2) * width
        ax.bar(x + offset, reds, width, label=DATASET_NAMES.get(ds_name, ds_name),
               color=dataset_colors.get(ds_name, "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels([PROVIDER_SHORT.get(p, p) for p in ALL_PROVIDERS], rotation=15)
    ax.set_ylabel("FNMR Reduction at 20% Rejection (%)")
    ax.set_title("Cross-Dataset FNMR Reduction Comparison (FNMR=10%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description="Step 8 Visualization")
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "vctk", "cnceleb", "all"])
    args = parser.parse_args()

    if args.dataset == "all":
        # Generate per-dataset plots for each available dataset
        for ds in ["voxceleb1", "vctk", "cnceleb"]:
            eval_path = os.path.join(DATA_DIR, "step8_eval", ds, "vqi_s", f"vqi_s_evaluation_{ds}.json")
            if os.path.exists(eval_path):
                logger.info(f"\n=== Generating plots for {ds} ===")
                _generate_per_dataset(ds)

        # Generate multi-dataset plots
        multi_s_data = {}
        for ds in ["voxceleb1", "vctk", "cnceleb"]:
            eval_path = os.path.join(DATA_DIR, "step8_eval", ds, "vqi_s", f"vqi_s_evaluation_{ds}.json")
            if os.path.exists(eval_path):
                multi_s_data[ds] = load_json(eval_path)

        if len(multi_s_data) >= 2:
            report_s_dir = os.path.join(REPORTS_DIR, "step8", "cross_dataset")
            os.makedirs(report_s_dir, exist_ok=True)
            logger.info("\n=== Generating cross-dataset plots ===")
            plot_erc_by_dataset(multi_s_data, os.path.join(report_s_dir, "erc_by_dataset.png"))
            plot_erc_ridgeline_across_datasets(multi_s_data,
                                                os.path.join(report_s_dir, "erc_ridgeline_across_datasets.png"))
        else:
            logger.warning("Need >= 2 datasets for cross-dataset plots")

        return

    dataset = args.dataset
    _generate_per_dataset(dataset)


def _generate_per_dataset(dataset):
    """Generate all per-dataset visualizations."""
    eval_base = os.path.join(DATA_DIR, "step8_eval", dataset)
    eval_s_dir = os.path.join(eval_base, "vqi_s")
    eval_v_dir = os.path.join(eval_base, "vqi_v")
    cross_dir = os.path.join(eval_base, "cross_system")
    dual_dir = os.path.join(eval_base, "dual_score")

    report_s_dir = os.path.join(REPORTS_DIR, "step8", dataset)
    report_v_dir = os.path.join(REPORTS_DIR, "step8_v", dataset)
    os.makedirs(report_s_dir, exist_ok=True)
    os.makedirs(report_v_dir, exist_ok=True)

    # Load evaluation results
    eval_s_path = os.path.join(eval_s_dir, f"vqi_s_evaluation_{dataset}.json")
    eval_v_path = os.path.join(eval_v_dir, f"vqi_v_evaluation_{dataset}.json")
    cross_s_path = os.path.join(cross_dir, f"cross_system_vqi_s_{dataset}.json")
    cross_v_path = os.path.join(cross_dir, f"cross_system_vqi_v_{dataset}.json")
    combined_path = os.path.join(dual_dir, f"combined_erc_{dataset}.json")
    quadrant_path = os.path.join(dual_dir, f"quadrant_analysis_{dataset}.json")
    bench_path = os.path.join(DATA_DIR, "test_scores", "benchmark_results.json")

    eval_s_data = load_json(eval_s_path) if os.path.exists(eval_s_path) else {}
    eval_v_data = load_json(eval_v_path) if os.path.exists(eval_v_path) else {}
    cross_s_data = load_json(cross_s_path) if os.path.exists(cross_s_path) else {}
    cross_v_data = load_json(cross_v_path) if os.path.exists(cross_v_path) else {}
    combined_data = load_json(combined_path) if os.path.exists(combined_path) else {}
    quadrant_data = load_json(quadrant_path) if os.path.exists(quadrant_path) else {}
    bench_data = load_json(bench_path) if os.path.exists(bench_path) else {}

    logger.info(f"Generating visualizations for {dataset}...")

    # ---- VQI-S Plots (1-14) ----
    if eval_s_data:
        logger.info("\n  VQI-S plots:")

        # 1. ERC at FNMR=1%
        plot_erc_all_providers(eval_s_data, "fnmr_1pct",
                               os.path.join(report_s_dir, "erc_fnmr1_all_providers.png"),
                               "FNMR=1%")
        # 2. ERC at FNMR=10%
        plot_erc_all_providers(eval_s_data, "fnmr_10pct",
                               os.path.join(report_s_dir, "erc_fnmr10_all_providers.png"),
                               "FNMR=10%")
        # 4. Ranked DET
        plot_ranked_det_all_providers(eval_s_data, dataset, eval_s_dir, save_dir=report_s_dir)

        # 7. ERC summary table
        plot_erc_summary_table(eval_s_data, os.path.join(report_s_dir, "erc_summary_table.png"))

        # 8. DET separation table
        plot_det_separation_table(eval_s_data, os.path.join(report_s_dir, "det_separation_table.png"))

        # 11. Provider performance radar
        plot_provider_performance_radar(eval_s_data, os.path.join(report_s_dir, "provider_performance_radar.png"))

        # 12. FNMR reduction forest plot
        plot_fnmr_reduction_forest(eval_s_data, os.path.join(report_s_dir, "fnmr_reduction_forest_plot.png"))

        # 13. ERC parallel coordinates
        plot_erc_parallel_coordinates(eval_s_data, os.path.join(report_s_dir, "erc_parallel_coordinates.png"))

        # 14. DET separation strip
        plot_det_separation_strip(eval_s_data, os.path.join(report_s_dir, "det_separation_strip.png"))

    # 5. Cross-system ERC
    if cross_s_data:
        plot_cross_system_erc(cross_s_data,
                              os.path.join(report_s_dir, "cross_system_erc.png"), "S")

    # 6. Cross-system ranked DET
    if eval_s_data and cross_s_data:
        plot_cross_system_ranked_det(eval_s_data, dataset, eval_s_dir,
                                     os.path.join(report_s_dir, "cross_system_ranked_det.png"))

    # 9. Timing benchmark
    plot_timing_benchmark(os.path.join(report_s_dir, "timing_benchmark.png"))

    # 15. Analysis.md
    if eval_s_data:
        write_analysis_s(eval_s_data, cross_s_data, bench_data, dataset,
                         os.path.join(report_s_dir, "analysis.md"))

    # ---- VQI-V Plots (16-25) ----
    if eval_v_data:
        logger.info("\n  VQI-V plots:")

        # 16. VQI-V ERC
        plot_erc_all_providers(eval_v_data, "fnmr_10pct",
                               os.path.join(report_v_dir, "erc_v_all_providers.png"),
                               "VQI-V FNMR=10%")

        # 17. VQI-V Ranked DET
        plot_ranked_det_all_providers(eval_v_data, dataset, eval_v_dir, prefix="_v", save_dir=report_v_dir)

    # 18. Cross-system V ERC
    if cross_v_data:
        plot_cross_system_erc(cross_v_data,
                              os.path.join(report_v_dir, "cross_system_v_erc.png"), "V")

        # Cross-system V ranked DET
        if eval_v_data:
            plot_cross_system_ranked_det(eval_v_data, dataset, eval_v_dir,
                                         os.path.join(report_v_dir, "cross_system_v_ranked_det.png"),
                                         prefix="_v")

    # 19. Combined ERC comparison
    if combined_data:
        plot_combined_erc_comparison(combined_data,
                                     os.path.join(report_v_dir, "combined_erc_comparison.png"))

        # 20. Combined ERC summary table
        plot_combined_erc_summary_table(combined_data,
                                        os.path.join(report_v_dir, "combined_erc_summary_table.png"))

    # 21. Quadrant EER comparison
    if quadrant_data:
        plot_quadrant_eer_comparison(quadrant_data,
                                     os.path.join(report_v_dir, "quadrant_eer_comparison.png"))

        # 22. Quadrant score distributions
        vqi_scores_path = os.path.join(DATA_DIR, "test_scores", f"vqi_scores_test_{dataset}.csv")
        pair_scores_path = os.path.join(DATA_DIR, "test_scores", f"pair_scores_{dataset}_P1_ECAPA.csv")
        plot_quadrant_score_distributions(vqi_scores_path, pair_scores_path, dataset,
                                          os.path.join(report_v_dir, "quadrant_score_distributions.png"))

        # 23. Quadrant performance table
        plot_quadrant_performance_table(quadrant_data,
                                        os.path.join(report_v_dir, "quadrant_performance_table.png"))

    # 24. Dual-score value assessment
    if combined_data and quadrant_data:
        plot_dual_score_value_assessment(combined_data, quadrant_data,
                                         os.path.join(report_v_dir, "dual_score_value_assessment.png"))

    # 25. Analysis V
    if eval_v_data:
        write_analysis_v(eval_v_data, combined_data, quadrant_data, dataset,
                         os.path.join(report_v_dir, "analysis_v.md"))

    logger.info(f"\nVisualization complete for {dataset}.")
    logger.info(f"  VQI-S reports: {report_s_dir}")
    logger.info(f"  VQI-V reports: {report_v_dir}")


if __name__ == "__main__":
    main()
