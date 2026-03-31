"""
Step 5.6: Generate comprehensive VQI-S feature evaluation report.
"""

import os
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "step5", "evaluation")
REPORTS = os.path.join(PROJECT_ROOT, "reports", "step5")


def main():
    os.makedirs(REPORTS, exist_ok=True)

    sp = pd.read_csv(os.path.join(EVAL_DIR, "spearman_correlations.csv"))
    removed = pd.read_csv(os.path.join(EVAL_DIR, "removed_redundant_features.csv"))
    imp = pd.read_csv(os.path.join(EVAL_DIR, "rf_importance_rankings.csv"))
    erc = pd.read_csv(os.path.join(EVAL_DIR, "erc_per_feature.csv"))
    with open(os.path.join(EVAL_DIR, "feature_selection_summary.yaml"), encoding="utf-8") as f:
        summary = yaml.safe_load(f)
    with open(os.path.join(EVAL_DIR, "selected_features.txt"), encoding="utf-8") as f:
        selected = set(l.strip() for l in f if l.strip())

    # Build full feature table
    sp_dict = sp.set_index("feature_name").to_dict("index")
    imp_dict = dict(zip(imp["feature_name"], imp["importance"]))
    erc_dict = dict(zip(erc["feature_name"], erc["auc_mean"]))
    removed_set = set(removed["removed_feature"]) if len(removed) > 0 else set()

    lines = [
        "# VQI-S Feature Evaluation Report",
        "",
        f"**Generated:** 2026-02-16",
        f"**N_candidates:** 544",
        f"**N_selected:** {summary['n_selected']}",
        f"**OOB accuracy:** {summary['final_oob_accuracy']:.4f}",
        "",
        "## Selection Funnel",
        "| Stage | Count | Removed |",
        "|-------|-------|---------|",
        "| Candidates | 544 | - |",
        "| Zero-variance removed | 513 | 31 |",
        f"| Redundancy removed | 449 | 64 |",
        f"| RF pruned | {summary['n_selected']} | {449 - summary['n_selected']} |",
        "",
        "## All 544 Features",
        "",
        "| # | Feature | |rho| | Retained | Reason | Importance | ERC AUC |",
        "|---|---------|-------|----------|--------|------------|---------|",
    ]

    for i, row in sp.iterrows():
        name = row["feature_name"]
        rho = f"{row['abs_rho_mean']:.4f}"
        if name in selected:
            retained = "Yes"
            reason = "Selected"
            importance = f"{imp_dict.get(name, 0):.6f}"
            erc_val = f"{erc_dict.get(name, 0):.4f}"
        elif name in removed_set:
            retained = "No"
            reason = "Redundant"
            importance = "-"
            erc_val = "-"
        elif row["abs_rho_mean"] == 0 and row["pval_P1"] == 1.0:
            retained = "No"
            reason = "Constant"
            importance = "-"
            erc_val = "-"
        else:
            retained = "No"
            reason = "RF pruned"
            importance = "-"
            erc_val = "-"
        lines.append(f"| {i+1} | {name} | {rho} | {retained} | {reason} | {importance} | {erc_val} |")

    # Redundancy pairs
    lines += [
        "",
        "## Redundancy Pairs Removed",
        "",
        "| Removed | Kept | Pearson r |",
        "|---------|------|-----------|",
    ]
    for _, row in removed.sort_values("pearson_r", ascending=False).iterrows():
        lines.append(f"| {row['removed_feature']} | {row['kept_feature']} | {row['pearson_r']:.4f} |")

    # Top features by each metric
    lines += [
        "",
        "## Top 10 by Spearman |rho|",
        "",
        "| Feature | |rho| |",
        "|---------|-------|",
    ]
    for _, row in sp.nlargest(10, "abs_rho_mean").iterrows():
        lines.append(f"| {row['feature_name']} | {row['abs_rho_mean']:.4f} |")

    lines += [
        "",
        "## Top 10 by RF Importance",
        "",
        "| Feature | Importance |",
        "|---------|------------|",
    ]
    for _, row in imp.nlargest(10, "importance").iterrows():
        lines.append(f"| {row['feature_name']} | {row['importance']:.6f} |")

    lines += [
        "",
        "## Top 10 by ERC AUC (best quality predictors)",
        "",
        "| Feature | ERC AUC |",
        "|---------|---------|",
    ]
    for _, row in erc.nsmallest(10, "auc_mean").iterrows():
        lines.append(f"| {row['feature_name']} | {row['auc_mean']:.4f} |")

    path = os.path.join(REPORTS, "feature_evaluation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"VQI-S report: {path}")


if __name__ == "__main__":
    main()
