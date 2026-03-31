"""Step 8 (PCA-90%): Evaluation of Predictive Power for PCA-90% models.

Adapted from run_step8.py for PCA dimensionality-reduced models.
Differences from the full-feature version:
  - Phase A is skipped entirely (data already extracted by full-feature runs)
  - VQI scores loaded from data/step8/pca90/test_scores_pca90/ (PCA-90% model predictions)
  - Pair definitions and provider pair scores loaded from data/step8/full_feature/test_scores/ (unchanged)
  - Output written to data/step8/pca90/step8_eval_pca90/{dataset}/
  - Phase F (benchmarks) is skipped (not model-specific)
  - DET curve .npz files include a "_pca90" suffix

Phases:
  B: VQI-S evaluation (ERC, DET for 3 datasets x 5 providers)
  C: VQI-V evaluation (ERC, DET)
  D: Cross-system (P4/P5 vs P1-P3)
  E: Dual-score (combined ERC, quadrant analysis)
  G: Summary tables + reports

Usage:
    python scripts/run_step8_pca90.py --dataset voxceleb1 [--resume]
    python scripts/run_step8_pca90.py --dataset vctk --resume
    python scripts/run_step8_pca90.py --dataset cnceleb --resume
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

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
SCORES_DIR = os.path.join(DATA_DIR, "step8", "full_feature", "test_scores")           # pair definitions + provider pair scores
SCORES_PCA90_DIR = os.path.join(DATA_DIR, "step8", "pca90", "test_scores_pca90")  # PCA-90% VQI scores

ALL_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]

DATASET_TO_SPLIT = {
    "voxceleb1": "test_voxceleb1",
    "vctk": "test_vctk",
    "cnceleb": "test_cnceleb",
    "vpqad": "test_vpqad",
    "vseadc": "test_vseadc",
}


def load_evaluation_data(dataset, split_name):
    """Load all data needed for evaluation phases.

    VQI scores come from the PCA-90% model output directory.
    Pair definitions and provider pair scores come from the standard test_scores directory.
    """
    # Load VQI scores from PCA-90% directory
    scores_df = pd.read_csv(os.path.join(SCORES_PCA90_DIR, f"vqi_scores_{split_name}.csv"))
    vqi_s = scores_df["vqi_s"].values
    vqi_v = scores_df["vqi_v"].values
    filenames = scores_df["filename"].tolist()
    speaker_ids = scores_df["speaker_id"].tolist()

    # Build filename -> index mapping
    fname_to_idx = {fn: i for i, fn in enumerate(filenames)}

    # Load pair definitions from standard directory
    pair_def_path = os.path.join(SCORES_DIR, f"pair_definitions_{dataset}.npz")
    pair_data = np.load(pair_def_path, allow_pickle=True)
    pairs = pair_data["pairs"]
    labels = pair_data["labels"]

    # Load pair scores per provider from standard directory
    provider_pair_scores = {}
    for pn in ALL_PROVIDERS:
        pair_csv = os.path.join(SCORES_DIR, f"pair_scores_{dataset}_{pn}.csv")
        if os.path.exists(pair_csv):
            pdf = pd.read_csv(pair_csv)
            cos_sim = pdf["cos_sim"].values.astype(np.float32)
            cos_sim_snorm = None
            if "cos_sim_snorm" in pdf.columns:
                cos_sim_snorm = pdf["cos_sim_snorm"].values.astype(np.float32)
            provider_pair_scores[pn] = {
                "cos_sim": cos_sim,
                "cos_sim_snorm": cos_sim_snorm,
            }

    return {
        "vqi_s": vqi_s,
        "vqi_v": vqi_v,
        "filenames": filenames,
        "speaker_ids": speaker_ids,
        "pairs": pairs,
        "labels": labels,
        "provider_pair_scores": provider_pair_scores,
        "fname_to_idx": fname_to_idx,
    }


def phase_b_vqi_s_evaluation(data, dataset, output_dir):
    """Phase B: VQI-S ERC + DET evaluation."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE B: VQI-S EVALUATION (PCA-90%)")
    logger.info("=" * 70)

    from vqi.evaluation.erc import (
        compute_erc, find_tau_for_fnmr,
        compute_fnmr_reduction_at_reject, compute_random_rejection_baseline,
    )
    from vqi.evaluation.det import compute_ranked_det

    os.makedirs(output_dir, exist_ok=True)
    pairs = data["pairs"]
    labels = data["labels"]
    vqi_s = data["vqi_s"]

    # Pairwise quality = min(q1, q2) for VQI-S
    quality_gen_s = np.minimum(vqi_s[pairs[labels == 1, 0]], vqi_s[pairs[labels == 1, 1]])
    quality_imp_s = np.minimum(vqi_s[pairs[labels == 0, 0]], vqi_s[pairs[labels == 0, 1]])

    gen_mask = labels == 1
    imp_mask = labels == 0

    results = {}

    for pn, pscores in data["provider_pair_scores"].items():
        logger.info(f"\n  Provider: {pn}")
        # Use S-norm scores if available, else raw cosine
        sim = pscores["cos_sim_snorm"] if pscores["cos_sim_snorm"] is not None else pscores["cos_sim"]

        gen_sim = sim[gen_mask]
        imp_sim = sim[imp_mask]

        provider_results = {}

        # ERC at FNMR=1% and FNMR=10%
        for target_fnmr, label in [(0.01, "fnmr_1pct"), (0.10, "fnmr_10pct")]:
            tau = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr)
            erc = compute_erc(gen_sim, imp_sim, quality_gen_s, quality_imp_s, tau)
            reductions = compute_fnmr_reduction_at_reject(erc)
            random_baseline = compute_random_rejection_baseline(gen_sim, tau, erc["reject_fracs"])

            provider_results[label] = {
                "tau": float(tau),
                "erc_reject_fracs": erc["reject_fracs"].tolist(),
                "erc_fnmr": erc["fnmr_values"].tolist(),
                "erc_fmr": erc["fmr_values"].tolist(),
                "q_thresholds": erc["q_thresholds"].tolist(),
                "random_baseline": random_baseline.tolist(),
                "reductions": {str(k): v for k, v in reductions.items()},
            }
            red_20 = reductions.get(0.20, {}).get("fnmr_reduction_pct", 0)
            logger.info(f"    {label}: tau={tau:.4f}, FNMR reduction@20%%={red_20:.1f}%%")

        # Ranked DET
        det_result = compute_ranked_det(gen_sim, imp_sim, quality_gen_s, quality_imp_s)
        provider_results["ranked_det"] = {
            "q_low": det_result["q_low"],
            "q_high": det_result["q_high"],
            "eer_separation": det_result["eer_separation"],
        }
        for gname, gdata in det_result["groups"].items():
            provider_results["ranked_det"][gname] = {
                "n_genuine": gdata["n_genuine"],
                "n_impostor": gdata["n_impostor"],
                "eer": gdata["det"]["eer"] if gdata["det"] else None,
                "fnmr_at_fmr_001": gdata["fnmr_at_fmr_001"],
                "fnmr_at_fmr_0001": gdata["fnmr_at_fmr_0001"],
            }
            if gdata["det"]:
                # Save DET curve data with _pca90 suffix
                det_fname = f"det_{dataset}_{pn}_{gname}_pca90.npz"
                np.savez(
                    os.path.join(output_dir, det_fname),
                    fmr=gdata["det"]["fmr"],
                    fnmr=gdata["det"]["fnmr"],
                    thresholds=gdata["det"]["thresholds"],
                )

        eer_sep = det_result["eer_separation"]
        logger.info(f"    Ranked DET: EER separation = {eer_sep:.2f}x")

        results[pn] = provider_results

    # Save results
    results_path = os.path.join(output_dir, f"vqi_s_evaluation_{dataset}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved: {results_path}")

    return results


def phase_c_vqi_v_evaluation(data, dataset, output_dir):
    """Phase C: VQI-V ERC + DET evaluation."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE C: VQI-V EVALUATION (PCA-90%)")
    logger.info("=" * 70)

    from vqi.evaluation.erc import (
        compute_erc, find_tau_for_fnmr,
        compute_fnmr_reduction_at_reject, compute_random_rejection_baseline,
    )
    from vqi.evaluation.det import compute_ranked_det

    os.makedirs(output_dir, exist_ok=True)
    pairs = data["pairs"]
    labels = data["labels"]
    vqi_v = data["vqi_v"]

    quality_gen_v = np.minimum(vqi_v[pairs[labels == 1, 0]], vqi_v[pairs[labels == 1, 1]])
    quality_imp_v = np.minimum(vqi_v[pairs[labels == 0, 0]], vqi_v[pairs[labels == 0, 1]])

    gen_mask = labels == 1
    imp_mask = labels == 0

    results = {}

    for pn, pscores in data["provider_pair_scores"].items():
        logger.info(f"\n  Provider: {pn}")
        sim = pscores["cos_sim_snorm"] if pscores["cos_sim_snorm"] is not None else pscores["cos_sim"]
        gen_sim = sim[gen_mask]
        imp_sim = sim[imp_mask]

        provider_results = {}

        for target_fnmr, label in [(0.01, "fnmr_1pct"), (0.10, "fnmr_10pct")]:
            tau = find_tau_for_fnmr(gen_sim, imp_sim, target_fnmr)
            erc = compute_erc(gen_sim, imp_sim, quality_gen_v, quality_imp_v, tau)
            reductions = compute_fnmr_reduction_at_reject(erc)
            random_baseline = compute_random_rejection_baseline(gen_sim, tau, erc["reject_fracs"])

            provider_results[label] = {
                "tau": float(tau),
                "erc_reject_fracs": erc["reject_fracs"].tolist(),
                "erc_fnmr": erc["fnmr_values"].tolist(),
                "erc_fmr": erc["fmr_values"].tolist(),
                "q_thresholds": erc["q_thresholds"].tolist(),
                "random_baseline": random_baseline.tolist(),
                "reductions": {str(k): v for k, v in reductions.items()},
            }
            red_20 = reductions.get(0.20, {}).get("fnmr_reduction_pct", 0)
            logger.info(f"    {label}: tau={tau:.4f}, FNMR reduction@20%%={red_20:.1f}%%")

        det_result = compute_ranked_det(gen_sim, imp_sim, quality_gen_v, quality_imp_v)
        provider_results["ranked_det"] = {
            "q_low": det_result["q_low"],
            "q_high": det_result["q_high"],
            "eer_separation": det_result["eer_separation"],
        }
        for gname, gdata in det_result["groups"].items():
            provider_results["ranked_det"][gname] = {
                "n_genuine": gdata["n_genuine"],
                "n_impostor": gdata["n_impostor"],
                "eer": gdata["det"]["eer"] if gdata["det"] else None,
                "fnmr_at_fmr_001": gdata["fnmr_at_fmr_001"],
                "fnmr_at_fmr_0001": gdata["fnmr_at_fmr_0001"],
            }
            if gdata["det"]:
                # Save DET curve data with _pca90 suffix
                np.savez(
                    os.path.join(output_dir, f"det_v_{dataset}_{pn}_{gname}_pca90.npz"),
                    fmr=gdata["det"]["fmr"],
                    fnmr=gdata["det"]["fnmr"],
                    thresholds=gdata["det"]["thresholds"],
                )

        eer_sep = det_result["eer_separation"]
        logger.info(f"    Ranked DET: EER separation = {eer_sep:.2f}x")

        results[pn] = provider_results

    results_path = os.path.join(output_dir, f"vqi_v_evaluation_{dataset}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved: {results_path}")

    return results


def phase_d_cross_system(data, dataset, output_dir):
    """Phase D: Cross-system generalization (P4/P5 vs P1-P3)."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE D: CROSS-SYSTEM GENERALIZATION (PCA-90%)")
    logger.info("=" * 70)

    from vqi.evaluation.cross_system import evaluate_cross_system

    os.makedirs(output_dir, exist_ok=True)
    pairs = data["pairs"]
    labels = data["labels"]
    vqi_s = data["vqi_s"]
    vqi_v = data["vqi_v"]

    gen_mask = labels == 1
    imp_mask = labels == 0

    # Pairwise qualities
    quality_gen_s = np.minimum(vqi_s[pairs[gen_mask, 0]], vqi_s[pairs[gen_mask, 1]])
    quality_imp_s = np.minimum(vqi_s[pairs[imp_mask, 0]], vqi_s[pairs[imp_mask, 1]])
    quality_gen_v = np.minimum(vqi_v[pairs[gen_mask, 0]], vqi_v[pairs[gen_mask, 1]])
    quality_imp_v = np.minimum(vqi_v[pairs[imp_mask, 0]], vqi_v[pairs[imp_mask, 1]])

    # Prepare provider data
    for score_type, quality_gen, quality_imp, label in [
        ("vqi_s", quality_gen_s, quality_imp_s, "S"),
        ("vqi_v", quality_gen_v, quality_imp_v, "V"),
    ]:
        logger.info(f"\n  Cross-system for VQI-{label}:")
        provider_data = {}
        for pn, pscores in data["provider_pair_scores"].items():
            sim = pscores["cos_sim_snorm"] if pscores["cos_sim_snorm"] is not None else pscores["cos_sim"]
            provider_data[pn] = {
                "genuine_sim": sim[gen_mask],
                "impostor_sim": sim[imp_mask],
            }

        results = evaluate_cross_system(provider_data, quality_gen, quality_imp)

        verdict = results["_verdict"]
        logger.info(f"    Verdict: {'PASS' if verdict['passed'] else 'FAIL'}")
        logger.info(f"    All monotonic: {verdict['all_monotonic']}")
        logger.info(f"    All positive reduction: {verdict['all_positive_reduction']}")
        logger.info(f"    Mean train reduction@20%%: {verdict['mean_train_reduction_20pct']:.1f}%%")
        logger.info(f"    Mean test reduction@20%%: {verdict['mean_test_reduction_20pct']:.1f}%%")

        # Save results (excluding numpy arrays for JSON serialization)
        save_results = {"_verdict": verdict}
        for pn in provider_data:
            if pn in results and pn != "_verdict":
                pres = results[pn]
                save_results[pn] = {
                    "is_train": pres["is_train"],
                    "erc": {},
                }
                for fnmr_key, erc_data in pres["erc"].items():
                    save_results[pn]["erc"][fnmr_key] = {
                        "tau": erc_data["tau"],
                        "monotonic": erc_data["monotonic"],
                        "reductions": {str(k): v for k, v in erc_data["reductions"].items()},
                    }
                if pres["det"]:
                    save_results[pn]["det"] = {
                        "q_low": pres["det"]["q_low"],
                        "q_high": pres["det"]["q_high"],
                        "eer_separation": pres["det"]["eer_separation"],
                    }

        results_path = os.path.join(output_dir, f"cross_system_{score_type}_{dataset}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"    Saved: {results_path}")


def phase_e_dual_score(data, dataset, output_dir):
    """Phase E: Dual-score evaluation (combined ERC, quadrant analysis)."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE E: DUAL-SCORE EVALUATION (PCA-90%)")
    logger.info("=" * 70)

    from vqi.evaluation.combined_erc import (
        compute_combined_erc, compute_combined_fnmr_reduction_summary,
    )
    from vqi.evaluation.quadrant_analysis import (
        assign_pair_quadrants, compute_quadrant_performance,
    )
    from vqi.evaluation.erc import find_tau_for_fnmr

    os.makedirs(output_dir, exist_ok=True)
    pairs = data["pairs"]
    labels = data["labels"]
    vqi_s = data["vqi_s"]
    vqi_v = data["vqi_v"]

    gen_mask = labels == 1
    imp_mask = labels == 0

    quality_gen_s = np.minimum(vqi_s[pairs[gen_mask, 0]], vqi_s[pairs[gen_mask, 1]])
    quality_gen_v = np.minimum(vqi_v[pairs[gen_mask, 0]], vqi_v[pairs[gen_mask, 1]])
    quality_imp_s = np.minimum(vqi_s[pairs[imp_mask, 0]], vqi_s[pairs[imp_mask, 1]])
    quality_imp_v = np.minimum(vqi_v[pairs[imp_mask, 0]], vqi_v[pairs[imp_mask, 1]])

    combined_results = {}
    quadrant_results = {}

    for pn, pscores in data["provider_pair_scores"].items():
        logger.info(f"\n  Provider: {pn}")
        sim = pscores["cos_sim_snorm"] if pscores["cos_sim_snorm"] is not None else pscores["cos_sim"]
        gen_sim = sim[gen_mask]
        imp_sim = sim[imp_mask]

        # Combined ERC at FNMR=10%
        tau = find_tau_for_fnmr(gen_sim, imp_sim, 0.10)
        cerc = compute_combined_erc(
            gen_sim, imp_sim,
            quality_gen_s, quality_gen_v,
            quality_imp_s, quality_imp_v,
            tau,
        )
        summary = compute_combined_fnmr_reduction_summary(cerc)

        combined_results[pn] = {
            "tau": float(tau),
            "strategies": {},
        }
        for strat, strat_data in cerc.items():
            combined_results[pn]["strategies"][strat] = {
                "reject_fracs": strat_data["reject_fracs"].tolist(),
                "fnmr_values": strat_data["fnmr_values"].tolist(),
            }
        combined_results[pn]["summary"] = {
            strat: {str(k): v for k, v in red.items()}
            for strat, red in summary.items()
        }

        # Log summary
        for strat in ["s_only", "v_only", "union", "intersection"]:
            red_20 = summary[strat].get(0.20, {}).get("fnmr_reduction_pct", 0)
            logger.info(f"    {strat}: FNMR reduction@20%% = {red_20:.1f}%%")

        # Quadrant analysis
        quad_gen = assign_pair_quadrants(vqi_s, vqi_v, pairs[gen_mask])
        quad_imp = assign_pair_quadrants(vqi_s, vqi_v, pairs[imp_mask])
        qperf = compute_quadrant_performance(gen_sim, imp_sim, quad_gen, quad_imp)

        quadrant_results[pn] = {
            "q1_eer_lt_q3_eer": qperf["q1_eer_lt_q3_eer"],
            "eer_q1": qperf["eer_q1"],
            "eer_q3": qperf["eer_q3"],
        }
        for qname, qdata in qperf["quadrants"].items():
            quadrant_results[pn][qname] = {
                "n_genuine": qdata["n_genuine"],
                "n_impostor": qdata["n_impostor"],
                "eer": qdata["eer"] if not np.isnan(qdata["eer"]) else None,
                "fnmr_at_fmr_001": qdata["fnmr_at_fmr_001"] if not np.isnan(qdata["fnmr_at_fmr_001"]) else None,
                "fnmr_at_fmr_0001": qdata["fnmr_at_fmr_0001"] if not np.isnan(qdata["fnmr_at_fmr_0001"]) else None,
            }

        logger.info(f"    Quadrant: Q1 EER={qperf['eer_q1']:.4f}, "
                    f"Q3 EER={qperf['eer_q3']:.4f}, "
                    f"Q1 < Q3: {qperf['q1_eer_lt_q3_eer']}")

    # Save results
    combined_path = os.path.join(output_dir, f"combined_erc_{dataset}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    quadrant_path = os.path.join(output_dir, f"quadrant_analysis_{dataset}.json")
    with open(quadrant_path, "w", encoding="utf-8") as f:
        json.dump(quadrant_results, f, indent=2)

    logger.info(f"\n  Saved: {combined_path}")
    logger.info(f"  Saved: {quadrant_path}")

    return combined_results, quadrant_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 8 (PCA-90%%): Evaluation of Predictive Power"
    )
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "vctk", "cnceleb", "vpqad", "vseadc"])
    parser.add_argument("--skip-s", action="store_true", help="Skip Phase B (VQI-S)")
    parser.add_argument("--skip-v", action="store_true", help="Skip Phase C (VQI-V)")
    parser.add_argument("--skip-cross", action="store_true", help="Skip Phase D (cross-system)")
    parser.add_argument("--skip-dual", action="store_true", help="Skip Phase E (dual-score)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    split_name = DATASET_TO_SPLIT[dataset]

    # Output directories — under step8_eval_pca90/
    eval_base = os.path.join(DATA_DIR, "step8", "pca90", "step8_eval_pca90")
    eval_s_dir = os.path.join(eval_base, dataset, "vqi_s")
    eval_v_dir = os.path.join(eval_base, dataset, "vqi_v")
    eval_cross_dir = os.path.join(eval_base, dataset, "cross_system")
    eval_dual_dir = os.path.join(eval_base, dataset, "dual_score")

    for d in [eval_s_dir, eval_v_dir, eval_cross_dir, eval_dual_dir]:
        os.makedirs(d, exist_ok=True)

    # Checkpoint
    checkpoint_path = os.path.join(eval_base, dataset, "_checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    completed_phases = set()
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        completed_phases = set(ckpt.get("completed_phases", []))
        logger.info(f"Resuming: phases {completed_phases} already done")

    def save_checkpoint(phase):
        completed_phases.add(phase)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_phases": sorted(completed_phases), "dataset": dataset}, f)

    logger.info(f"Step 8 PCA-90%% Evaluation for dataset: {dataset} (split: {split_name})")
    logger.info(f"VQI scores from: {SCORES_PCA90_DIR}")
    logger.info(f"Pair data from:  {SCORES_DIR}")
    logger.info(f"Output to:       {eval_base}/{dataset}/")
    logger.info(f"Skips: S={args.skip_s}, V={args.skip_v}, "
                f"cross={args.skip_cross}, dual={args.skip_dual}")

    t0 = time.time()

    # No Phase A — data already extracted by full-feature Step 8 runs

    # Load data for evaluation phases
    try:
        data = load_evaluation_data(dataset, split_name)
        logger.info(f"\nLoaded evaluation data: {len(data['vqi_s'])} files, "
                    f"{len(data['pairs'])} pairs, "
                    f"{len(data['provider_pair_scores'])} providers")
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {e}")
        logger.error("Ensure PCA-90%% VQI scores exist in data/step8/pca90/test_scores_pca90/ "
                     "and pair data exists in data/step8/full_feature/test_scores/.")
        sys.exit(1)

    # Phase B: VQI-S
    if not args.skip_s and "B" not in completed_phases:
        phase_b_vqi_s_evaluation(data, dataset, eval_s_dir)
        save_checkpoint("B")

    # Phase C: VQI-V
    if not args.skip_v and "C" not in completed_phases:
        phase_c_vqi_v_evaluation(data, dataset, eval_v_dir)
        save_checkpoint("C")

    # Phase D: Cross-system
    if not args.skip_cross and "D" not in completed_phases:
        phase_d_cross_system(data, dataset, eval_cross_dir)
        save_checkpoint("D")

    # Phase E: Dual-score
    if not args.skip_dual and "E" not in completed_phases:
        phase_e_dual_score(data, dataset, eval_dual_dir)
        save_checkpoint("E")

    # No Phase F — benchmarks are not model-specific

    # Cleanup checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info(f"Step 8 PCA-90%% evaluation for '{dataset}' complete in {elapsed/60:.1f} min")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
