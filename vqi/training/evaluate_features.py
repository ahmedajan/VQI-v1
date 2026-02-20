"""
Step 5: Feature Evaluation and Selection

Three-stage pipeline:
  1. Spearman correlation with Fisher Ratio (d') per provider
  2. Redundancy removal (Pearson |r| > threshold)
  3. Random Forest importance pruning (iterative, <0.5% of max)

Plus ERC-based single-feature quality evaluation.

Shared pipeline function handles both VQI-S (544 features) and VQI-V (161 features).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def load_candidate_features(
    features_path: str,
    names_path: str,
    training_csv: str,
    fisher_csv: str,
) -> Tuple[np.ndarray, List[str], np.ndarray, pd.DataFrame, np.ndarray]:
    """Load features, labels, and Fisher values; identify zero-variance columns.

    Returns:
        X: (N, D) feature matrix
        feature_names: list of D feature name strings
        labels: (N,) binary labels
        fisher_df: DataFrame with fisher_P1/P2/P3 aligned to rows of X
        valid_mask: (D,) boolean mask, True for non-constant features
    """
    logger.info("Loading features from %s", features_path)
    X = np.load(features_path)
    with open(names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    assert X.shape[1] == len(feature_names), (
        f"Feature matrix columns ({X.shape[1]}) != names ({len(feature_names)})"
    )

    logger.info("Loading training labels from %s", training_csv)
    train_df = pd.read_csv(training_csv)
    assert len(train_df) == X.shape[0], (
        f"Training CSV rows ({len(train_df)}) != feature rows ({X.shape[0]})"
    )
    labels = train_df["label"].values.astype(np.int32)

    logger.info("Loading Fisher values from %s", fisher_csv)
    fisher_all = pd.read_csv(fisher_csv)
    fisher_df = train_df[["filename"]].merge(
        fisher_all[["filename", "fisher_P1", "fisher_P2", "fisher_P3"]],
        on="filename",
        how="left",
    )
    n_missing = fisher_df["fisher_P1"].isna().sum()
    if n_missing > 0:
        logger.warning("%d samples missing Fisher values", n_missing)

    # Identify zero-variance columns
    stds = np.std(X, axis=0)
    valid_mask = stds > 1e-12
    n_const = int(np.sum(~valid_mask))
    logger.info(
        "Loaded %d samples x %d features (%d zero-variance excluded)",
        X.shape[0], X.shape[1], n_const,
    )

    return X, feature_names, labels, fisher_df, valid_mask


def spearman_evaluation(
    X: np.ndarray,
    names: List[str],
    fisher_df: pd.DataFrame,
    valid_mask: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """Compute Spearman correlation between each feature and Fisher d' (P1/P2/P3).

    Returns DataFrame with columns:
        feature_name, feature_idx, rho_P1, rho_P2, rho_P3, rho_mean,
        pval_P1, pval_P2, pval_P3, pval_mean, abs_rho_mean
    """
    os.makedirs(output_dir, exist_ok=True)
    n_features = X.shape[1]
    providers = ["P1", "P2", "P3"]
    fisher_cols = [f"fisher_{p}" for p in providers]

    rows = []
    for i in range(n_features):
        row = {"feature_name": names[i], "feature_idx": i}
        if not valid_mask[i]:
            # Constant feature -> zero correlation
            for p in providers:
                row[f"rho_{p}"] = 0.0
                row[f"pval_{p}"] = 1.0
            row["rho_mean"] = 0.0
            row["pval_mean"] = 1.0
            row["abs_rho_mean"] = 0.0
        else:
            rhos = []
            pvals = []
            for j, p in enumerate(providers):
                fisher_vals = fisher_df[fisher_cols[j]].values
                mask = ~np.isnan(fisher_vals)
                if mask.sum() < 10:
                    rho, pval = 0.0, 1.0
                else:
                    rho, pval = spearmanr(X[mask, i], fisher_vals[mask])
                    if np.isnan(rho):
                        rho, pval = 0.0, 1.0
                row[f"rho_{p}"] = float(rho)
                row[f"pval_{p}"] = float(pval)
                rhos.append(rho)
                pvals.append(pval)
            row["rho_mean"] = float(np.mean(rhos))
            row["pval_mean"] = float(np.mean(pvals))
            row["abs_rho_mean"] = float(np.mean(np.abs(rhos)))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "spearman_correlations.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(
        "Spearman correlations saved to %s (%d features)", out_path, len(df)
    )

    # Summary stats
    above_02 = (df["abs_rho_mean"] > 0.2).sum()
    above_03 = (df["abs_rho_mean"] > 0.3).sum()
    logger.info(
        "Features with |rho| > 0.2: %d, > 0.3: %d", above_02, above_03
    )

    return df


def remove_redundant_features(
    X: np.ndarray,
    names: List[str],
    spearman_df: pd.DataFrame,
    valid_mask: np.ndarray,
    output_dir: str,
    threshold: float = 0.95,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Remove features with Pearson |r| > threshold (keep higher Spearman).

    Returns:
        X_reduced: (N, D_reduced)
        names_reduced: list of D_reduced names
        kept_indices: (D_reduced,) original column indices
    """
    os.makedirs(output_dir, exist_ok=True)

    # Work only with valid (non-constant) features
    valid_idx = np.where(valid_mask)[0]
    X_valid = X[:, valid_idx]
    names_valid = [names[i] for i in valid_idx]

    logger.info(
        "Computing correlation matrix for %d valid features...",
        len(valid_idx),
    )
    corr_matrix = np.corrcoef(X_valid.T)
    np.fill_diagonal(corr_matrix, 0.0)  # ignore self-correlation

    # Save full correlation matrix
    corr_path = os.path.join(output_dir, "feature_correlation_matrix.npy")
    np.save(corr_path, corr_matrix)

    # Build Spearman lookup: feature_name -> abs_rho_mean
    spearman_lookup = dict(
        zip(spearman_df["feature_name"], spearman_df["abs_rho_mean"])
    )

    # Iterative greedy removal
    removed = set()
    removed_records = []
    n_valid = len(valid_idx)

    while True:
        # Find all pairs exceeding threshold
        pairs = []
        for i in range(n_valid):
            if i in removed:
                continue
            for j in range(i + 1, n_valid):
                if j in removed:
                    continue
                r = abs(corr_matrix[i, j])
                if r > threshold:
                    pairs.append((i, j, r))

        if not pairs:
            break

        # Sort by descending |r|
        pairs.sort(key=lambda x: -x[2])

        # Remove one from the top pair
        i, j, r = pairs[0]
        rho_i = spearman_lookup.get(names_valid[i], 0.0)
        rho_j = spearman_lookup.get(names_valid[j], 0.0)

        if rho_i >= rho_j:
            to_remove = j
            kept = i
        else:
            to_remove = i
            kept = j

        removed.add(to_remove)
        removed_records.append({
            "removed_feature": names_valid[to_remove],
            "kept_feature": names_valid[kept],
            "pearson_r": float(r),
            "removed_abs_rho": float(spearman_lookup.get(names_valid[to_remove], 0.0)),
            "kept_abs_rho": float(spearman_lookup.get(names_valid[kept], 0.0)),
        })

    # Save removed features
    removed_df = pd.DataFrame(removed_records)
    removed_path = os.path.join(output_dir, "removed_redundant_features.csv")
    removed_df.to_csv(removed_path, index=False, encoding="utf-8")

    # Build reduced set
    kept_local = sorted(set(range(n_valid)) - removed)
    kept_indices = valid_idx[kept_local]
    X_reduced = X[:, kept_indices]
    names_reduced = [names[i] for i in kept_indices]

    logger.info(
        "Redundancy removal: %d -> %d features (%d removed at threshold %.2f)",
        len(valid_idx), len(kept_indices), len(removed), threshold,
    )

    return X_reduced, names_reduced, kept_indices


def rf_importance_pruning(
    X: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    output_dir: str,
    n_estimators: int = 500,
    importance_threshold_frac: float = 0.005,
    max_iterations: int = 20,
    n_selected_range: Tuple[int, int] = (30, 250),
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict]:
    """Iterative RF importance pruning.

    Returns:
        X_selected: (N, N_selected)
        names_selected: list of N_selected names
        kept_indices: original column indices (relative to X input)
        summary: dict with OOB score, iterations, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    current_X = X.copy()
    current_names = list(names)
    current_idx = np.arange(X.shape[1])
    all_importance_records = []

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "RF iteration %d: %d features", iteration, current_X.shape[1]
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True,
        )
        rf.fit(current_X, labels)
        oob_acc = rf.oob_score_
        importances = rf.feature_importances_

        logger.info(
            "  OOB accuracy: %.4f, max importance: %.6f",
            oob_acc, importances.max(),
        )

        # Record importances for this iteration
        for i, name in enumerate(current_names):
            all_importance_records.append({
                "iteration": iteration,
                "feature_name": name,
                "importance": float(importances[i]),
                "oob_accuracy": float(oob_acc),
            })

        # Find features to prune
        max_imp = importances.max()
        threshold = max_imp * importance_threshold_frac
        keep_mask = importances >= threshold
        n_keep = keep_mask.sum()

        if n_keep == current_X.shape[1]:
            logger.info("  Stable: no features below threshold. Done.")
            break

        n_prune = current_X.shape[1] - n_keep
        logger.info(
            "  Pruning %d features below %.6f (0.5%% of max)",
            n_prune, threshold,
        )

        current_X = current_X[:, keep_mask]
        current_names = [n for n, k in zip(current_names, keep_mask) if k]
        current_idx = current_idx[keep_mask]

    # Final RF for clean importances
    rf_final = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    rf_final.fit(current_X, labels)
    final_importances = rf_final.feature_importances_
    final_oob = rf_final.oob_score_

    n_selected = len(current_names)
    lo, hi = n_selected_range
    if n_selected < lo or n_selected > hi:
        logger.warning(
            "N_selected=%d outside expected range [%d, %d]", n_selected, lo, hi
        )

    # Save importance rankings
    importance_df = pd.DataFrame({
        "feature_name": current_names,
        "importance": final_importances,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(
        os.path.join(output_dir, "rf_importance_rankings.csv"),
        index=False, encoding="utf-8",
    )

    # Save selected features
    selected_path = os.path.join(output_dir, "selected_features.txt")
    with open(selected_path, "w", encoding="utf-8") as f:
        for name in current_names:
            f.write(name + "\n")

    # Save iteration history
    history_df = pd.DataFrame(all_importance_records)
    history_df.to_csv(
        os.path.join(output_dir, "rf_pruning_history.csv"),
        index=False, encoding="utf-8",
    )

    # Summary YAML
    summary = {
        "n_candidates": int(X.shape[1]),
        "n_selected": n_selected,
        "n_iterations": iteration,
        "final_oob_accuracy": float(final_oob),
        "importance_threshold_frac": float(importance_threshold_frac),
        "n_estimators": n_estimators,
        "top_10_features": [
            {"name": row["feature_name"], "importance": round(row["importance"], 6)}
            for _, row in importance_df.head(10).iterrows()
        ],
    }
    summary_path = os.path.join(output_dir, "feature_selection_summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    logger.info(
        "RF pruning complete: %d -> %d features, OOB=%.4f",
        X.shape[1], n_selected, final_oob,
    )

    return current_X, current_names, current_idx, summary


def erc_feature_evaluation(
    X_selected: np.ndarray,
    names_selected: List[str],
    training_csv: str,
    output_dir: str,
) -> pd.DataFrame:
    """Evaluate each selected feature via Error-Reject Curve (ERC).

    For each feature, normalize to [0,100], then measure how well rejecting
    low-quality samples reduces FNMR at fixed thresholds.

    Returns DataFrame with AUC values per feature per provider.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(training_csv)
    providers = ["P1", "P2", "P3"]
    score_cols = [f"score_{p}" for p in providers]

    # Genuine scores per provider (Class 1 samples)
    genuine_mask = train_df["label"].values == 1

    records = []
    for feat_idx, feat_name in enumerate(names_selected):
        x = X_selected[:, feat_idx]

        # Normalize to [0, 100]
        x_min, x_max = x.min(), x.max()
        rng = x_max - x_min
        if rng < 1e-12:
            q = np.full_like(x, 50.0)
        else:
            q = 100.0 * (x - x_min) / rng

        row = {"feature_name": feat_name}

        for p, score_col in zip(providers, score_cols):
            scores = train_df[score_col].values

            # Compute FNMR at reference thresholds
            # Use P10 and P90 as FNMR targets
            genuine_scores = scores[genuine_mask]
            fnmr_target_10 = np.percentile(genuine_scores, 10)  # ~10% FNMR
            fnmr_target_1 = np.percentile(genuine_scores, 1)    # ~1% FNMR

            for fnmr_label, tau in [("fnmr10", fnmr_target_10), ("fnmr1", fnmr_target_1)]:
                # Sweep quality thresholds
                quality_thresholds = np.linspace(0, 100, 101)
                fnmr_values = []
                rejection_rates = []

                for qt in quality_thresholds:
                    # Accept only samples with quality >= qt
                    accept_mask = q >= qt
                    n_accepted = accept_mask.sum()
                    if n_accepted < 10:
                        fnmr_values.append(1.0)
                        rejection_rates.append(1.0)
                        continue

                    accepted_genuine = accept_mask & genuine_mask
                    n_genuine_accepted = accepted_genuine.sum()
                    if n_genuine_accepted < 5:
                        fnmr_values.append(1.0)
                        rejection_rates.append(float(1 - n_accepted / len(q)))
                        continue

                    # FNMR among accepted genuine pairs
                    accepted_scores = scores[accepted_genuine]
                    fnmr = np.mean(accepted_scores < tau)
                    fnmr_values.append(float(fnmr))
                    rejection_rates.append(float(1 - n_accepted / len(q)))

                # AUC of FNMR vs rejection rate
                rejection_rates = np.array(rejection_rates)
                fnmr_values = np.array(fnmr_values)
                # Sort by rejection rate
                sort_idx = np.argsort(rejection_rates)
                auc = float(np.trapezoid(fnmr_values[sort_idx], rejection_rates[sort_idx]))
                # Normalize: ideal = 0 (immediate drop to 0 FNMR), worst = area under constant FNMR
                # Lower AUC is better (quality feature reduces FNMR faster)
                row[f"auc_{fnmr_label}_{p}"] = auc

        # Mean AUC across providers and targets
        auc_vals = [v for k, v in row.items() if k.startswith("auc_")]
        row["auc_mean"] = float(np.mean(auc_vals)) if auc_vals else 1.0

        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("auc_mean")
    out_path = os.path.join(output_dir, "erc_per_feature.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("ERC evaluation saved to %s (%d features)", out_path, len(df))

    return df


def _run_selection_pipeline(
    score_type: str,
    features_path: str,
    names_path: str,
    training_csv: str,
    fisher_csv: str,
    output_dir: str,
    redundancy_threshold: float = 0.95,
    importance_threshold_frac: float = 0.005,
    n_selected_range: Tuple[int, int] = (30, 250),
    checkpoint_path: Optional[str] = None,
    completed_stages: Optional[set] = None,
) -> Dict:
    """Run the full 3-stage selection pipeline.

    Args:
        score_type: "s" or "v"
        checkpoint_path: path to checkpoint YAML (if resuming)
        completed_stages: set of already-completed stage names

    Returns dict with all results.
    """
    if completed_stages is None:
        completed_stages = set()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Stage 0: Load
    stage_name = f"load_{score_type}"
    logger.info("=== Pipeline %s: Loading data ===", score_type.upper())
    X, feature_names, labels, fisher_df, valid_mask = load_candidate_features(
        features_path, names_path, training_csv, fisher_csv,
    )
    results["n_total"] = X.shape[1]
    results["n_valid"] = int(valid_mask.sum())
    results["n_samples"] = X.shape[0]

    # Stage 1: Spearman
    stage_name = f"spearman_{score_type}"
    if stage_name in completed_stages:
        logger.info("=== Skipping Spearman (already complete) ===")
        spearman_df = pd.read_csv(
            os.path.join(output_dir, "spearman_correlations.csv")
        )
    else:
        logger.info("=== Pipeline %s: Spearman correlation ===", score_type.upper())
        spearman_df = spearman_evaluation(X, feature_names, fisher_df, valid_mask, output_dir)
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, completed_stages | {stage_name})

    results["spearman_df"] = spearman_df

    # Stage 2: Redundancy removal
    stage_name = f"redundancy_{score_type}"
    if stage_name in completed_stages:
        logger.info("=== Skipping redundancy removal (already complete) ===")
        # Reload from saved outputs
        removed_df = pd.read_csv(
            os.path.join(output_dir, "removed_redundant_features.csv")
        )
        removed_names = set(removed_df["removed_feature"].tolist()) if len(removed_df) > 0 else set()
        valid_idx = np.where(valid_mask)[0]
        kept_local = [
            i for i, idx in enumerate(valid_idx)
            if feature_names[idx] not in removed_names
        ]
        kept_indices = valid_idx[np.array(kept_local)]
        X_reduced = X[:, kept_indices]
        names_reduced = [feature_names[i] for i in kept_indices]
    else:
        logger.info("=== Pipeline %s: Redundancy removal ===", score_type.upper())
        X_reduced, names_reduced, kept_indices = remove_redundant_features(
            X, feature_names, spearman_df, valid_mask, output_dir, redundancy_threshold,
        )
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, completed_stages | {stage_name})

    results["n_after_redundancy"] = len(names_reduced)

    # Stage 3: RF importance pruning
    stage_name = f"rf_pruning_{score_type}"
    if stage_name in completed_stages:
        logger.info("=== Skipping RF pruning (already complete) ===")
        selected_path = os.path.join(output_dir, "selected_features.txt")
        with open(selected_path, "r", encoding="utf-8") as f:
            names_selected = [line.strip() for line in f if line.strip()]
        sel_mask = np.array([n in set(names_selected) for n in names_reduced])
        X_selected = X_reduced[:, sel_mask]
        summary_path = os.path.join(output_dir, "feature_selection_summary.yaml")
        with open(summary_path, "r", encoding="utf-8") as f:
            rf_summary = yaml.safe_load(f)
    else:
        logger.info("=== Pipeline %s: RF importance pruning ===", score_type.upper())
        X_selected, names_selected, _, rf_summary = rf_importance_pruning(
            X_reduced, labels, names_reduced, output_dir,
            n_selected_range=n_selected_range,
            importance_threshold_frac=importance_threshold_frac,
        )
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, completed_stages | {stage_name})

    results["n_selected"] = len(names_selected)
    results["names_selected"] = names_selected
    results["rf_summary"] = rf_summary

    # Stage 4: ERC evaluation
    stage_name = f"erc_{score_type}"
    if stage_name in completed_stages:
        logger.info("=== Skipping ERC (already complete) ===")
        erc_df = pd.read_csv(os.path.join(output_dir, "erc_per_feature.csv"))
    else:
        logger.info("=== Pipeline %s: ERC evaluation ===", score_type.upper())
        erc_df = erc_feature_evaluation(
            X_selected, names_selected, training_csv, output_dir,
        )
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, completed_stages | {stage_name})

    results["erc_df"] = erc_df

    logger.info(
        "=== Pipeline %s COMPLETE: %d -> %d -> %d -> %d features ===",
        score_type.upper(),
        results["n_total"], results["n_valid"],
        results["n_after_redundancy"], results["n_selected"],
    )

    return results


def _save_checkpoint(checkpoint_path: str, completed_stages: set):
    """Save checkpoint with completed stages."""
    data = {
        "completed_stages": sorted(completed_stages),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


def _load_checkpoint(checkpoint_path: str) -> set:
    """Load checkpoint, return set of completed stage names."""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data and "completed_stages" in data:
        return set(data["completed_stages"])
    return set()


# ---- VQI-S public API ----

def run_vqi_s_pipeline(
    features_path: str,
    names_path: str,
    training_csv: str,
    fisher_csv: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict:
    """Run the full VQI-S (Signal Quality) feature selection pipeline.

    544 candidates -> N_selected features.
    """
    completed = set()
    if resume and checkpoint_path:
        completed = _load_checkpoint(checkpoint_path)
        if completed:
            logger.info("Resuming VQI-S: completed stages = %s", completed)

    return _run_selection_pipeline(
        score_type="s",
        features_path=features_path,
        names_path=names_path,
        training_csv=training_csv,
        fisher_csv=fisher_csv,
        output_dir=output_dir,
        redundancy_threshold=0.95,
        importance_threshold_frac=0.005,
        n_selected_range=(30, 250),
        checkpoint_path=checkpoint_path,
        completed_stages=completed,
    )
