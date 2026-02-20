"""
Step 7.16: VQI-S x VQI-V Feature-Level Cross-Analysis

Six experiments answering: "Do we need both VQI-S (430 features) and VQI-V (133 features)?"

  A. Combined Model Performance — RF on 563 combined features vs S-only/V-only
  B. Cross-Correlation Matrix — Spearman between all S and V features
  C. Feature Importance Redistribution — which features dominate when combined?
  D. Ablation / Unique Contribution — block permutation + incremental features
  E. Cross-Prediction — can S features replicate V model decisions (and vice versa)?
  F. Validation Set Comparison — combined vs S-only vs V-only on held-out 50K
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from vqi.training.train_rf import FIXED_PARAMS

logger = logging.getLogger(__name__)

# Grid for combined model search (plan spec)
COMBINED_N_EST_GRID = [500, 750, 1000]
COMBINED_MAX_FEAT_GRID = [5, 8, 10, 12, 15, "sqrt"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_training_data(
    x_s_path: str,
    x_v_path: str,
    y_path: str,
    s_names_path: str,
    v_names_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Load S and V training arrays plus labels."""
    X_s = np.load(x_s_path)
    X_v = np.load(x_v_path)
    y = np.load(y_path)
    with open(s_names_path, "r", encoding="utf-8") as f:
        s_names = [line.strip() for line in f if line.strip()]
    with open(v_names_path, "r", encoding="utf-8") as f:
        v_names = [line.strip() for line in f if line.strip()]
    assert X_s.shape[0] == X_v.shape[0] == len(y), "Row count mismatch"
    assert X_s.shape[1] == len(s_names), f"S cols {X_s.shape[1]} != names {len(s_names)}"
    assert X_v.shape[1] == len(v_names), f"V cols {X_v.shape[1]} != names {len(v_names)}"
    return X_s, X_v, y, np.hstack([X_s, X_v]), s_names, v_names


def _save_yaml(data: dict, path: str):
    """Write dict to YAML with utf-8."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# A. Combined Model Performance
# ---------------------------------------------------------------------------

def experiment_a_combined_model(
    X_s: np.ndarray,
    X_v: np.ndarray,
    y: np.ndarray,
    s_names: List[str],
    v_names: List[str],
    output_dir: str,
    model_s_path: str,
    model_v_path: str,
) -> Dict:
    """Train RF on 563 combined features. Compare with S-only and V-only via 5-fold CV.

    Returns dict with cv_comparison rows and combined model.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_combined = np.hstack([X_s, X_v])
    combined_names = s_names + v_names
    n_s, n_v = X_s.shape[1], X_v.shape[1]
    logger.info(
        "[Exp A] Combined features: %d S + %d V = %d total",
        n_s, n_v, X_combined.shape[1],
    )

    # --- Grid search for combined model (OOB) ---
    logger.info("[Exp A] Grid search for combined model...")
    best_oob = 0.0
    best_params = {}
    grid_rows = []
    for n_est in COMBINED_N_EST_GRID:
        for max_feat in COMBINED_MAX_FEAT_GRID:
            params = {**FIXED_PARAMS, "n_estimators": n_est, "max_features": max_feat}
            clf = RandomForestClassifier(**params)
            clf.fit(X_combined, y)
            oob = clf.oob_score_
            grid_rows.append({
                "n_estimators": n_est,
                "max_features": str(max_feat),
                "oob_accuracy": round(oob, 6),
            })
            if oob > best_oob:
                best_oob = oob
                best_params = {"n_estimators": n_est, "max_features": max_feat}
            logger.info(
                "  n_est=%d, max_feat=%s -> OOB=%.4f", n_est, max_feat, oob,
            )

    grid_df = pd.DataFrame(grid_rows).sort_values("oob_accuracy", ascending=False)
    grid_df.to_csv(os.path.join(output_dir, "combined_grid_search.csv"), index=False)

    # --- Train final combined model ---
    logger.info("[Exp A] Training final combined model with %s...", best_params)
    params = {**FIXED_PARAMS, **best_params}
    clf_combined = RandomForestClassifier(**params)
    clf_combined.fit(X_combined, y)
    combined_oob = clf_combined.oob_score_
    combined_train_acc = clf_combined.score(X_combined, y)

    combined_model_path = os.path.join(
        os.path.dirname(model_s_path), "vqi_combined_rf_model.joblib"
    )
    joblib.dump(clf_combined, combined_model_path)
    logger.info("[Exp A] Combined model saved: %s", combined_model_path)

    # --- 5-fold CV for all 3 configs ---
    logger.info("[Exp A] Running 5-fold stratified CV for S-only, V-only, combined...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_s, y))

    cv_scores = {"S-only": [], "V-only": [], "Combined": []}
    datasets = {"S-only": X_s, "V-only": X_v, "Combined": X_combined}

    # Use S-only and V-only models' params for their respective CV
    clf_s = joblib.load(model_s_path)
    clf_v = joblib.load(model_v_path)
    model_params = {
        "S-only": {
            **FIXED_PARAMS,
            "n_estimators": clf_s.n_estimators,
            "max_features": clf_s.max_features,
        },
        "V-only": {
            **FIXED_PARAMS,
            "n_estimators": clf_v.n_estimators,
            "max_features": clf_v.max_features,
        },
        "Combined": params,
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        for config_name, X_data in datasets.items():
            rf = RandomForestClassifier(**model_params[config_name])
            rf.fit(X_data[train_idx], y[train_idx])
            acc = accuracy_score(y[val_idx], rf.predict(X_data[val_idx]))
            cv_scores[config_name].append(acc)
        logger.info(
            "  Fold %d: S=%.4f, V=%.4f, C=%.4f",
            fold_idx + 1,
            cv_scores["S-only"][-1],
            cv_scores["V-only"][-1],
            cv_scores["Combined"][-1],
        )

    # --- Paired t-tests with Bonferroni correction ---
    alpha = 0.05 / 3  # 0.0167
    comparisons = [
        ("Combined vs S-only", cv_scores["Combined"], cv_scores["S-only"]),
        ("Combined vs V-only", cv_scores["Combined"], cv_scores["V-only"]),
        ("S-only vs V-only", cv_scores["S-only"], cv_scores["V-only"]),
    ]
    ttest_results = []
    for name, a, b in comparisons:
        t_stat, p_val = stats.ttest_rel(a, b)
        sig = p_val < alpha
        ttest_results.append({
            "comparison": name,
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_bonferroni": bool(sig),
        })
        logger.info(
            "  %s: t=%.4f, p=%.6f, sig=%s", name, t_stat, p_val, sig,
        )

    # --- Build comparison table ---
    cv_rows = []
    for config_name in ["S-only", "V-only", "Combined"]:
        scores = cv_scores[config_name]
        cv_rows.append({
            "config": config_name,
            "n_features": datasets[config_name].shape[1],
            "cv_mean": round(float(np.mean(scores)), 6),
            "cv_std": round(float(np.std(scores)), 6),
            "cv_folds": [round(s, 6) for s in scores],
        })

    cv_df = pd.DataFrame([{
        "config": r["config"],
        "n_features": r["n_features"],
        "cv_mean": r["cv_mean"],
        "cv_std": r["cv_std"],
    } for r in cv_rows])
    cv_df.to_csv(os.path.join(output_dir, "cv_comparison.csv"), index=False)

    # --- Save metrics ---
    metrics = {
        "combined_best_params": {
            "n_estimators": int(best_params["n_estimators"]),
            "max_features": best_params["max_features"],
        },
        "combined_oob_accuracy": round(float(combined_oob), 6),
        "combined_train_accuracy": round(float(combined_train_acc), 6),
        "combined_n_features": int(X_combined.shape[1]),
        "cv_results": cv_rows,
        "paired_ttests": ttest_results,
        "bonferroni_alpha": round(alpha, 4),
    }
    _save_yaml(metrics, os.path.join(output_dir, "combined_training_metrics.yaml"))

    logger.info(
        "[Exp A] Done. S CV=%.4f, V CV=%.4f, Combined CV=%.4f",
        np.mean(cv_scores["S-only"]),
        np.mean(cv_scores["V-only"]),
        np.mean(cv_scores["Combined"]),
    )
    return {
        "clf_combined": clf_combined,
        "combined_names": combined_names,
        "cv_scores": cv_scores,
        "metrics": metrics,
        "model_path": combined_model_path,
    }


# ---------------------------------------------------------------------------
# B. Cross-Correlation Matrix
# ---------------------------------------------------------------------------

def experiment_b_cross_correlation(
    X_s: np.ndarray,
    X_v: np.ndarray,
    s_names: List[str],
    v_names: List[str],
    output_dir: str,
) -> Dict:
    """Full Spearman correlation between all S and V features.

    Returns dict with correlation matrix and summary stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_s, n_v = X_s.shape[1], X_v.shape[1]
    logger.info("[Exp B] Computing %d x %d Spearman cross-correlations...", n_s, n_v)

    # Compute column-by-column Spearman
    corr_matrix = np.zeros((n_s, n_v), dtype=np.float32)
    for j in range(n_v):
        for i in range(n_s):
            rho, _ = stats.spearmanr(X_s[:, i], X_v[:, j])
            corr_matrix[i, j] = rho
        if (j + 1) % 20 == 0:
            logger.info("  Computed %d/%d V columns", j + 1, n_v)

    # Sanity: no NaN (would only happen if a column is constant)
    nan_count = np.isnan(corr_matrix).sum()
    if nan_count > 0:
        logger.warning("[Exp B] %d NaN correlations (constant columns?), replacing with 0", nan_count)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    np.save(os.path.join(output_dir, "cross_corr_spearman.npy"), corr_matrix)
    logger.info("[Exp B] Saved cross_corr_spearman.npy shape=%s", corr_matrix.shape)

    # Summary stats
    abs_corr = np.abs(corr_matrix)
    thresholds = [0.3, 0.5, 0.7]
    frac_above = {}
    for t in thresholds:
        frac = float((abs_corr > t).sum()) / abs_corr.size
        frac_above[f"frac_above_{t}"] = round(frac, 6)
        logger.info("  |rho| > %.1f: %.2f%% of pairs", t, frac * 100)

    # Top-50 most correlated cross-pairs
    pairs = []
    for i in range(n_s):
        for j in range(n_v):
            pairs.append({
                "s_feature": s_names[i],
                "v_feature": v_names[j],
                "spearman_rho": round(float(corr_matrix[i, j]), 6),
                "abs_rho": round(float(abs_corr[i, j]), 6),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False)
    top50 = pairs_df.head(50)
    top50.to_csv(os.path.join(output_dir, "top_correlated_pairs.csv"), index=False)

    summary = {
        "matrix_shape": [n_s, n_v],
        "total_pairs": n_s * n_v,
        "mean_abs_rho": round(float(abs_corr.mean()), 6),
        "max_abs_rho": round(float(abs_corr.max()), 6),
        "median_abs_rho": round(float(np.median(abs_corr)), 6),
        **frac_above,
        "nan_count": int(nan_count),
        "top_5_pairs": top50.head(5).to_dict("records"),
    }
    _save_yaml(summary, os.path.join(output_dir, "cross_corr_summary.yaml"))

    logger.info(
        "[Exp B] Done. Mean |rho|=%.4f, max |rho|=%.4f",
        abs_corr.mean(), abs_corr.max(),
    )
    return {"corr_matrix": corr_matrix, "summary": summary, "top_pairs": top50}


# ---------------------------------------------------------------------------
# C. Feature Importance Redistribution
# ---------------------------------------------------------------------------

def experiment_c_importance_redistribution(
    clf_combined: RandomForestClassifier,
    combined_names: List[str],
    s_names: List[str],
    v_names: List[str],
    imp_s_csv: str,
    imp_v_csv: str,
    output_dir: str,
) -> Dict:
    """Analyze how feature importances shift in the combined model.

    Returns dict with redistribution analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_s = len(s_names)
    n_v = len(v_names)
    logger.info("[Exp C] Analyzing feature importance redistribution...")

    # Combined importances
    combined_imp = clf_combined.feature_importances_
    assert len(combined_imp) == len(combined_names)

    combined_df = pd.DataFrame({
        "feature": combined_names,
        "combined_importance": combined_imp,
        "origin": ["S"] * n_s + ["V"] * n_v,
    }).sort_values("combined_importance", ascending=False).reset_index(drop=True)
    combined_df["combined_rank"] = range(1, len(combined_df) + 1)

    # Solo importances
    imp_s = pd.read_csv(imp_s_csv)
    imp_v = pd.read_csv(imp_v_csv)
    solo_imp_map = {}
    solo_rank_map = {}
    for _, row in imp_s.iterrows():
        solo_imp_map[row["feature"]] = row["importance"]
        solo_rank_map[row["feature"]] = int(row["rank"])
    for _, row in imp_v.iterrows():
        solo_imp_map[row["feature"]] = row["importance"]
        solo_rank_map[row["feature"]] = int(row["rank"])

    combined_df["solo_importance"] = combined_df["feature"].map(solo_imp_map)
    combined_df["solo_rank"] = combined_df["feature"].map(solo_rank_map)
    combined_df["rank_shift"] = combined_df["solo_rank"] - combined_df["combined_rank"]

    combined_df.to_csv(
        os.path.join(output_dir, "combined_feature_importances.csv"), index=False,
    )

    # Aggregate: total importance share by origin
    s_share = float(combined_df[combined_df["origin"] == "S"]["combined_importance"].sum())
    v_share = float(combined_df[combined_df["origin"] == "V"]["combined_importance"].sum())
    total = s_share + v_share

    # Top-30 feature origin breakdown
    top30 = combined_df.head(30)
    n_s_top30 = int((top30["origin"] == "S").sum())
    n_v_top30 = int((top30["origin"] == "V").sum())

    redistribution = {
        "s_total_importance": round(s_share, 6),
        "v_total_importance": round(v_share, 6),
        "s_share_pct": round(s_share / total * 100, 2),
        "v_share_pct": round(v_share / total * 100, 2),
        "s_features_in_top30": n_s_top30,
        "v_features_in_top30": n_v_top30,
        "top10_features": top30.head(10)[["feature", "combined_importance", "origin", "combined_rank", "solo_rank"]].to_dict("records"),
        "mean_rank_shift_s": round(float(combined_df[combined_df["origin"] == "S"]["rank_shift"].mean()), 2),
        "mean_rank_shift_v": round(float(combined_df[combined_df["origin"] == "V"]["rank_shift"].mean()), 2),
    }
    _save_yaml(redistribution, os.path.join(output_dir, "importance_redistribution.yaml"))

    logger.info(
        "[Exp C] Done. S share=%.1f%%, V share=%.1f%%, top-30: %dS/%dV",
        redistribution["s_share_pct"], redistribution["v_share_pct"],
        n_s_top30, n_v_top30,
    )
    return {"combined_df": combined_df, "redistribution": redistribution}


# ---------------------------------------------------------------------------
# D. Ablation / Unique Contribution
# ---------------------------------------------------------------------------

def experiment_d_ablation(
    X_s: np.ndarray,
    X_v: np.ndarray,
    y: np.ndarray,
    clf_combined: RandomForestClassifier,
    combined_names: List[str],
    s_names: List[str],
    v_names: List[str],
    output_dir: str,
    n_permutation_seeds: int = 10,
) -> Dict:
    """Block permutation + incremental feature analysis.

    Returns dict with ablation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_combined = np.hstack([X_s, X_v])
    n_s = X_s.shape[1]
    n_v = X_v.shape[1]
    n_total = n_s + n_v

    # --- Method 1: Block permutation ---
    logger.info("[Exp D] Block permutation (S and V blocks, %d seeds)...", n_permutation_seeds)
    baseline_acc = accuracy_score(y, clf_combined.predict(X_combined))
    logger.info("  Baseline accuracy: %.4f", baseline_acc)

    perm_rows = []
    for seed in range(n_permutation_seeds):
        rng = np.random.RandomState(seed)

        # Permute S block (columns 0..n_s-1)
        X_perm_s = X_combined.copy()
        perm_idx = rng.permutation(len(y))
        X_perm_s[:, :n_s] = X_perm_s[perm_idx, :n_s]
        acc_perm_s = accuracy_score(y, clf_combined.predict(X_perm_s))

        # Permute V block (columns n_s..n_total-1)
        X_perm_v = X_combined.copy()
        perm_idx = rng.permutation(len(y))
        X_perm_v[:, n_s:] = X_perm_v[perm_idx, n_s:]
        acc_perm_v = accuracy_score(y, clf_combined.predict(X_perm_v))

        perm_rows.append({
            "seed": seed,
            "baseline_accuracy": round(baseline_acc, 6),
            "acc_permute_s": round(float(acc_perm_s), 6),
            "acc_permute_v": round(float(acc_perm_v), 6),
            "drop_permute_s": round(baseline_acc - float(acc_perm_s), 6),
            "drop_permute_v": round(baseline_acc - float(acc_perm_v), 6),
        })

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(os.path.join(output_dir, "block_permutation.csv"), index=False)

    mean_drop_s = float(perm_df["drop_permute_s"].mean())
    mean_drop_v = float(perm_df["drop_permute_v"].mean())
    std_drop_s = float(perm_df["drop_permute_s"].std())
    std_drop_v = float(perm_df["drop_permute_v"].std())
    logger.info(
        "  Mean drop when permuting S: %.4f +/- %.4f",
        mean_drop_s, std_drop_s,
    )
    logger.info(
        "  Mean drop when permuting V: %.4f +/- %.4f",
        mean_drop_v, std_drop_v,
    )

    # --- Method 2: Incremental features ---
    logger.info("[Exp D] Incremental features by combined importance...")
    combined_imp = clf_combined.feature_importances_
    sorted_idx = np.argsort(combined_imp)[::-1]
    origins = np.array(["S"] * n_s + ["V"] * n_v)

    k_values = [10, 20, 50, 100, 150, 200, 300, 430, 500, n_total]
    incr_rows = []

    for k in k_values:
        if k > n_total:
            continue
        top_k_idx = sorted_idx[:k]
        X_k = X_combined[:, top_k_idx]
        origins_k = origins[top_k_idx]
        n_s_k = int((origins_k == "S").sum())
        n_v_k = int((origins_k == "V").sum())

        rf_k = RandomForestClassifier(
            **FIXED_PARAMS,
            n_estimators=500,
            max_features=min(k, 10),
        )
        rf_k.fit(X_k, y)
        oob_acc = rf_k.oob_score_

        incr_rows.append({
            "k": k,
            "oob_accuracy": round(float(oob_acc), 6),
            "n_s_features": n_s_k,
            "n_v_features": n_v_k,
            "s_fraction": round(n_s_k / k, 4),
            "v_fraction": round(n_v_k / k, 4),
        })
        logger.info(
            "  k=%d: OOB=%.4f, S=%d (%.0f%%), V=%d (%.0f%%)",
            k, oob_acc, n_s_k, n_s_k / k * 100, n_v_k, n_v_k / k * 100,
        )

    incr_df = pd.DataFrame(incr_rows)
    incr_df.to_csv(os.path.join(output_dir, "incremental_features.csv"), index=False)

    ablation_results = {
        "block_permutation": {
            "baseline_accuracy": round(baseline_acc, 6),
            "mean_drop_permute_s": round(mean_drop_s, 6),
            "std_drop_permute_s": round(std_drop_s, 6),
            "mean_drop_permute_v": round(mean_drop_v, 6),
            "std_drop_permute_v": round(std_drop_v, 6),
            "s_unique_pct": round(mean_drop_s / baseline_acc * 100, 2),
            "v_unique_pct": round(mean_drop_v / baseline_acc * 100, 2),
        },
        "incremental_features": incr_rows,
    }
    _save_yaml(ablation_results, os.path.join(output_dir, "ablation_results.yaml"))

    logger.info("[Exp D] Done.")
    return {
        "perm_df": perm_df,
        "incr_df": incr_df,
        "results": ablation_results,
    }


# ---------------------------------------------------------------------------
# E. Cross-Prediction
# ---------------------------------------------------------------------------

def experiment_e_cross_prediction(
    X_s: np.ndarray,
    X_v: np.ndarray,
    y: np.ndarray,
    model_s_path: str,
    model_v_path: str,
    output_dir: str,
) -> Dict:
    """Can S features replicate V model's decisions (and vice versa)?

    Returns dict with cross-prediction results.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("[Exp E] Cross-prediction analysis...")

    clf_s = joblib.load(model_s_path)
    clf_v = joblib.load(model_v_path)

    # Predictions from each model
    preds_s = clf_s.predict(X_s)
    preds_v = clf_v.predict(X_v)
    proba_s = clf_s.predict_proba(X_s)[:, 1]
    proba_v = clf_v.predict_proba(X_v)[:, 1]

    # Agreement between S and V models on true labels
    kappa = cohen_kappa_score(preds_s, preds_v)
    agreement_rate = float(np.mean(preds_s == preds_v))
    proba_corr, proba_pval = stats.spearmanr(proba_s, proba_v)
    logger.info(
        "  S-V agreement: %.4f, kappa=%.4f, proba rho=%.4f (p=%.2e)",
        agreement_rate, kappa, proba_corr, proba_pval,
    )

    # Train RF: X=S features, y=V model predictions
    logger.info("  Training S -> V cross-predictor...")
    rf_s2v = RandomForestClassifier(**FIXED_PARAMS, n_estimators=500, max_features=10)
    rf_s2v.fit(X_s, preds_v)
    s2v_oob = rf_s2v.oob_score_
    s2v_acc = accuracy_score(preds_v, rf_s2v.predict(X_s))
    logger.info("  S -> V: OOB=%.4f, train_acc=%.4f", s2v_oob, s2v_acc)

    # Train RF: X=V features, y=S model predictions
    logger.info("  Training V -> S cross-predictor...")
    rf_v2s = RandomForestClassifier(**FIXED_PARAMS, n_estimators=500, max_features=5)
    rf_v2s.fit(X_v, preds_s)
    v2s_oob = rf_v2s.oob_score_
    v2s_acc = accuracy_score(preds_s, rf_v2s.predict(X_v))
    logger.info("  V -> S: OOB=%.4f, train_acc=%.4f", v2s_oob, v2s_acc)

    # Agreement table
    agree_df = pd.DataFrame({
        "metric": [
            "agreement_rate",
            "cohen_kappa",
            "spearman_proba_rho",
            "spearman_proba_pval",
            "s_to_v_oob_accuracy",
            "v_to_s_oob_accuracy",
        ],
        "value": [
            round(agreement_rate, 6),
            round(float(kappa), 6),
            round(float(proba_corr), 6),
            float(proba_pval),
            round(float(s2v_oob), 6),
            round(float(v2s_oob), 6),
        ],
    })
    agree_df.to_csv(os.path.join(output_dir, "agreement_table.csv"), index=False)

    cross_pred = {
        "agreement_rate": round(agreement_rate, 6),
        "cohen_kappa": round(float(kappa), 6),
        "spearman_proba_rho": round(float(proba_corr), 6),
        "spearman_proba_pval": float(proba_pval),
        "s_to_v_cross_prediction_oob": round(float(s2v_oob), 6),
        "v_to_s_cross_prediction_oob": round(float(v2s_oob), 6),
        "interpretation": (
            "High cross-prediction (>0.75) = redundancy; "
            "Low (<0.65) = independence"
        ),
    }
    _save_yaml(cross_pred, os.path.join(output_dir, "cross_prediction.yaml"))

    # Save probabilities for scatter plot
    np.savez_compressed(
        os.path.join(output_dir, "cross_prediction_probas.npz"),
        proba_s=proba_s,
        proba_v=proba_v,
        preds_s=preds_s,
        preds_v=preds_v,
        y_true=y,
    )

    logger.info("[Exp E] Done.")
    return {"cross_pred": cross_pred, "proba_s": proba_s, "proba_v": proba_v}


# ---------------------------------------------------------------------------
# F. Validation Set Comparison
# ---------------------------------------------------------------------------

def experiment_f_validation_comparison(
    features_s_val_npy: str,
    features_v_val_npy: str,
    feature_names_s_json: str,
    feature_names_v_json: str,
    selected_s_txt: str,
    selected_v_txt: str,
    model_s_path: str,
    model_v_path: str,
    combined_model_path: str,
    s_names: List[str],
    v_names: List[str],
    output_dir: str,
    thresholds_yaml: Optional[str] = None,
    validation_csv: Optional[str] = None,
    provider_scores_dir: Optional[str] = None,
) -> Dict:
    """Combined model vs S-only vs V-only on 50K held-out validation set.

    If thresholds/validation/provider paths provided, computes confusion metrics
    on labeled subset. Otherwise, compares score distributions only.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)
    logger.info("[Exp F] Validation set comparison...")

    # --- Load and select validation features ---
    X_full_s = np.load(features_s_val_npy)
    X_full_v = np.load(features_v_val_npy)

    with open(feature_names_s_json, "r", encoding="utf-8") as f:
        all_s_names = json.load(f)
    with open(feature_names_v_json, "r", encoding="utf-8") as f:
        all_v_names = json.load(f)

    with open(selected_s_txt, "r", encoding="utf-8") as f:
        sel_s = [l.strip() for l in f if l.strip()]
    with open(selected_v_txt, "r", encoding="utf-8") as f:
        sel_v = [l.strip() for l in f if l.strip()]

    # Map to indices
    s_name2idx = {n: i for i, n in enumerate(all_s_names)}
    v_name2idx = {n: i for i, n in enumerate(all_v_names)}
    s_idx = np.array([s_name2idx[n] for n in sel_s])
    v_idx = np.array([v_name2idx[n] for n in sel_v])

    X_val_s = X_full_s[:, s_idx].astype(np.float32)
    X_val_v = X_full_v[:, v_idx].astype(np.float32)
    X_val_combined = np.hstack([X_val_s, X_val_v])

    logger.info(
        "  Val features: S=%s, V=%s, Combined=%s",
        X_val_s.shape, X_val_v.shape, X_val_combined.shape,
    )

    # --- Load models and predict ---
    clf_s = joblib.load(model_s_path)
    clf_v = joblib.load(model_v_path)
    clf_c = joblib.load(combined_model_path)

    proba_s = clf_s.predict_proba(X_val_s)[:, 1]
    proba_v = clf_v.predict_proba(X_val_v)[:, 1]
    proba_c = clf_c.predict_proba(X_val_combined)[:, 1]

    scores_s = np.clip(np.round(proba_s * 100).astype(int), 0, 100)
    scores_v = np.clip(np.round(proba_v * 100).astype(int), 0, 100)
    scores_c = np.clip(np.round(proba_c * 100).astype(int), 0, 100)

    preds_s = (scores_s >= 50).astype(int)
    preds_v = (scores_v >= 50).astype(int)
    preds_c = (scores_c >= 50).astype(int)

    # --- Try to compute labels for confusion matrix ---
    val_labels = None
    if thresholds_yaml and validation_csv and provider_scores_dir:
        try:
            from vqi.training.validate_model import (
                merge_provider_scores,
                compute_validation_labels,
            )
            val_df = merge_provider_scores(validation_csv, provider_scores_dir)
            val_df = compute_validation_labels(val_df, thresholds_yaml)
            val_labels = val_df["label"].values
            n_labeled = int(np.isfinite(val_labels).sum())
            logger.info("  Validation labels: %d labeled out of %d", n_labeled, len(val_labels))
        except Exception as e:
            logger.warning("  Could not compute validation labels: %s", e)

    # --- Metrics on labeled subset (if available) ---
    metrics_by_config = {}
    if val_labels is not None:
        valid = np.isfinite(val_labels)
        y_true = val_labels[valid].astype(int)
        for name, scores, preds in [
            ("S-only", scores_s, preds_s),
            ("V-only", scores_v, preds_v),
            ("Combined", scores_c, preds_c),
        ]:
            y_sc = scores[valid]
            y_pr = preds[valid]
            acc = accuracy_score(y_true, y_pr)
            f1 = f1_score(y_true, y_pr, zero_division=0)
            auc = roc_auc_score(y_true, y_sc)
            metrics_by_config[name] = {
                "accuracy": round(float(acc), 6),
                "f1": round(float(f1), 6),
                "auc_roc": round(float(auc), 6),
                "n_labeled": int(valid.sum()),
            }
            logger.info(
                "  %s: acc=%.4f, F1=%.4f, AUC=%.4f",
                name, acc, f1, auc,
            )

        # McNemar's test: S vs Combined, V vs Combined
        mcnemar_results = []
        for name_a, preds_a, name_b, preds_b in [
            ("S-only", preds_s, "Combined", preds_c),
            ("V-only", preds_v, "Combined", preds_c),
            ("S-only", preds_s, "V-only", preds_v),
        ]:
            pa = preds_a[valid]
            pb = preds_b[valid]
            # 2x2 table: (both correct, a wrong b right, a right b wrong, both wrong)
            a_correct = pa == y_true
            b_correct = pb == y_true
            n01 = int((~a_correct & b_correct).sum())  # a wrong, b right
            n10 = int((a_correct & ~b_correct).sum())   # a right, b wrong
            # McNemar statistic with continuity correction
            if n01 + n10 == 0:
                chi2 = 0.0
                p_val = 1.0
            else:
                chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                p_val = float(stats.chi2.sf(chi2, df=1))
            mcnemar_results.append({
                "comparison": f"{name_a} vs {name_b}",
                "n01": n01,
                "n10": n10,
                "chi2": round(chi2, 4),
                "p_value": round(p_val, 6),
                "significant_005": bool(p_val < 0.05),
            })
            logger.info(
                "  McNemar %s vs %s: n01=%d, n10=%d, chi2=%.4f, p=%.6f",
                name_a, name_b, n01, n10, chi2, p_val,
            )
    else:
        mcnemar_results = []
        logger.info("  No validation labels available - skipping confusion metrics")

    # --- Disagreement profile ---
    # Samples where S and V disagree
    disagree_mask = preds_s != preds_v
    n_disagree = int(disagree_mask.sum())
    logger.info("  S/V disagreement: %d/%d (%.1f%%)", n_disagree, len(preds_s), n_disagree / len(preds_s) * 100)

    disagree_profiles = []
    if n_disagree > 0:
        # Z-score each feature column
        s_mean = X_val_s.mean(axis=0)
        s_std = X_val_s.std(axis=0)
        s_std[s_std == 0] = 1.0
        v_mean = X_val_v.mean(axis=0)
        v_std = X_val_v.std(axis=0)
        v_std[v_std == 0] = 1.0

        X_val_s_z = (X_val_s - s_mean) / s_std
        X_val_v_z = (X_val_v - v_mean) / v_std

        # S-high/V-low: S predicts 1, V predicts 0
        s_high_v_low = disagree_mask & (preds_s == 1) & (preds_v == 0)
        s_low_v_high = disagree_mask & (preds_s == 0) & (preds_v == 1)

        for group_name, group_mask in [("S_high_V_low", s_high_v_low), ("S_low_V_high", s_low_v_high)]:
            n_group = int(group_mask.sum())
            if n_group == 0:
                continue
            mean_s_z = X_val_s_z[group_mask].mean(axis=0)
            mean_v_z = X_val_v_z[group_mask].mean(axis=0)

            # Top features with largest absolute z-score deviation
            top_s_idx = np.argsort(np.abs(mean_s_z))[::-1][:10]
            top_v_idx = np.argsort(np.abs(mean_v_z))[::-1][:10]

            for idx in top_s_idx:
                disagree_profiles.append({
                    "group": group_name,
                    "feature": sel_s[idx],
                    "origin": "S",
                    "mean_z_score": round(float(mean_s_z[idx]), 4),
                    "n_samples": n_group,
                })
            for idx in top_v_idx:
                disagree_profiles.append({
                    "group": group_name,
                    "feature": sel_v[idx],
                    "origin": "V",
                    "mean_z_score": round(float(mean_v_z[idx]), 4),
                    "n_samples": n_group,
                })

    disagree_df = pd.DataFrame(disagree_profiles) if disagree_profiles else pd.DataFrame()
    if len(disagree_df) > 0:
        disagree_df.to_csv(os.path.join(output_dir, "disagreement_profiles.csv"), index=False)

    # --- Save comparison CSV ---
    val_comp_df = pd.DataFrame({
        "score_s": scores_s,
        "score_v": scores_v,
        "score_combined": scores_c,
        "pred_s": preds_s,
        "pred_v": preds_v,
        "pred_combined": preds_c,
    })
    if val_labels is not None:
        val_comp_df["label"] = val_labels
    val_comp_df.to_csv(
        os.path.join(output_dir, "validation_comparison.csv"), index=False,
    )

    # --- Save results ---
    results = {
        "n_validation": len(scores_s),
        "n_disagreements": n_disagree,
        "disagreement_rate": round(n_disagree / len(scores_s), 6),
        "metrics_by_config": metrics_by_config,
        "mcnemar_results": mcnemar_results,
    }
    if mcnemar_results:
        _save_yaml(
            {"mcnemar": mcnemar_results},
            os.path.join(output_dir, "mcnemar_results.yaml"),
        )

    logger.info("[Exp F] Done.")
    return results


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_verdict(
    exp_a_metrics: Dict,
    exp_d_results: Dict,
    exp_e_results: Dict,
) -> Dict:
    """Apply decision framework from the plan.

    Returns dict with verdict string and evidence summary.
    """
    cv_s = float(np.mean(exp_a_metrics["cv_results"][0]["cv_folds"]))
    cv_v = float(np.mean(exp_a_metrics["cv_results"][1]["cv_folds"]))
    cv_c = float(np.mean(exp_a_metrics["cv_results"][2]["cv_folds"]))
    max_solo = max(cv_s, cv_v)
    combined_gain = cv_c - max_solo

    block_perm = exp_d_results["block_permutation"]
    s_unique = block_perm["s_unique_pct"]
    v_unique = block_perm["v_unique_pct"]

    cross_pred = exp_e_results
    s2v_oob = cross_pred["s_to_v_cross_prediction_oob"]
    v2s_oob = cross_pred["v_to_s_cross_prediction_oob"]
    max_cross = max(s2v_oob, v2s_oob)

    # Decision framework
    if combined_gain <= 0.01 and max_cross > 0.75:
        verdict = "ONE_SUFFICES"
        explanation = (
            f"Combined accuracy gain ({combined_gain:.4f}) is marginal (<= 0.01) "
            f"and cross-prediction is high ({max_cross:.4f} > 0.75). "
            f"One feature set is sufficient -- keep the better one."
        )
    elif combined_gain > 0.01 and s_unique > 2.0 and v_unique > 2.0:
        verdict = "BOTH_NEEDED"
        explanation = (
            f"Combined accuracy gain ({combined_gain:.4f}) is meaningful (> 0.01) "
            f"and both blocks have >2% unique contribution "
            f"(S={s_unique:.1f}%, V={v_unique:.1f}%). "
            f"Both feature sets carry complementary information."
        )
    elif combined_gain > 0.01 and (s_unique < 1.0 or v_unique < 1.0):
        minor = "V" if v_unique < s_unique else "S"
        verdict = "MARGINAL_BENEFIT"
        explanation = (
            f"Combined accuracy improves ({combined_gain:.4f}) but "
            f"{minor} features have <1% unique contribution. "
            f"The dominant set could work alone with minor loss."
        )
    elif max_cross < 0.65:
        verdict = "TRULY_INDEPENDENT"
        explanation = (
            f"Cross-prediction is near chance ({max_cross:.4f} < 0.65) "
            f"and cross-correlations are low. "
            f"The feature sets are truly independent -- combining is most valuable."
        )
    else:
        # Default / mixed signals
        if combined_gain > 0.005:
            verdict = "BOTH_RECOMMENDED"
        else:
            verdict = "EITHER_SUFFICIENT"
        explanation = (
            f"Mixed signals: combined gain={combined_gain:.4f}, "
            f"S unique={s_unique:.1f}%, V unique={v_unique:.1f}%, "
            f"max cross-pred={max_cross:.4f}."
        )

    result = {
        "verdict": verdict,
        "explanation": explanation,
        "evidence": {
            "cv_s": round(cv_s, 6),
            "cv_v": round(cv_v, 6),
            "cv_combined": round(cv_c, 6),
            "combined_gain_over_best_solo": round(combined_gain, 6),
            "s_unique_contribution_pct": round(s_unique, 2),
            "v_unique_contribution_pct": round(v_unique, 2),
            "s_to_v_cross_prediction_oob": round(s2v_oob, 6),
            "v_to_s_cross_prediction_oob": round(v2s_oob, 6),
        },
    }
    logger.info("[Verdict] %s: %s", verdict, explanation)
    return result
