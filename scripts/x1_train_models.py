"""
Step X1.2: Train and Tune Alternative Models (M2-M7)

Trains 6 model families with hyperparameter search for both VQI-S and VQI-V.
Uses 5-fold stratified CV on training data. Supports --resume to skip completed models.

Models:
    M2: XGBoost (RandomizedSearchCV, 100/384)
    M3: LightGBM (RandomizedSearchCV, 80/192)
    M4: Logistic Regression (GridSearchCV, 10)
    M5: SVM RBF (GridSearchCV, 16)
    M6: MLP PyTorch (2-stage: 30 coarse + 5 fine)
    M7: TabNet (Random 30/108)

Usage:
    python scripts/x1_train_models.py --score-type both
    python scripts/x1_train_models.py --score-type s --resume
    python scripts/x1_train_models.py --score-type s --model xgboost
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import load_training_data

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
X1_DATA_DIR = os.path.join(DATA_DIR, "x1_models")
BLUEPRINT_DIR = os.path.join(PROJECT_ROOT, "..", "blueprint")

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _save_best_params(name, score_type, params):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DATA_DIR, f"{name}{suffix}_best_params.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(dict(params), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved best params: %s", path)


def _save_cv_results(name, score_type, cv_results_df):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DATA_DIR, f"{name}{suffix}_cv_results.csv")
    cv_results_df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved CV results: %s (%d rows)", path, len(cv_results_df))


def _model_done(name, score_type):
    """Check if model outputs already exist (for --resume)."""
    suffix = "_v" if score_type == "v" else ""
    params_path = os.path.join(X1_DATA_DIR, f"{name}{suffix}_best_params.yaml")
    return os.path.exists(params_path)


def _save_checkpoint(last_model, score_type, auc, elapsed_min):
    """Save checkpoint JSON and update blueprint docs."""
    ckpt = os.path.join(X1_DATA_DIR, f"_checkpoint_{score_type}.json")
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump({
            "last_completed_model": last_model,
            "auc": auc,
            "elapsed_min": elapsed_min,
        }, f, indent=2)

    # Update progress log with checkpoint
    _update_progress_log_checkpoint(last_model, score_type, auc, elapsed_min)


def _update_progress_log_checkpoint(model_name, score_type, auc, elapsed_min):
    """Append checkpoint info to PROGRESS_LOG.md."""
    log_path = os.path.join(BLUEPRINT_DIR, "PROGRESS_LOG.md")
    if not os.path.exists(log_path):
        return

    # Read current content
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if X1.2 checkpoint section exists
    marker = "### X1.2 Training Checkpoints"
    if marker not in content:
        content += f"\n\n---\n\n{marker}\n\n"
        content += "| Model | Score | AUC | Time (min) | Status |\n"
        content += "|-------|-------|-----|------------|--------|\n"

    # Append row
    row = f"| {model_name} | VQI-{score_type.upper()} | {auc:.4f} | {elapsed_min:.1f} | DONE |\n"
    content += row

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)


def _remove_checkpoint(score_type):
    ckpt = os.path.join(X1_DATA_DIR, f"_checkpoint_{score_type}.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)


# ---------------------------------------------------------------------------
# M2: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, score_type):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    import xgboost as xgb

    logger.info("Training XGBoost (%s)...", score_type.upper())
    _flush()

    pos_weight = float(np.sum(y_train == 0)) / float(np.sum(y_train == 1))

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "scale_pos_weight": [1.0, pos_weight],
    }

    # n_jobs on estimator only, not on search — avoids Windows deadlock
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        clf, param_dist,
        n_iter=100,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,  # Sequential CV to avoid Windows multiprocessing deadlock
        verbose=2,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    logger.info("XGBoost best AUC: %.4f (%.1f min)", search.best_score_, elapsed / 60)
    logger.info("XGBoost best params: %s", search.best_params_)

    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_xgboost_model.json")
    search.best_estimator_.save_model(model_path)
    logger.info("Saved XGBoost model: %s", model_path)

    _save_best_params("xgboost", score_type, search.best_params_)
    _save_cv_results("xgboost", score_type, pd.DataFrame(search.cv_results_))
    _flush()

    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# M3: LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, score_type):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    import lightgbm as lgb

    logger.info("Training LightGBM (%s)...", score_type.upper())
    _flush()

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "is_unbalance": [True],
    }

    clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        clf, param_dist,
        n_iter=80,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    logger.info("LightGBM best AUC: %.4f (%.1f min)", search.best_score_, elapsed / 60)
    logger.info("LightGBM best params: %s", search.best_params_)

    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_lightgbm_model.txt")
    search.best_estimator_.booster_.save_model(model_path)
    logger.info("Saved LightGBM model: %s", model_path)

    joblib_path = os.path.join(MODELS_DIR, f"{prefix}_lightgbm_model.joblib")
    joblib.dump(search.best_estimator_, joblib_path)

    _save_best_params("lightgbm", score_type, search.best_params_)
    _save_cv_results("lightgbm", score_type, pd.DataFrame(search.cv_results_))
    _flush()

    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# M4: Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train, score_type, scaler):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    logger.info("Training Logistic Regression (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train)

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
    }

    clf = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        clf, param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=2,
    )

    t0 = time.time()
    search.fit(X_scaled, y_train)
    elapsed = time.time() - t0

    logger.info("LogReg best AUC: %.4f (%.1f min)", search.best_score_, elapsed / 60)
    logger.info("LogReg best params: %s", search.best_params_)

    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_logreg_model.joblib")
    joblib.dump(search.best_estimator_, model_path)
    logger.info("Saved LogReg model: %s", model_path)

    _save_best_params("logreg", score_type, search.best_params_)
    _save_cv_results("logreg", score_type, pd.DataFrame(search.cv_results_))
    _flush()

    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# M5: SVM (RBF)
# ---------------------------------------------------------------------------

def train_svm(X_train, y_train, score_type, scaler):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    logger.info("Training SVM RBF (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train)

    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }

    clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        clf, param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=2,
    )

    t0 = time.time()
    search.fit(X_scaled, y_train)
    elapsed = time.time() - t0

    logger.info("SVM best AUC: %.4f (%.1f min)", search.best_score_, elapsed / 60)
    logger.info("SVM best params: %s", search.best_params_)

    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_svm_model.joblib")
    joblib.dump(search.best_estimator_, model_path)
    logger.info("Saved SVM model: %s", model_path)

    _save_best_params("svm", score_type, search.best_params_)
    _save_cv_results("svm", score_type, pd.DataFrame(search.cv_results_))
    _flush()

    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# M6: MLP (PyTorch)
# ---------------------------------------------------------------------------

def train_mlp(X_train, y_train, score_type, scaler):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score

    logger.info("Training MLP (%s)...", score_type.upper())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("MLP device: %s", device)
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)
    input_dim = X_scaled.shape[1]

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_scaled[train_idx], y_train[train_idx]
    X_iv, y_iv = X_scaled[val_idx], y_train[val_idx]

    pos_weight = torch.tensor([float(np.sum(y_tr == 0)) / float(np.sum(y_tr == 1))]).to(device)

    class VQI_MLP(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_layers:
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(dropout),
                ])
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    def train_one_config(hidden_layers, dropout, lr, batch_size, max_epochs, patience):
        model = VQI_MLP(input_dim, hidden_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)

        X_iv_t = torch.from_numpy(X_iv).float().to(device)
        y_iv_np = y_iv

        best_auc = 0.0
        best_state = None
        wait = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb).squeeze(-1)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_iv_t).squeeze(-1).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                auc = roc_auc_score(y_iv_np, probs)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_auc, best_state, model

    # Hyperparameter space
    configs = []
    for hidden in [[128, 64], [256, 128, 64], [512, 256, 128]]:
        for dropout in [0.1, 0.3, 0.5]:
            for lr in [0.001, 0.0005, 0.0001]:
                for batch_size in [64, 128]:
                    for max_epochs in [100, 200]:
                        configs.append({
                            "hidden_layers": hidden,
                            "dropout": dropout,
                            "lr": lr,
                            "batch_size": batch_size,
                            "max_epochs": max_epochs,
                        })

    # Stage 1: Coarse search (30 random configs at 50 epochs)
    rng = np.random.RandomState(RANDOM_STATE)
    coarse_indices = rng.choice(len(configs), size=min(30, len(configs)), replace=False)
    coarse_results = []

    logger.info("MLP coarse search: %d configs at 50 epochs...", len(coarse_indices))
    _flush()
    for i, ci in enumerate(coarse_indices):
        cfg = configs[ci]
        auc, state, _ = train_one_config(
            cfg["hidden_layers"], cfg["dropout"], cfg["lr"],
            cfg["batch_size"], max_epochs=50, patience=20,
        )
        coarse_results.append({"idx": int(ci), "auc": auc, **cfg})
        logger.info("  [%d/%d] AUC=%.4f %s dr=%.1f lr=%.4f",
                     i + 1, len(coarse_indices), auc,
                     str(cfg["hidden_layers"]), cfg["dropout"], cfg["lr"])
        _flush()

    # Stage 2: Fine-tune top 5 at 200 epochs
    coarse_results.sort(key=lambda x: x["auc"], reverse=True)
    top5 = coarse_results[:5]

    logger.info("MLP fine search: top 5 configs at 200 epochs...")
    _flush()
    best_overall_auc = 0.0
    best_overall_state = None
    best_overall_config = None
    fine_results = []

    for i, cfg in enumerate(top5):
        auc, state, model = train_one_config(
            cfg["hidden_layers"], cfg["dropout"], cfg["lr"],
            cfg["batch_size"], max_epochs=200, patience=20,
        )
        fine_results.append({"auc": auc, **{k: cfg[k] for k in ["hidden_layers", "dropout", "lr", "batch_size"]}})
        logger.info("  [%d/5] AUC=%.4f %s", i + 1, auc, str(cfg["hidden_layers"]))
        _flush()

        if auc > best_overall_auc:
            best_overall_auc = auc
            best_overall_state = state
            best_overall_config = cfg

    # Save best model
    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_mlp_model.pt")
    torch.save(best_overall_state, model_path)
    logger.info("Saved MLP model: %s (AUC=%.4f)", model_path, best_overall_auc)

    config_path = os.path.join(MODELS_DIR, f"{prefix}_mlp_config.yaml")
    mlp_config = {
        "input_dim": input_dim,
        "hidden_layers": best_overall_config["hidden_layers"],
        "dropout": best_overall_config["dropout"],
        "lr": best_overall_config["lr"],
        "batch_size": best_overall_config["batch_size"],
        "best_val_auc": round(float(best_overall_auc), 6),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(mlp_config, f, default_flow_style=False, sort_keys=False)

    _save_best_params("mlp", score_type, mlp_config)

    all_results = coarse_results + fine_results
    cv_df = pd.DataFrame(all_results)
    _save_cv_results("mlp", score_type, cv_df)
    _flush()

    return best_overall_auc


# ---------------------------------------------------------------------------
# M7: TabNet
# ---------------------------------------------------------------------------

def train_tabnet(X_train, y_train, score_type, scaler):
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    from sklearn.metrics import roc_auc_score

    logger.info("Training TabNet (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_scaled[train_idx], y_train[train_idx]
    X_iv, y_iv = X_scaled[val_idx], y_train[val_idx]

    configs = []
    for n_d_a in [8, 16, 32]:
        for n_steps in [3, 5, 7]:
            for gamma in [1.0, 1.3, 1.5]:
                for lambda_sparse in [0.001, 0.01]:
                    for lr in [0.01, 0.02]:
                        configs.append({
                            "n_d": n_d_a, "n_a": n_d_a,
                            "n_steps": n_steps,
                            "gamma": gamma,
                            "lambda_sparse": lambda_sparse,
                            "lr": lr,
                        })

    rng2 = np.random.RandomState(RANDOM_STATE)
    sample_indices = rng2.choice(len(configs), size=min(30, len(configs)), replace=False)

    best_auc = 0.0
    best_model = None
    best_config = None
    results = []

    logger.info("TabNet search: %d configs...", len(sample_indices))
    _flush()
    for i, ci in enumerate(sample_indices):
        cfg = configs[ci]
        try:
            clf = TabNetClassifier(
                n_d=cfg["n_d"], n_a=cfg["n_a"],
                n_steps=cfg["n_steps"],
                gamma=cfg["gamma"],
                lambda_sparse=cfg["lambda_sparse"],
                optimizer_params=dict(lr=cfg["lr"]),
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0,
                seed=RANDOM_STATE,
            )

            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_iv, y_iv)],
                eval_metric=["auc"],
                max_epochs=200,
                patience=20,
                batch_size=256,
            )

            preds = clf.predict_proba(X_iv)[:, 1]
            auc = roc_auc_score(y_iv, preds)

            results.append({"auc": auc, **cfg})
            logger.info("  [%d/%d] AUC=%.4f n_d=%d n_steps=%d",
                         i + 1, len(sample_indices), auc, cfg["n_d"], cfg["n_steps"])
            _flush()

            if auc > best_auc:
                best_auc = auc
                best_model = clf
                best_config = cfg

        except Exception as e:
            logger.warning("  [%d/%d] TabNet config failed: %s", i + 1, len(sample_indices), e)
            results.append({"auc": 0.0, "error": str(e), **cfg})
            _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    model_path = os.path.join(MODELS_DIR, f"{prefix}_tabnet_model")
    best_model.save_model(model_path)
    logger.info("Saved TabNet model: %s (AUC=%.4f)", model_path, best_auc)

    best_config["best_val_auc"] = round(float(best_auc), 6)
    _save_best_params("tabnet", score_type, best_config)

    cv_df = pd.DataFrame(results)
    _save_cv_results("tabnet", score_type, cv_df)
    _flush()

    return best_auc


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    ("xgboost", train_xgboost, False),
    ("lightgbm", train_lightgbm, False),
    ("logreg", train_logreg, True),
    ("svm", train_svm, True),
    ("mlp", train_mlp, True),
    ("tabnet", train_tabnet, True),
]


def train_all(score_type, resume=False, only_model=None):
    """Train all M2-M7 models for a given score type."""
    logger.info("=" * 70)
    logger.info("X1.2 Training: VQI-%s", score_type.upper())
    logger.info("=" * 70)
    _flush()

    X_train, y_train = load_training_data(score_type)
    logger.info("Training data: X=%s, y=%s, class0=%d, class1=%d",
                X_train.shape, y_train.shape,
                int((y_train == 0).sum()), int((y_train == 1).sum()))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    prefix = "x1_v" if score_type == "v" else "x1"
    scaler_path = os.path.join(MODELS_DIR, f"{prefix}_feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved feature scaler: %s", scaler_path)
    _flush()

    results_summary = {}

    for name, train_fn, needs_scaler in MODEL_ORDER:
        if only_model and name != only_model:
            continue

        if resume and _model_done(name, score_type):
            logger.info("Skipping %s (already done, --resume)", name)
            _flush()
            continue

        logger.info("-" * 50)
        logger.info("Starting %s for VQI-%s...", name.upper(), score_type.upper())
        _flush()
        t0 = time.time()

        try:
            if needs_scaler:
                result = train_fn(X_train, y_train, score_type, scaler)
            else:
                result = train_fn(X_train, y_train, score_type)

            if isinstance(result, tuple):
                _, auc = result
            else:
                auc = result

            elapsed = time.time() - t0
            elapsed_min = round(elapsed / 60, 1)
            results_summary[name] = {"auc": round(float(auc), 4), "time_min": elapsed_min}
            logger.info("CHECKPOINT: %s VQI-%s complete: AUC=%.4f, time=%.1f min",
                        name.upper(), score_type.upper(), auc, elapsed_min)
            _flush()

            _save_checkpoint(name, score_type, round(float(auc), 4), elapsed_min)

        except Exception as e:
            logger.error("FAILED training %s: %s", name, e, exc_info=True)
            results_summary[name] = {"error": str(e)}
            _flush()

    # Final summary
    logger.info("=" * 70)
    logger.info("Training Summary (VQI-%s):", score_type.upper())
    for name, info in results_summary.items():
        if "error" in info:
            logger.info("  %s: FAILED - %s", name, info["error"])
        else:
            logger.info("  %s: AUC=%.4f (%.1f min)", name, info["auc"], info["time_min"])
    logger.info("=" * 70)
    _flush()

    suffix = "_v" if score_type == "v" else ""
    summary_path = os.path.join(X1_DATA_DIR, f"training_summary{suffix}.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(results_summary, f, default_flow_style=False, sort_keys=False)

    _remove_checkpoint(score_type)
    return results_summary


def verify_models(score_type):
    """Verify all trained models can predict."""
    logger.info("Verifying models for VQI-%s...", score_type.upper())
    _flush()

    X_train, _ = load_training_data(score_type)
    sample = X_train[:5]

    prefix = "x1_v" if score_type == "v" else "x1"
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_feature_scaler.joblib"))
    sample_scaled = scaler.transform(sample)

    prefix_m = "vqi_v" if score_type == "v" else "vqi"

    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, f"{prefix_m}_xgboost_model.json"))
    p = xgb_model.predict_proba(sample)[:, 1]
    logger.info("  XGBoost: predict_proba OK, scores=%s", np.round(p * 100).astype(int))

    lgb_model = joblib.load(os.path.join(MODELS_DIR, f"{prefix_m}_lightgbm_model.joblib"))
    p = lgb_model.predict_proba(sample)[:, 1]
    logger.info("  LightGBM: predict_proba OK, scores=%s", np.round(p * 100).astype(int))

    lr_model = joblib.load(os.path.join(MODELS_DIR, f"{prefix_m}_logreg_model.joblib"))
    p = lr_model.predict_proba(sample_scaled)[:, 1]
    logger.info("  LogReg: predict_proba OK, scores=%s", np.round(p * 100).astype(int))

    svm_model = joblib.load(os.path.join(MODELS_DIR, f"{prefix_m}_svm_model.joblib"))
    p = svm_model.predict_proba(sample_scaled)[:, 1]
    logger.info("  SVM: predict_proba OK, scores=%s", np.round(p * 100).astype(int))

    import torch
    import torch.nn as nn
    config_path = os.path.join(MODELS_DIR, f"{prefix_m}_mlp_config.yaml")
    with open(config_path, "r") as f:
        mlp_cfg = yaml.safe_load(f)

    class VQI_MLP(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_layers:
                layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
        def forward(self, x):
            return self.network(x)

    mlp = VQI_MLP(mlp_cfg["input_dim"], mlp_cfg["hidden_layers"], mlp_cfg["dropout"])
    mlp.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{prefix_m}_mlp_model.pt"), weights_only=True))
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.from_numpy(sample_scaled.astype(np.float32))).squeeze(-1).numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
    logger.info("  MLP: predict OK, scores=%s", np.round(p * 100).astype(int))

    from pytorch_tabnet.tab_model import TabNetClassifier
    tabnet = TabNetClassifier()
    tabnet.load_model(os.path.join(MODELS_DIR, f"{prefix_m}_tabnet_model.zip"))
    p = tabnet.predict_proba(sample_scaled)[:, 1]
    logger.info("  TabNet: predict_proba OK, scores=%s", np.round(p * 100).astype(int))

    logger.info("All model verifications PASSED for VQI-%s", score_type.upper())
    _flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X1.2: Train alternative models")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"],
                        default=None, help="Train only this model")
    args = parser.parse_args()

    os.makedirs(X1_DATA_DIR, exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]

    for st in score_types:
        train_all(st, resume=args.resume, only_model=args.model)
        if not args.model:
            verify_models(st)
