"""
Step X1.8: Train Regression Models on Continuous Mean Provider Scores

Trains 7 regression models (R1-R7) for both VQI-S and VQI-V using continuous
mean provider scores as targets instead of binary labels.

Same model families as X1.2 classification, but with regression variants:
  R1: RandomForestRegressor
  R2: XGBRegressor
  R3: LGBMRegressor
  R4: Ridge
  R5: SVR (RBF)
  R6: MLP (MSELoss)
  R7: TabNetRegressor

Usage:
    python scripts/x1_train_regression_models.py [--score-type s|v|both] [--resume] [--model <name>]
"""

import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.x1_prepare_data import load_training_data

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
REG_DIR = os.path.join(DATA_DIR, "x1_regression")

RANDOM_STATE = 42
MODEL_ORDER = ["rf", "xgboost", "lightgbm", "ridge", "svm", "mlp", "tabnet"]

logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _save_best_params(name, score_type, params):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DIR, f"reg_{name}{suffix}_best_params.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved params: %s", path)


def _save_cv_results(name, score_type, df):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DIR, f"reg_{name}{suffix}_cv_results.csv")
    df.to_csv(path, index=False)


def _save_checkpoint(completed, score_type):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DIR, f"_reg_train_checkpoint{suffix}.json")
    with open(path, "w") as f:
        json.dump({"completed": list(completed)}, f)


def _load_checkpoint(score_type):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DIR, f"_reg_train_checkpoint{suffix}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()


def _load_regression_target():
    """Load fused regression target (same for S and V — target is provider-based)."""
    return np.load(os.path.join(REG_DIR, "y_reg_train_fused.npy"))


def _load_scaler(score_type):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(MODELS_DIR, f"x1{suffix}_feature_scaler.joblib")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# R1: Random Forest Regressor
# ---------------------------------------------------------------------------
def train_rf(X_train, y_reg, score_type):
    from sklearn.ensemble import RandomForestRegressor

    logger.info("Training RF Regressor (%s)...", score_type.upper())
    _flush()

    param_grid = {
        "n_estimators": [200, 300, 400, 500, 750, 1000],
        "max_features": [5, 8, 10, 12, "sqrt"],
    }
    base = RandomForestRegressor(
        min_samples_leaf=5,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search = GridSearchCV(
        base, param_grid, scoring="neg_mean_squared_error",
        cv=5, n_jobs=1, verbose=2,
    )
    search.fit(X_train, y_reg)

    best = search.best_estimator_
    best_mse = -search.best_score_
    logger.info("  Best MSE=%.6f, params=%s", best_mse, search.best_params_)
    _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    path = os.path.join(MODELS_DIR, f"{prefix}_reg_rf_model.joblib")
    joblib.dump(best, path)
    logger.info("  Saved: %s", path)

    _save_best_params("rf", score_type, {**search.best_params_, "best_mse": round(best_mse, 6)})
    _save_cv_results("rf", score_type, pd.DataFrame(search.cv_results_))

    return best_mse


# ---------------------------------------------------------------------------
# R2: XGBoost Regressor
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_reg, score_type):
    import xgboost as xgb

    logger.info("Training XGBoost Regressor (%s)...", score_type.upper())
    _flush()

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
    }
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=100, scoring="neg_mean_squared_error",
        cv=5, random_state=RANDOM_STATE, n_jobs=1, verbose=2,
    )
    search.fit(X_train, y_reg)

    best = search.best_estimator_
    best_mse = -search.best_score_
    logger.info("  Best MSE=%.6f, params=%s", best_mse, search.best_params_)
    _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    path = os.path.join(MODELS_DIR, f"{prefix}_reg_xgboost_model.json")
    best.save_model(path)
    logger.info("  Saved: %s", path)

    _save_best_params("xgboost", score_type, {
        **{k: (int(v) if isinstance(v, (np.integer,)) else v)
           for k, v in search.best_params_.items()},
        "best_mse": round(best_mse, 6),
    })
    _save_cv_results("xgboost", score_type, pd.DataFrame(search.cv_results_))

    return best_mse


# ---------------------------------------------------------------------------
# R3: LightGBM Regressor
# ---------------------------------------------------------------------------
def train_lightgbm(X_train, y_reg, score_type):
    import lightgbm as lgb

    logger.info("Training LightGBM Regressor (%s)...", score_type.upper())
    _flush()

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
    }
    base = lgb.LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=80, scoring="neg_mean_squared_error",
        cv=5, random_state=RANDOM_STATE, n_jobs=1, verbose=2,
    )
    search.fit(X_train, y_reg)

    best = search.best_estimator_
    best_mse = -search.best_score_
    logger.info("  Best MSE=%.6f, params=%s", best_mse, search.best_params_)
    _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    path_joblib = os.path.join(MODELS_DIR, f"{prefix}_reg_lightgbm_model.joblib")
    path_txt = os.path.join(MODELS_DIR, f"{prefix}_reg_lightgbm_model.txt")
    joblib.dump(best, path_joblib)
    best.booster_.save_model(path_txt)
    logger.info("  Saved: %s", path_joblib)

    _save_best_params("lightgbm", score_type, {
        **{k: (int(v) if isinstance(v, (np.integer,)) else v)
           for k, v in search.best_params_.items()},
        "best_mse": round(best_mse, 6),
    })
    _save_cv_results("lightgbm", score_type, pd.DataFrame(search.cv_results_))

    return best_mse


# ---------------------------------------------------------------------------
# R4: Ridge Regression
# ---------------------------------------------------------------------------
def train_ridge(X_train, y_reg, score_type, scaler):
    from sklearn.linear_model import Ridge

    logger.info("Training Ridge Regressor (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    base = Ridge(random_state=RANDOM_STATE)
    search = GridSearchCV(
        base, param_grid, scoring="neg_mean_squared_error",
        cv=5, n_jobs=1, verbose=2,
    )
    search.fit(X_scaled, y_reg)

    best = search.best_estimator_
    best_mse = -search.best_score_
    logger.info("  Best MSE=%.6f, params=%s", best_mse, search.best_params_)
    _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    path = os.path.join(MODELS_DIR, f"{prefix}_reg_ridge_model.joblib")
    joblib.dump(best, path)
    logger.info("  Saved: %s", path)

    _save_best_params("ridge", score_type, {**search.best_params_, "best_mse": round(best_mse, 6)})
    _save_cv_results("ridge", score_type, pd.DataFrame(search.cv_results_))

    return best_mse


# ---------------------------------------------------------------------------
# R5: SVR (RBF)
# ---------------------------------------------------------------------------
def train_svm(X_train, y_reg, score_type, scaler):
    from sklearn.svm import SVR

    logger.info("Training SVR (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)

    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    base = SVR(kernel="rbf")
    search = GridSearchCV(
        base, param_grid, scoring="neg_mean_squared_error",
        cv=5, n_jobs=1, verbose=2,
    )
    search.fit(X_scaled, y_reg)

    best = search.best_estimator_
    best_mse = -search.best_score_
    logger.info("  Best MSE=%.6f, params=%s", best_mse, search.best_params_)
    _flush()

    prefix = "vqi_v" if score_type == "v" else "vqi"
    path = os.path.join(MODELS_DIR, f"{prefix}_reg_svm_model.joblib")
    joblib.dump(best, path)
    logger.info("  Saved: %s", path)

    _save_best_params("svm", score_type, {
        **{k: (str(v) if isinstance(v, str) else v)
           for k, v in search.best_params_.items()},
        "best_mse": round(best_mse, 6),
    })
    _save_cv_results("svm", score_type, pd.DataFrame(search.cv_results_))

    return best_mse


# ---------------------------------------------------------------------------
# R6: MLP (MSELoss)
# ---------------------------------------------------------------------------
def train_mlp(X_train, y_reg, score_type, scaler):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("Training MLP Regressor (%s)...", score_type.upper())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)
    input_dim = X_scaled.shape[1]

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_scaled[train_idx], y_reg[train_idx]
    X_iv, y_iv = X_scaled[val_idx], y_reg[val_idx]

    class VQI_MLP_Reg(nn.Module):
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
        model = VQI_MLP_Reg(input_dim, hidden_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)

        X_iv_t = torch.from_numpy(X_iv).float().to(device)
        y_iv_np = y_iv

        best_mse = float("inf")
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
                preds = model(X_iv_t).squeeze(-1).cpu().numpy()
                mse = float(np.mean((preds - y_iv_np) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_mse, best_state, model

    # Hyperparameter space
    configs = []
    for hidden in [[128, 64], [256, 128, 64], [512, 256, 128]]:
        for dropout in [0.1, 0.3, 0.5]:
            for lr in [0.001, 0.0005, 0.0001]:
                for batch_size in [64, 128]:
                    configs.append({
                        "hidden_layers": hidden,
                        "dropout": dropout,
                        "lr": lr,
                        "batch_size": batch_size,
                    })

    # Stage 1: Coarse search
    rng = np.random.RandomState(RANDOM_STATE)
    coarse_indices = rng.choice(len(configs), size=min(30, len(configs)), replace=False)
    coarse_results = []

    logger.info("  MLP coarse search: %d configs at 50 epochs...", len(coarse_indices))
    _flush()
    for i, ci in enumerate(coarse_indices):
        cfg = configs[ci]
        mse, state, _ = train_one_config(
            cfg["hidden_layers"], cfg["dropout"], cfg["lr"],
            cfg["batch_size"], max_epochs=50, patience=20,
        )
        coarse_results.append({"idx": int(ci), "mse": mse, **cfg})
        logger.info("    [%d/%d] MSE=%.6f %s dr=%.1f lr=%.4f",
                     i + 1, len(coarse_indices), mse,
                     str(cfg["hidden_layers"]), cfg["dropout"], cfg["lr"])
        _flush()

    # Stage 2: Fine-tune top 5
    coarse_results.sort(key=lambda x: x["mse"])
    top5 = coarse_results[:5]

    logger.info("  MLP fine search: top 5 at 200 epochs...")
    _flush()
    best_overall_mse = float("inf")
    best_overall_state = None
    best_overall_config = None
    fine_results = []

    for i, cfg in enumerate(top5):
        mse, state, model = train_one_config(
            cfg["hidden_layers"], cfg["dropout"], cfg["lr"],
            cfg["batch_size"], max_epochs=200, patience=20,
        )
        fine_results.append({"mse": mse, **{k: cfg[k] for k in ["hidden_layers", "dropout", "lr", "batch_size"]}})
        logger.info("    [%d/5] MSE=%.6f %s", i + 1, mse, str(cfg["hidden_layers"]))
        _flush()

        if mse < best_overall_mse:
            best_overall_mse = mse
            best_overall_state = state
            best_overall_config = cfg

    # Save
    prefix = "vqi_v" if score_type == "v" else "vqi"
    import torch
    model_path = os.path.join(MODELS_DIR, f"{prefix}_reg_mlp_model.pt")
    torch.save(best_overall_state, model_path)
    logger.info("  Saved: %s (MSE=%.6f)", model_path, best_overall_mse)

    config_path = os.path.join(MODELS_DIR, f"{prefix}_reg_mlp_config.yaml")
    mlp_config = {
        "input_dim": input_dim,
        "hidden_layers": best_overall_config["hidden_layers"],
        "dropout": best_overall_config["dropout"],
        "lr": best_overall_config["lr"],
        "batch_size": best_overall_config["batch_size"],
        "best_val_mse": round(float(best_overall_mse), 6),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(mlp_config, f, default_flow_style=False, sort_keys=False)

    _save_best_params("mlp", score_type, mlp_config)

    all_results = coarse_results + fine_results
    _save_cv_results("mlp", score_type, pd.DataFrame(all_results))
    _flush()

    return best_overall_mse


# ---------------------------------------------------------------------------
# R7: TabNet Regressor
# ---------------------------------------------------------------------------
def train_tabnet(X_train, y_reg, score_type, scaler):
    from pytorch_tabnet.tab_model import TabNetRegressor

    logger.info("Training TabNet Regressor (%s)...", score_type.upper())
    _flush()

    X_scaled = scaler.transform(X_train).astype(np.float32)

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_scaled[train_idx], y_reg[train_idx].reshape(-1, 1)
    X_iv, y_iv = X_scaled[val_idx], y_reg[val_idx].reshape(-1, 1)

    # Config space
    configs = []
    for n_d in [8, 16, 32]:
        for n_steps in [3, 5, 7]:
            for lr in [0.01, 0.02, 0.005]:
                for batch_size in [256, 512]:
                    configs.append({
                        "n_d": n_d, "n_a": n_d,
                        "n_steps": n_steps,
                        "lr": lr,
                        "batch_size": batch_size,
                    })

    # Random 30 configs
    indices = rng.choice(len(configs), size=min(30, len(configs)), replace=False)
    results = []
    best_mse = float("inf")
    best_model = None
    best_cfg = None

    for i, ci in enumerate(indices):
        cfg = configs[ci]
        try:
            model = TabNetRegressor(
                n_d=cfg["n_d"], n_a=cfg["n_a"],
                n_steps=cfg["n_steps"],
                optimizer_params={"lr": cfg["lr"]},
                verbose=0,
                seed=RANDOM_STATE,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_iv, y_iv)],
                eval_metric=["mse"],
                max_epochs=200,
                patience=20,
                batch_size=cfg["batch_size"],
            )
            preds = model.predict(X_iv)
            mse = float(np.mean((preds.flatten() - y_iv.flatten()) ** 2))
        except Exception as e:
            logger.warning("    TabNet config %d failed: %s", ci, e)
            mse = float("inf")

        results.append({"idx": int(ci), "mse": mse, **cfg})
        logger.info("    [%d/%d] MSE=%.6f n_d=%d n_steps=%d lr=%.3f",
                     i + 1, len(indices), mse, cfg["n_d"], cfg["n_steps"], cfg["lr"])
        _flush()

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_cfg = cfg

    # Save
    prefix = "vqi_v" if score_type == "v" else "vqi"
    path = os.path.join(MODELS_DIR, f"{prefix}_reg_tabnet_model")
    best_model.save_model(path)
    logger.info("  Saved: %s.zip (MSE=%.6f)", path, best_mse)

    _save_best_params("tabnet", score_type, {**best_cfg, "best_mse": round(best_mse, 6)})
    _save_cv_results("tabnet", score_type, pd.DataFrame(results))
    _flush()

    return best_mse


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_models(score_type):
    """Load all regression models and verify they produce predictions."""
    logger.info("Verifying regression models (%s)...", score_type.upper())
    prefix = "vqi_v" if score_type == "v" else "vqi"
    suffix = "_v" if score_type == "v" else ""

    X_train, _ = load_training_data(score_type)
    sample = X_train[:5].astype(np.float32)
    scaler = _load_scaler(score_type)
    sample_scaled = scaler.transform(sample).astype(np.float32)

    results = []

    # RF
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_rf_model.joblib"))
    preds = clf.predict(sample)
    logger.info("  RF: preds=%s", np.round(preds, 3))
    results.append(("rf", preds))

    # XGBoost
    import xgboost as xgb
    clf = xgb.XGBRegressor()
    clf.load_model(os.path.join(MODELS_DIR, f"{prefix}_reg_xgboost_model.json"))
    preds = clf.predict(sample)
    logger.info("  XGBoost: preds=%s", np.round(preds, 3))
    results.append(("xgboost", preds))

    # LightGBM
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_lightgbm_model.joblib"))
    preds = clf.predict(sample)
    logger.info("  LightGBM: preds=%s", np.round(preds, 3))
    results.append(("lightgbm", preds))

    # Ridge
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_ridge_model.joblib"))
    preds = clf.predict(sample_scaled)
    logger.info("  Ridge: preds=%s", np.round(preds, 3))
    results.append(("ridge", preds))

    # SVR
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_svm_model.joblib"))
    preds = clf.predict(sample_scaled)
    logger.info("  SVR: preds=%s", np.round(preds, 3))
    results.append(("svm", preds))

    # MLP
    import torch
    import torch.nn as nn

    config_path = os.path.join(MODELS_DIR, f"{prefix}_reg_mlp_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    class VQI_MLP_Reg(nn.Module):
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

    mlp = VQI_MLP_Reg(cfg["input_dim"], cfg["hidden_layers"], cfg["dropout"])
    mlp.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{prefix}_reg_mlp_model.pt"),
                                    weights_only=True, map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        preds = mlp(torch.from_numpy(sample_scaled)).squeeze(-1).numpy()
    logger.info("  MLP: preds=%s", np.round(preds, 3))
    results.append(("mlp", preds))

    # TabNet
    from pytorch_tabnet.tab_model import TabNetRegressor
    tab = TabNetRegressor()
    tab.load_model(os.path.join(MODELS_DIR, f"{prefix}_reg_tabnet_model.zip"))
    preds = tab.predict(sample_scaled).flatten()
    logger.info("  TabNet: preds=%s", np.round(preds, 3))
    results.append(("tabnet", preds))

    # Check all predictions are reasonable
    for name, preds in results:
        scores = np.clip(np.round(preds * 100), 0, 100).astype(int)
        logger.info("  %s VQI scores: %s", name, scores)

    logger.info("All regression models verified.")
    _flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="X1.8: Train regression models")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Train only this model (rf, xgboost, lightgbm, ridge, svm, mlp, tabnet)")
    args = parser.parse_args()

    os.makedirs(X1_DIR, exist_ok=True)
    os.makedirs(REG_DIR, exist_ok=True)

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]
    models_to_train = [args.model] if args.model else MODEL_ORDER

    y_reg = _load_regression_target()
    logger.info("Regression target: shape=%s, range=[%.4f, %.4f], mean=%.4f",
                y_reg.shape, y_reg.min(), y_reg.max(), y_reg.mean())

    for st in score_types:
        label = "VQI-S" if st == "s" else "VQI-V"
        logger.info("=" * 70)
        logger.info("Training regression models for %s", label)
        logger.info("=" * 70)
        _flush()

        X_train, _ = load_training_data(st)
        scaler = _load_scaler(st)

        completed = _load_checkpoint(st) if args.resume else set()

        summary = []

        for name in models_to_train:
            if name in completed:
                logger.info("Skipping %s (already done)", name)
                continue

            t0 = time.time()

            if name == "rf":
                mse = train_rf(X_train, y_reg, st)
            elif name == "xgboost":
                mse = train_xgboost(X_train, y_reg, st)
            elif name == "lightgbm":
                mse = train_lightgbm(X_train, y_reg, st)
            elif name == "ridge":
                mse = train_ridge(X_train, y_reg, st, scaler)
            elif name == "svm":
                mse = train_svm(X_train, y_reg, st, scaler)
            elif name == "mlp":
                mse = train_mlp(X_train, y_reg, st, scaler)
            elif name == "tabnet":
                mse = train_tabnet(X_train, y_reg, st, scaler)
            else:
                logger.warning("Unknown model: %s", name)
                continue

            elapsed = (time.time() - t0) / 60.0
            summary.append({"model": name, "score_type": st, "mse": mse, "time_min": round(elapsed, 1)})
            logger.info("DONE: %s MSE=%.6f (%.1f min)", name, mse, elapsed)
            _flush()

            completed.add(name)
            _save_checkpoint(completed, st)

        # Save summary
        if summary:
            suffix = "_v" if st == "v" else ""
            df = pd.DataFrame(summary)
            df.to_csv(os.path.join(X1_DIR, f"reg_training_summary{suffix}.csv"), index=False)

        # Verify
        if not args.model:
            verify_models(st)

        # Remove checkpoint
        suffix = "_v" if st == "v" else ""
        cp = os.path.join(X1_DIR, f"_reg_train_checkpoint{suffix}.json")
        if os.path.exists(cp):
            os.remove(cp)

    print("\n" + "=" * 70)
    print("X1.8 Regression Training COMPLETE")
    print("=" * 70)
