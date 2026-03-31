"""X1.9g: Retrain all 28 models on expanded training data.

Uses best hyperparameters from X1.2 (classifiers) and X1.8 (regressors)
and refits on the merged 58,108-sample training set.

Models: 7 classifiers + 7 regressors × 2 score types = 28 total.
  Classifiers: RF, XGBoost, LightGBM, LogReg, SVM, MLP, TabNet
  Regressors:  RF, XGBoost, LightGBM, Ridge, SVR, MLP, TabNet

Usage:
    python scripts/x1_retrain_expanded.py [--score-type s|v|both] [--resume]
"""

import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
X1_DIR = os.path.join(DATA_DIR, "x1_models")
EXPANDED_S = os.path.join(DATA_DIR, "step6", "full_feature", "training_expanded")
EXPANDED_V = os.path.join(DATA_DIR, "step6", "full_feature", "training_expanded_v")
REG_EXPANDED = os.path.join(DATA_DIR, "x1_regression", "expanded")
RESULTS_DIR = os.path.join(DATA_DIR, "x1_expanded_models")

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _load_params(name, score_type):
    """Load best params YAML for a model."""
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(X1_DIR, f"{name}{suffix}_best_params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.unsafe_load(f)
    # Filter out metric keys (best_mse, best_val_mse, etc.) — keep only model params
    metric_keys = {"best_mse", "best_val_mse", "best_score", "best_auc"}
    return {k: v for k, v in params.items() if k not in metric_keys}


def _model_output_path(name, score_type, ext):
    prefix = "vqi_v" if score_type == "v" else "vqi"
    return os.path.join(MODELS_DIR, f"{prefix}_exp_{name}_model.{ext}")


def _model_done(name, score_type):
    """Check if expanded model already exists."""
    for ext in ["joblib", "json", "pt", "zip", "txt"]:
        if os.path.exists(_model_output_path(name, score_type, ext)):
            return True
    return False


def load_expanded_data(score_type):
    base = EXPANDED_S if score_type == "s" else EXPANDED_V
    X = np.load(os.path.join(base, "X_train.npy"))
    y = np.load(os.path.join(base, "y_train.npy"))
    return X, y


def load_expanded_reg_target():
    return np.load(os.path.join(REG_EXPANDED, "y_reg_train_fused.npy"))


def get_scaler(X_train, score_type):
    """Fit StandardScaler on expanded data (for SVM, LogReg, MLP, TabNet)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(MODELS_DIR, f"x1_exp_feature_scaler{suffix}.joblib")
    joblib.dump(scaler, path)
    return scaler


# ===== CLASSIFIERS =====

def train_rf_clf(X, y, score_type):
    from sklearn.ensemble import RandomForestClassifier
    logger.info("Training RF Classifier (expanded, %s)...", score_type.upper())

    clf = RandomForestClassifier(
        n_estimators=500, criterion="gini", max_features="sqrt",
        max_depth=None, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X, y)
    oob_path = _model_output_path("rf", score_type, "joblib")
    joblib.dump(clf, oob_path)
    logger.info("  RF saved: %s", oob_path)
    return clf


def train_xgboost_clf(X, y, score_type):
    import xgboost as xgb
    logger.info("Training XGBoost Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("xgboost", score_type)
    clf = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 7),
        learning_rate=params.get("learning_rate", 0.2),
        subsample=params.get("subsample", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X, y)
    path = _model_output_path("xgboost", score_type, "json")
    clf.save_model(path)
    logger.info("  XGBoost saved: %s", path)
    return clf


def train_lightgbm_clf(X, y, score_type):
    import lightgbm as lgb
    logger.info("Training LightGBM Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("lightgbm", score_type)
    clf = lgb.LGBMClassifier(
        num_leaves=params.get("num_leaves", 63),
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", -1),
        learning_rate=params.get("learning_rate", 0.1),
        is_unbalance=params.get("is_unbalance", True),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    clf.fit(X, y)
    path = _model_output_path("lightgbm", score_type, "joblib")
    joblib.dump(clf, path)
    # Also save text format
    txt_path = _model_output_path("lightgbm", score_type, "txt")
    clf.booster_.save_model(txt_path)
    logger.info("  LightGBM saved: %s", path)
    return clf


def train_logreg_clf(X, y, score_type, scaler):
    from sklearn.linear_model import LogisticRegression
    logger.info("Training LogReg Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("logreg", score_type)
    X_scaled = scaler.transform(X)
    penalty = params.get("penalty", "l2")
    solver = "saga" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        C=params.get("C", 0.1),
        penalty=penalty,
        solver=solver,
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_scaled, y)
    path = _model_output_path("logreg", score_type, "joblib")
    joblib.dump(clf, path)
    logger.info("  LogReg saved: %s", path)
    return clf


def train_svm_clf(X, y, score_type, scaler):
    from sklearn.svm import SVC
    logger.info("Training SVM Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("svm", score_type)
    X_scaled = scaler.transform(X)
    clf = SVC(
        C=params.get("C", 10.0),
        gamma=params.get("gamma", "auto"),
        kernel="rbf", probability=True, random_state=RANDOM_STATE,
    )
    clf.fit(X_scaled, y)
    path = _model_output_path("svm", score_type, "joblib")
    joblib.dump(clf, path)
    logger.info("  SVM saved: %s", path)
    return clf


def train_mlp_clf(X, y, score_type, scaler):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score

    logger.info("Training MLP Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("mlp", score_type)
    X_scaled = scaler.transform(X).astype(np.float32)
    input_dim = X_scaled.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_scaled[train_idx], y[train_idx]
    X_iv, y_iv = X_scaled[val_idx], y[val_idx]

    pos_weight = torch.tensor(
        [float(np.sum(y_tr == 0)) / float(np.sum(y_tr == 1))]
    ).to(device)

    class VQI_MLP(nn.Module):
        def __init__(self, in_dim, hidden_layers, dropout):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden_layers:
                layers.extend([
                    nn.Linear(prev, h), nn.ReLU(),
                    nn.BatchNorm1d(h), nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    hidden = params.get("hidden_layers", [512, 256, 128])
    dropout = params.get("dropout", 0.5)
    lr = params.get("lr", 0.0005)
    batch_size = params.get("batch_size", 64)

    model = VQI_MLP(input_dim, hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    X_iv_t = torch.from_numpy(X_iv).float().to(device)

    best_auc = 0.0
    best_state = None
    patience = 20
    wait = 0

    for epoch in range(200):
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
            auc = roc_auc_score(y_iv, probs)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    path = _model_output_path("mlp", score_type, "pt")
    torch.save(best_state, path)

    config_path = path.replace(".pt", "_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "input_dim": input_dim, "hidden_layers": hidden,
            "dropout": dropout, "lr": lr, "batch_size": batch_size,
            "best_val_auc": float(best_auc),
        }, f)
    logger.info("  MLP saved: %s (AUC=%.4f)", path, best_auc)
    return model


def train_tabnet_clf(X, y, score_type, scaler):
    from pytorch_tabnet.tab_model import TabNetClassifier
    logger.info("Training TabNet Classifier (expanded, %s)...", score_type.upper())

    params = _load_params("tabnet", score_type)
    X_scaled = scaler.transform(X).astype(np.float64)

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    X_tr, y_tr = X_scaled[idx[:split]], y[idx[:split]]
    X_iv, y_iv = X_scaled[idx[split:]], y[idx[split:]]

    clf = TabNetClassifier(
        n_d=params.get("n_d", 16), n_a=params.get("n_a", 16),
        n_steps=params.get("n_steps", 3),
        gamma=params.get("gamma", 1.5),
        lambda_sparse=params.get("lambda_sparse", 0.001),
        optimizer_params={"lr": params.get("lr", 0.02)},
        seed=RANDOM_STATE, verbose=0,
    )
    clf.fit(
        X_tr, y_tr, eval_set=[(X_iv, y_iv)],
        eval_metric=["auc"], max_epochs=200,
        patience=20, batch_size=256,
    )
    path = _model_output_path("tabnet", score_type, "zip")
    clf.save_model(path)
    logger.info("  TabNet saved: %s", path)
    return clf


# ===== REGRESSORS =====

def train_rf_reg(X, y_reg, score_type):
    from sklearn.ensemble import RandomForestRegressor
    logger.info("Training RF Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_rf", score_type)
    reg = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 750),
        max_features=params.get("max_features", "sqrt"),
        max_depth=None, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg.fit(X, y_reg)
    path = _model_output_path("reg_rf", score_type, "joblib")
    joblib.dump(reg, path)
    logger.info("  RF Reg saved: %s", path)
    return reg


def train_xgboost_reg(X, y_reg, score_type):
    import xgboost as xgb
    logger.info("Training XGBoost Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_xgboost", score_type)
    reg = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 7),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg.fit(X, y_reg)
    path = _model_output_path("reg_xgboost", score_type, "json")
    reg.save_model(path)
    logger.info("  XGBoost Reg saved: %s", path)
    return reg


def train_lightgbm_reg(X, y_reg, score_type):
    import lightgbm as lgb
    logger.info("Training LightGBM Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_lightgbm", score_type)
    reg = lgb.LGBMRegressor(
        num_leaves=params.get("num_leaves", 63),
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", -1),
        learning_rate=params.get("learning_rate", 0.05),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    reg.fit(X, y_reg)
    path = _model_output_path("reg_lightgbm", score_type, "joblib")
    joblib.dump(reg, path)
    txt_path = _model_output_path("reg_lightgbm", score_type, "txt")
    reg.booster_.save_model(txt_path)
    logger.info("  LightGBM Reg saved: %s", path)
    return reg


def train_ridge_reg(X, y_reg, score_type, scaler):
    from sklearn.linear_model import Ridge
    logger.info("Training Ridge Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_ridge", score_type)
    X_scaled = scaler.transform(X)
    reg = Ridge(alpha=params.get("alpha", 100.0), random_state=RANDOM_STATE)
    reg.fit(X_scaled, y_reg)
    path = _model_output_path("reg_ridge", score_type, "joblib")
    joblib.dump(reg, path)
    logger.info("  Ridge saved: %s", path)
    return reg


def train_svm_reg(X, y_reg, score_type, scaler):
    from sklearn.svm import SVR
    logger.info("Training SVR (expanded, %s)...", score_type.upper())

    params = _load_params("reg_svm", score_type)
    X_scaled = scaler.transform(X)
    reg = SVR(
        C=params.get("C", 1.0),
        gamma=params.get("gamma", "auto"),
        kernel="rbf",
    )
    reg.fit(X_scaled, y_reg)
    path = _model_output_path("reg_svm", score_type, "joblib")
    joblib.dump(reg, path)
    logger.info("  SVR saved: %s", path)
    return reg


def train_mlp_reg(X, y_reg, score_type, scaler):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("Training MLP Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_mlp", score_type)
    X_scaled = scaler.transform(X).astype(np.float32)
    input_dim = X_scaled.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    X_tr, y_tr = X_scaled[idx[:split]], y_reg[idx[:split]]
    X_iv, y_iv = X_scaled[idx[split:]], y_reg[idx[split:]]

    class VQI_MLP_Reg(nn.Module):
        def __init__(self, in_dim, hidden_layers, dropout):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden_layers:
                layers.extend([
                    nn.Linear(prev, h), nn.ReLU(),
                    nn.BatchNorm1d(h), nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    hidden = params.get("hidden_layers", [512, 256, 128])
    dropout = params.get("dropout", 0.1)
    lr = params.get("lr", 0.001)
    batch_size = params.get("batch_size", 128)

    model = VQI_MLP_Reg(input_dim, hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    X_iv_t = torch.from_numpy(X_iv).float().to(device)
    y_iv_t = torch.from_numpy(y_iv).float().to(device)

    best_mse = float("inf")
    best_state = None
    patience = 20
    wait = 0

    for epoch in range(200):
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
            preds = model(X_iv_t).squeeze(-1)
            mse = criterion(preds, y_iv_t).item()

        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    path = _model_output_path("reg_mlp", score_type, "pt")
    torch.save(best_state, path)

    config_path = path.replace(".pt", "_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "input_dim": input_dim, "hidden_layers": hidden,
            "dropout": dropout, "lr": lr, "batch_size": batch_size,
            "best_val_mse": float(best_mse),
        }, f)
    logger.info("  MLP Reg saved: %s (MSE=%.6f)", path, best_mse)
    return model


def train_tabnet_reg(X, y_reg, score_type, scaler):
    from pytorch_tabnet.tab_model import TabNetRegressor
    logger.info("Training TabNet Regressor (expanded, %s)...", score_type.upper())

    params = _load_params("reg_tabnet", score_type)
    X_scaled = scaler.transform(X).astype(np.float64)

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_scaled)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    X_tr, y_tr = X_scaled[idx[:split]], y_reg[idx[:split]].reshape(-1, 1)
    X_iv, y_iv = X_scaled[idx[split:]], y_reg[idx[split:]].reshape(-1, 1)

    reg = TabNetRegressor(
        n_d=params.get("n_d", 16), n_a=params.get("n_a", 16),
        n_steps=params.get("n_steps", 5),
        optimizer_params={"lr": params.get("lr", 0.02)},
        seed=RANDOM_STATE, verbose=0,
    )
    reg.fit(
        X_tr, y_tr, eval_set=[(X_iv, y_iv)],
        eval_metric=["mse"], max_epochs=200,
        patience=20, batch_size=params.get("batch_size", 256),
    )
    path = _model_output_path("reg_tabnet", score_type, "zip")
    reg.save_model(path)
    logger.info("  TabNet Reg saved: %s", path)
    return reg


# ===== MAIN =====

CLF_ORDER = ["rf", "xgboost", "lightgbm", "logreg", "svm", "mlp", "tabnet"]
REG_ORDER = ["rf", "xgboost", "lightgbm", "ridge", "svm", "mlp", "tabnet"]


def main():
    parser = argparse.ArgumentParser(
        description="X1.9g: Retrain all models on expanded data")
    parser.add_argument("--score-type", choices=["s", "v", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("X1.9g: Retrain All Models on Expanded Data")
    logger.info("=" * 70)

    score_types = ["s", "v"] if args.score_type == "both" else [args.score_type]
    y_reg = load_expanded_reg_target()
    logger.info("Regression target: shape=%s, range=[%.4f, %.4f]",
                y_reg.shape, y_reg.min(), y_reg.max())

    results = []

    for st in score_types:
        label = "VQI-S" if st == "s" else "VQI-V"
        logger.info("\n" + "=" * 70)
        logger.info("Score type: %s", label)
        logger.info("=" * 70)

        X_train, y_train = load_expanded_data(st)
        logger.info("Training data: X=%s, y=%s (C0=%d, C1=%d)",
                    X_train.shape, y_train.shape,
                    (y_train == 0).sum(), (y_train == 1).sum())

        scaler = get_scaler(X_train, st)

        # ===== Classifiers =====
        logger.info("\n--- Classifiers ---")

        clf_funcs = {
            "rf": lambda: train_rf_clf(X_train, y_train, st),
            "xgboost": lambda: train_xgboost_clf(X_train, y_train, st),
            "lightgbm": lambda: train_lightgbm_clf(X_train, y_train, st),
            "logreg": lambda: train_logreg_clf(X_train, y_train, st, scaler),
            "svm": lambda: train_svm_clf(X_train, y_train, st, scaler),
            "mlp": lambda: train_mlp_clf(X_train, y_train, st, scaler),
            "tabnet": lambda: train_tabnet_clf(X_train, y_train, st, scaler),
        }

        for name in CLF_ORDER:
            if args.resume and _model_done(name, st):
                logger.info("  SKIP %s (already exists)", name)
                continue
            t0 = time.time()
            clf_funcs[name]()
            elapsed = time.time() - t0
            results.append({
                "type": "classifier", "model": name,
                "score_type": st, "time_min": elapsed / 60,
            })
            logger.info("  %s done in %.1f min", name, elapsed / 60)
            _flush()

        # ===== Regressors =====
        logger.info("\n--- Regressors ---")

        reg_funcs = {
            "rf": lambda: train_rf_reg(X_train, y_reg, st),
            "xgboost": lambda: train_xgboost_reg(X_train, y_reg, st),
            "lightgbm": lambda: train_lightgbm_reg(X_train, y_reg, st),
            "ridge": lambda: train_ridge_reg(X_train, y_reg, st, scaler),
            "svm": lambda: train_svm_reg(X_train, y_reg, st, scaler),
            "mlp": lambda: train_mlp_reg(X_train, y_reg, st, scaler),
            "tabnet": lambda: train_tabnet_reg(X_train, y_reg, st, scaler),
        }

        for name in REG_ORDER:
            reg_name = f"reg_{name}"
            if args.resume and _model_done(reg_name, st):
                logger.info("  SKIP %s (already exists)", reg_name)
                continue
            t0 = time.time()
            reg_funcs[name]()
            elapsed = time.time() - t0
            results.append({
                "type": "regressor", "model": name,
                "score_type": st, "time_min": elapsed / 60,
            })
            logger.info("  %s done in %.1f min", reg_name, elapsed / 60)
            _flush()

    # Save results summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "training_summary.csv"), index=False)

    logger.info("\n" + "=" * 70)
    logger.info("X1.9g Retrain COMPLETE")
    logger.info("=" * 70)
    logger.info("Total models: %d", len(results))
    logger.info("Total time: %.1f min",
                sum(r["time_min"] for r in results))


if __name__ == "__main__":
    main()
