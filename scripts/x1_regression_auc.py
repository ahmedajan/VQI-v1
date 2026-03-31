"""
Quick evaluation: Compute AUC-ROC for all 14 regression models on validation set.

Regression models predict continuous [0,1] values. These can be used directly as
"probability of being high-quality" for AUC computation against binary labels.

Usage:
    python scripts/x1_regression_auc.py
"""

import json
import os
import sys

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "x1")

from scripts.x1_prepare_data import load_validation_data
import joblib


def load_scaler(score_type):
    suffix = "_v" if score_type == "v" else ""
    path = os.path.join(MODELS_DIR, f"x1{suffix}_feature_scaler.joblib")
    return joblib.load(path)


def predict_all_regression(X_val, score_type):
    """Load all 7 regression models and predict on X_val."""
    prefix = "vqi_v" if score_type == "v" else "vqi"
    scaler = load_scaler(score_type)
    X_scaled = scaler.transform(X_val).astype(np.float32)

    predictions = {}

    # RF
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_rf_model.joblib"))
    predictions["RF (Reg)"] = clf.predict(X_val)

    # XGBoost
    import xgboost as xgb
    clf = xgb.XGBRegressor()
    clf.load_model(os.path.join(MODELS_DIR, f"{prefix}_reg_xgboost_model.json"))
    predictions["XGBoost (Reg)"] = clf.predict(X_val)

    # LightGBM
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_lightgbm_model.joblib"))
    predictions["LightGBM (Reg)"] = clf.predict(X_val)

    # Ridge
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_ridge_model.joblib"))
    predictions["Ridge (Reg)"] = clf.predict(X_scaled)

    # SVR
    clf = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_reg_svm_model.joblib"))
    predictions["SVR (Reg)"] = clf.predict(X_scaled)

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
        preds = mlp(torch.from_numpy(X_scaled)).squeeze(-1).numpy()
    predictions["MLP (Reg)"] = preds

    # TabNet
    from pytorch_tabnet.tab_model import TabNetRegressor
    tab = TabNetRegressor()
    tab.load_model(os.path.join(MODELS_DIR, f"{prefix}_reg_tabnet_model.zip"))
    predictions["TabNet (Reg)"] = tab.predict(X_scaled).flatten()

    return predictions


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = []

    for score_type in ["s", "v"]:
        label = "VQI-S" if score_type == "s" else "VQI-V"
        print(f"\n{'='*60}")
        print(f"  {label} — Regression Models on Validation Set")
        print(f"{'='*60}")

        X_val, y_val = load_validation_data(score_type)
        print(f"  Validation: {X_val.shape[0]} samples, {y_val.sum()} positive")

        predictions = predict_all_regression(X_val, score_type)

        print(f"\n  {'Model':<20} {'AUC-ROC':>8} {'Brier':>8} {'F1@0.5':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

        for name, preds in predictions.items():
            # Clip to [0,1]
            preds_clipped = np.clip(preds, 0, 1)

            auc = roc_auc_score(y_val, preds_clipped)
            brier = brier_score_loss(y_val, preds_clipped)
            y_pred_bin = (preds_clipped >= 0.5).astype(int)
            f1 = f1_score(y_val, y_pred_bin)

            print(f"  {name:<20} {auc:>8.4f} {brier:>8.4f} {f1:>8.4f}")
            results.append({
                "score_type": score_type,
                "model": name,
                "auc_roc": round(auc, 4),
                "brier": round(brier, 4),
                "f1": round(f1, 4),
            })

    # Save results
    import csv
    out_path = os.path.join(DATA_DIR, "x1_models", "reg_validation_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["score_type", "model", "auc_roc", "brier", "f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
