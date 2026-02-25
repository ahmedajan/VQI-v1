# Pre-trained VQI Models

This directory contains the pre-trained Random Forest models for VQI scoring.

## Full-Feature Baseline Models (v1.0)

| File | Description | Features | Trees | Size |
|------|-------------|----------|-------|------|
| `vqi_rf_model.joblib` | VQI-S (Signal Quality) | 430 | 1000 | ~192MB |
| `vqi_v_rf_model.joblib` | VQI-V (Voice Distinctiveness) | 133 | 1000 | ~204MB |

## PCA-90% Deployed Models (v2.0)

| File | Description | Components | Trees | Size |
|------|-------------|-----------|-------|------|
| `vqi_pca_scaler_s.joblib` | VQI-S StandardScaler | 430→430 | — | 12K |
| `vqi_pca_transformer_s.joblib` | VQI-S PCA (90% variance) | 430→99 | — | 172K |
| `vqi_rf_pca_model.joblib` | VQI-S PCA RF classifier | 99 | 1000 | ~212MB |
| `vqi_pca_scaler_v.joblib` | VQI-V StandardScaler | 133→133 | — | 4K |
| `vqi_pca_transformer_v.joblib` | VQI-V PCA (90% variance) | 133→47 | — | 28K |
| `vqi_v_rf_pca_model.joblib` | VQI-V PCA RF classifier | 47 | 500 | ~97MB |

### PCA-90% Scoring Pipeline

```
features → StandardScaler → PCA → RF.predict_proba → score [0-100]
```

- VQI-S: 430 selected features → 99 PCA components → RF (OOB = 0.8036)
- VQI-V: 133 selected features → 47 PCA components → RF (OOB = 0.8082)

## Download

These files exceed GitHub's 100MB limit. If they are not present (not using Git LFS), download from:

**GitHub Releases:** [https://github.com/YOUR_USERNAME/VQI/releases](https://github.com/YOUR_USERNAME/VQI/releases)

Or use the download script:
```bash
python scripts/download_models.py
```

## Training

To retrain the full-feature models from scratch:
```bash
python scripts/run_step6.py
```

To retrain the PCA-90% models:
```bash
python scripts/train_pca_models.py
```

This will generate new model files in this directory.
