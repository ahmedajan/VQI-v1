# VQI v4.0 Models

## Model Files

| File | Description | Size |
|------|-------------|------|
| `vqi_v4_meta.json` | Metadata: DR config, feature counts, AUC, ERC metrics | <1KB |
| `vqi_v4_scaler_s.joblib` | VQI-S StandardScaler (430 features) | ~11KB |
| `vqi_v4_model_s.joblib` | VQI-S Ridge Regressor (20K training samples) | ~2.7KB |
| `vqi_v4_scaler_v.joblib` | VQI-V StandardScaler (133 features) | ~3.8KB |
| `vqi_v4_model_v.json` | VQI-V XGBoost Regressor (58K expanded training samples) | ~3.3MB |

## Scoring Pipeline

```
raw_features -> scaler.transform(features) -> model.predict() * 100 -> VQI score [0-100]
```

- **VQI-S (Signal Quality):** 430 full features, Ridge Regressor, AUC=0.8803, ERC@20%=20.2%
- **VQI-V (Voice Distinctiveness):** 133 full features, XGBoost Regressor, AUC=0.9130, ERC@20%=14.3%

## Changes from v3.0

- Added per-score-type scalers (`vqi_v4_scaler_s.joblib`, `vqi_v4_scaler_v.joblib`)
- Added metadata file (`vqi_v4_meta.json`) with DR config and performance metrics
- Model files renamed with `v4_` prefix for clarity
- No PCA or other dimensionality reduction -- operates on full features
- Total model size: ~3.4MB (down 48x from v2.0's 321MB)
