# PCA-Reduced RF Model Comparison (3-Way)

**Date:** 2026-02-21
**Method:** StandardScaler -> PCA -> RF grid search (same 30-config grid)

## 3-Way Comparison: Full vs 95% PCA vs 90% PCA

| Metric | VQI-S Full | VQI-S 95% | VQI-S 90% | VQI-V Full | VQI-V 95% | VQI-V 90% |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| Features / PCs | 430 | 156 | 99 | 133 | 60 | 47 |
| Best n_estimators | 1000 | 1000 | 1000 | 1000 | 1000 | 500 |
| Best max_features | 8 | 12 | 8 | 5 | 10 | 10 |
| OOB accuracy | 0.8176 | 0.8016 | 0.8036 | 0.8206 | 0.8086 | 0.8082 |
| OOB accuracy diff | — | -0.0160 | -0.0139 | — | -0.0120 | -0.0124 |
| Training accuracy | 0.9772 | 0.9972 | 0.9955 | 0.9787 | 0.9857 | 0.9811 |
| Model size (MB) | 191.93 | 203.68 | 211.27 | 203.98 | 194.58 | 96.23 |
| Precision (Class 0) | — | 0.9993 | 0.9976 | — | 0.9879 | 0.9823 |
| Recall (Class 0) | — | 0.9952 | 0.9933 | — | 0.9833 | 0.9799 |
| Precision (Class 1) | — | 0.9952 | 0.9933 | — | 0.9834 | 0.9799 |
| Recall (Class 1) | — | 0.9993 | 0.9976 | — | 0.9880 | 0.9824 |

## Interpretation

The 95% variance threshold retains more discriminative information than the 90% threshold. VQI-S goes from 430 features to 156 PCs (95%) or 99 PCs (90%), with OOB accuracy drops of -0.0160 and -0.0139 respectively. VQI-V goes from 133 features to 60 PCs (95%) or 47 PCs (90%), with OOB accuracy drops of -0.0120 and -0.0124 respectively.

The 95% threshold recovers approximately -0.0020 pp for VQI-S and 0.0004 pp for VQI-V compared to 90%, at the cost of 57 additional PCs for VQI-S and 13 for VQI-V. The full-feature models remain the production default, as PCA reduction consistently underperforms the original feature set.

## Output Files

### 90% variance threshold

| File | Description |
|------|-------------|
| `models/vqi_rf_pca_model.joblib` | PCA 90% VQI-S RF model |
| `models/vqi_v_rf_pca_model.joblib` | PCA 90% VQI-V RF model |
| `models/vqi_pca_scaler_s.joblib` | VQI-S StandardScaler (90%) |
| `models/vqi_pca_scaler_v.joblib` | VQI-V StandardScaler (90%) |
| `models/vqi_pca_transformer_s.joblib` | VQI-S PCA transformer (90%) |
| `models/vqi_pca_transformer_v.joblib` | VQI-V PCA transformer (90%) |
| `data/training_pca/training_metrics.yaml` | VQI-S 90% training metrics |
| `data/training_pca_v/training_metrics.yaml` | VQI-V 90% training metrics |

### 95% variance threshold

| File | Description |
|------|-------------|
| `models/vqi_rf_pca95_model.joblib` | PCA 95% VQI-S RF model |
| `models/vqi_v_rf_pca95_model.joblib` | PCA 95% VQI-V RF model |
| `models/vqi_pca95_scaler_s.joblib` | VQI-S StandardScaler (95%) |
| `models/vqi_pca95_scaler_v.joblib` | VQI-V StandardScaler (95%) |
| `models/vqi_pca95_transformer_s.joblib` | VQI-S PCA transformer (95%) |
| `models/vqi_pca95_transformer_v.joblib` | VQI-V PCA transformer (95%) |
| `data/training_pca95/training_metrics.yaml` | VQI-S 95% training metrics |
| `data/training_pca95_v/training_metrics.yaml` | VQI-V 95% training metrics |

| `reports/pca/comparison.md` | This file |
