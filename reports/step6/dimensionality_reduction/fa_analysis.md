# Factor Analysis Dimensionality Reduction -- Analysis

**Date:** 2026-03-05
**Method:** StandardScaler -> BIC sweep -> FactorAnalysis(BIC-optimal n) -> RF grid search (same 30-config grid)

## Component Selection (BIC Minimization)

### VQI-S

- **BIC sweep:** coarse (20-420, step 20) + fine (step 1 around minimum)
- **BIC-optimal components:** 320 (BIC = 1034631.1)
- **Previous (PCA-90% match):** 99
- **Change:** +221 components

### VQI-V

- **BIC sweep:** coarse (5-125, step 5) + fine (step 1 around minimum)
- **BIC-optimal components:** 121 (BIC = 2151541.1)
- **Previous (PCA-90% match):** 47
- **Change:** +74 components

## Results Summary

| Metric | VQI-S Full | VQI-S FA | VQI-V Full | VQI-V FA |
|--------|-----------|---------|-----------|---------|
| Features / Factors | 430 | 320 | 133 | 121 |
| Best n_estimators | 1000 | 1000 | 1000 | 1000 |
| Best max_features | 8 | sqrt | 5 | 8 |
| OOB accuracy | 0.8176 | 0.7794 | 0.8206 | 0.8183 |
| OOB diff vs full | -- | -0.0382 | -- | -0.0023 |
| Training accuracy | 0.9772 | 0.9966 | 0.9787 | 0.9974 |
| Precision (Class 0) | -- | 0.9999 | -- | 0.9987 |
| Recall (Class 0) | -- | 0.9932 | -- | 0.9961 |
| Precision (Class 1) | -- | 0.9932 | -- | 0.9961 |
| Recall (Class 1) | -- | 0.9999 | -- | 0.9987 |

## Factor Analysis Statistics

### VQI-S (320 factors, BIC-selected)

- **Convergence iterations:** 235
- **Noise variance range:** [0.0000, 0.9076]
- **Noise variance mean:** 0.0426
- **Communality range:** [0.0924, 1.0000]
- **Communality mean:** 0.9574
- **Features with communality > 0.5:** 423
- **Features with communality > 0.8:** 409

### VQI-V (121 factors, BIC-selected)

- **Convergence iterations:** 55
- **Noise variance range:** [0.0000, 0.9382]
- **Noise variance mean:** 0.1164
- **Communality range:** [0.0618, 1.0000]
- **Communality mean:** 0.8836
- **Features with communality > 0.5:** 127
- **Features with communality > 0.8:** 104

## Interpretation

Factor Analysis models observed features as linear combinations of latent factors plus 
per-feature noise. Unlike PCA, FA explicitly separates shared variance (communality) from 
unique variance (noise). Features with low communality are mostly noise and contribute 
little shared information.

BIC (Bayesian Information Criterion) penalizes model complexity, selecting the number of 
factors that best balances fit (log-likelihood) against parsimony. This replaces the previous 
approach of matching PCA-90%'s component count, allowing FA to select its own optimal 
dimensionality.

The noise variance plots show which features have the most unique/unexplained variance. 
The loadings heatmaps reveal which original features load onto which factors, showing 
the latent structure.

## Output Files

| File | Description |
|------|-------------|
| `models/vqi_rf_fa_model.joblib` | FA VQI-S RF model |
| `models/vqi_v_rf_fa_model.joblib` | FA VQI-V RF model |
| `models/vqi_fa_scaler_s.joblib` | VQI-S StandardScaler |
| `models/vqi_fa_scaler_v.joblib` | VQI-V StandardScaler |
| `models/vqi_fa_transformer_s.joblib` | VQI-S FA transformer |
| `models/vqi_fa_transformer_v.joblib` | VQI-V FA transformer |
| `data/training_fa/training_metrics.yaml` | VQI-S training metrics |
| `data/training_fa_v/training_metrics.yaml` | VQI-V training metrics |
| `reports/fa_bic_vs_components_s.png` | VQI-S BIC curve |
| `reports/fa_bic_vs_components_v.png` | VQI-V BIC curve |
| `reports/noise_variance_s.png` | VQI-S noise variance plot |
| `reports/noise_variance_v.png` | VQI-V noise variance plot |
| `reports/loadings_heatmap_s.png` | VQI-S loadings heatmap |
| `reports/loadings_heatmap_v.png` | VQI-V loadings heatmap |
| `reports/analysis.md` | This file |
