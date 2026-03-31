# ICA Dimensionality Reduction -- Analysis

**Date:** 2026-03-05
**Method:** StandardScaler -> Parallel Analysis -> FastICA(PA-optimal n) -> RF grid search (same 30-config grid)

## Component Selection (Parallel Analysis -- Horn's Method)

- **Permutations:** 100
- **Threshold:** 95th percentile of random eigenvalues
- **Random state:** 42

### VQI-S

- **PA-optimal components:** 50
- **Previous (PCA-90% match):** 99
- **Change:** -49 components
- **Last retained eigenvalue:** 1.2004 (threshold: 1.1969)

### VQI-V

- **PA-optimal components:** 27
- **Previous (PCA-90% match):** 47
- **Change:** -20 components
- **Last retained eigenvalue:** 1.1030 (threshold: 1.0821)

## Results Summary

| Metric | VQI-S Full | VQI-S ICA | VQI-V Full | VQI-V ICA |
|--------|-----------|----------|-----------|----------|
| Features / ICs | 430 | 50 | 133 | 27 |
| Best n_estimators | 1000 | 500 | 1000 | 1000 |
| Best max_features | 8 | 5 | 5 | 12 |
| OOB accuracy | 0.8176 | 0.7952 | 0.8206 | 0.7969 |
| OOB diff vs full | -- | -0.0223 | -- | -0.0237 |
| Training accuracy | 0.9772 | 0.9791 | 0.9787 | 0.9723 |
| Precision (Class 0) | -- | 0.9776 | -- | 0.9681 |
| Recall (Class 0) | -- | 0.9806 | -- | 0.9767 |
| Precision (Class 1) | -- | 0.9805 | -- | 0.9765 |
| Recall (Class 1) | -- | 0.9775 | -- | 0.9679 |

## ICA Component Statistics

### VQI-S (50 ICs, PA-selected)

- **Convergence iterations:** 67
- **Kurtosis range:** [-0.92, 2030.42]
- **Kurtosis mean:** 105.51
- **Kurtosis median:** 6.15
- **ICs with |kurtosis| > 1 (super-Gaussian):** 45
- **ICs with |kurtosis| > 3 (highly non-Gaussian):** 38

### VQI-V (27 ICs, PA-selected)

- **Convergence iterations:** 60
- **Kurtosis range:** [-0.98, 96.12]
- **Kurtosis mean:** 11.05
- **Kurtosis median:** 2.11
- **ICs with |kurtosis| > 1 (super-Gaussian):** 22
- **ICs with |kurtosis| > 3 (highly non-Gaussian):** 12

## Interpretation

ICA (Independent Component Analysis) seeks statistically independent components by 
maximizing non-Gaussianity, unlike PCA which maximizes variance. The kurtosis values 
above indicate how non-Gaussian each component is -- higher absolute kurtosis means 
more non-Gaussian and potentially more informative for separating classes.

Parallel Analysis (Horn's method) determines the number of components by comparing 
real eigenvalues against those from column-permuted random data. Components are retained 
only if their eigenvalue exceeds the 95th percentile of the random distribution, ensuring 
they capture more structure than noise. This replaces the previous approach of matching 
PCA-90%'s component count.

The mixing matrix heatmaps show how original features contribute to the independent 
components, revealing which features form natural independent groupings.

## Output Files

| File | Description |
|------|-------------|
| `models/vqi_rf_ica_model.joblib` | ICA VQI-S RF model |
| `models/vqi_v_rf_ica_model.joblib` | ICA VQI-V RF model |
| `models/vqi_ica_scaler_s.joblib` | VQI-S StandardScaler |
| `models/vqi_ica_scaler_v.joblib` | VQI-V StandardScaler |
| `models/vqi_ica_transformer_s.joblib` | VQI-S ICA transformer |
| `models/vqi_ica_transformer_v.joblib` | VQI-V ICA transformer |
| `data/training_ica/training_metrics.yaml` | VQI-S training metrics |
| `data/training_ica_v/training_metrics.yaml` | VQI-V training metrics |
| `reports/ica_parallel_analysis_s.png` | VQI-S parallel analysis plot |
| `reports/ica_parallel_analysis_v.png` | VQI-V parallel analysis plot |
| `reports/kurtosis_distribution_s.png` | VQI-S kurtosis plot |
| `reports/kurtosis_distribution_v.png` | VQI-V kurtosis plot |
| `reports/mixing_matrix_heatmap_s.png` | VQI-S mixing matrix |
| `reports/mixing_matrix_heatmap_v.png` | VQI-V mixing matrix |
| `reports/analysis.md` | This file |
