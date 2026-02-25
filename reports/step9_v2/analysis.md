# Step 9 v2: VQI Software v2.0 (PCA-90%) Analysis

## Conformance Set Overview
- **Files scored:** 200
- **VQI-S:** min=13, max=91, mean=48.1, std=17.7
- **VQI-V:** min=7, max=95, mean=46.5, std=23.5
- **Correlation (S vs V):** r = 0.839

## v1.0 vs v2.0 Comparison
- **Files compared:** 200
- **VQI-S:** Pearson r = 0.924, MAE = 10.2, mean diff = +1.9 +/- 11.5
- **VQI-V:** Pearson r = 0.934, MAE = 6.7, mean diff = +2.4 +/- 8.5

## Interpretation
The PCA-90% model (v2.0) uses 77% fewer features for VQI-S (430->99 PCs) and 65% fewer for VQI-V (133->47 PCs). The high correlation with v1.0 scores confirms that the dimensionality reduction preserves the essential quality signal.

## Processing Time
- **Mean:** 2545 ms
- **Median:** 1739 ms
- **Min:** 887 ms
- **Max:** 11355 ms
