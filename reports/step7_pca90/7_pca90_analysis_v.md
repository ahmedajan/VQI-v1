# Step 7: VQI-V Validation & Dual-Score Analysis (PCA-90%)

## VQI-V Model Performance (PCA-90%)

| Metric | Value |
|--------|-------|
| Accuracy | 0.759941 |
| Precision | 0.908397 |
| Recall | 0.737603 |
| F1-Score | 0.814139 |
| AUC-ROC | 0.861933 |
| Youden's J | 43.0 |

## Training Summary (PCA-90%)
- n_estimators: 500
- n_features: 47
- OOB error: 0.191838
- Training accuracy: 0.981122

## CDF Shift Validation
Higher VQI-V scores correspond to higher genuine comparison scores across
all providers, confirming that voice distinctiveness positively correlates
with speaker verification performance even with PCA-90% reduced features.

## Dual-Score Correlation (PCA-90%)
- VQI-S vs VQI-V: Spearman rho = 0.7322
- The two scores measure different aspects (signal quality vs voice
  distinctiveness) and show moderate correlation, confirming partial independence.

## Quadrant Analysis (PCA-90%)

| Quadrant | Count | % | Class 1 Rate | Mean S | Mean V |
|----------|-------|---|-------------|--------|--------|
| Q1 (High S, High V) | 11,536 | 23.1% | 0.939 | 61.6 | 64.5 |
| Q2 (Low S, High V) | 8,714 | 17.4% | 0.726 | 41.6 | 52.2 |
| Q3 (Low S, Low V) | 27,344 | 54.7% | 0.308 | 34.6 | 27.6 |
| Q4 (High S, Low V) | 2,406 | 4.8% | 0.696 | 54.3 | 35.0 |

## Key Findings

1. **Q1 (High S, High V)** achieves the highest Class 1 rate, confirming
   that both signal quality and voice distinctiveness contribute to
   speaker recognition success.

2. **Q3 (Low S, Low V)** has the lowest Class 1 rate, representing the
   most challenging samples for speaker verification.

3. **Q2 vs Q4 asymmetry**: Comparing Q2 (Low S, High V) vs Q4 (High S,
   Low V) reveals whether signal quality or voice distinctiveness has
   more impact on recognition.

4. **Combined rejection strategies**: The intersection strategy (reject
   if EITHER score is low) is more conservative but achieves higher
   accuracy among accepted samples.

5. **PCA-90% Impact**: Dimensionality reduction retains the core
   predictive structure while reducing VQI-V features from 133 to 47.

## Visualizations

- VQI-V score distribution, CDF, confusion matrix, OOB, ROC, scatter (8 plots)
- Dual-score scatter, quadrant bar, combined rejection (3 plots)
- Dual hexbin, quadrant table, quadrant violin (3 plots)
