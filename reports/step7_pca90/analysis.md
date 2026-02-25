# Step 7: Model Validation Analysis (PCA-90%)

## Summary

Step 7 validates the VQI-S and VQI-V Random Forest models trained with
PCA-90% dimensionality reduction on a held-out set of 50,000 samples.
This validation confirms that higher VQI scores correspond to higher
speaker recognition utility, even with reduced feature dimensionality.

## VQI-S Validation Results (PCA-90%)

| Metric | Value |
|--------|-------|
| Accuracy | 0.759205 |
| Precision | 0.924503 |
| Recall | 0.721074 |
| F1-Score | 0.810215 |
| AUC-ROC | 0.859985 |
| Youden's J | 46.0 |

## VQI-V Validation Results (PCA-90%)

| Metric | Value |
|--------|-------|
| Accuracy | 0.759941 |
| Precision | 0.908397 |
| Recall | 0.737603 |
| F1-Score | 0.814139 |
| AUC-ROC | 0.861933 |
| Youden's J | 43.0 |

## Dual-Score Quadrant Analysis (PCA-90%)

| Quadrant | Count (%) | Class 1 Rate | Mean S | Mean V |
|----------|-----------|-------------|--------|--------|
| Q1 (High S, High V) | 11,536 (23.1%) | 93.9% | 61.56 | 64.45 |
| Q2 (Low S, High V) | 8,714 (17.4%) | 72.6% | 41.63 | 52.23 |
| Q3 (Low S, Low V) | 27,344 (54.7%) | 30.9% | 34.56 | 27.63 |
| Q4 (High S, Low V) | 2,406 (4.8%) | 69.6% | 54.31 | 35.0 |

## Key Findings

1. **CDF Shift Validation**: Higher VQI scores correspond to higher genuine
   comparison scores across all three providers, confirming predictive validity
   even with PCA-90% reduced features.

2. **Confusion Matrix**: Both PCA-90% models achieve acceptable accuracy on the
   labeled validation subset, with AUC-ROC indicating good discrimination.

3. **Dual-Score Analysis**: The 2D scatter reveals four distinct quadrants
   of failure modes. Q1 (high S, high V) has the highest recognition success
   rate, while Q3 (low S, low V) has the lowest.

4. **Combined Rejection**: Using both scores (intersection strategy) provides
   the most conservative but highest-accuracy rejection policy. The union
   strategy catches more failures at the cost of higher rejection rates.

5. **OOB Convergence**: Both PCA-90% models converged well before the selected
   number of trees, confirming sufficient ensemble size.

6. **CV Stability**: Cross-validation standard deviation < 0.03 for both
   models, indicating stable generalization.

7. **Dimensionality Reduction Impact**: PCA-90% reduces features from 430->99
   (VQI-S) and 133->47 (VQI-V) while retaining most predictive power.

## Visualizations

- Score distributions (2 plots)
- CDF per quality bin per provider (6 plots)
- Confusion matrices (2 plots)
- OOB convergence (2 plots)
- ROC curves (2 plots)
- Score vs genuine scatter (2 plots)
- Dual-score scatter (1 plot)
- Quadrant bar chart (1 plot)
- Combined rejection curve (1 plot)

Total: 19 plots + 2 validation reports + this analysis
