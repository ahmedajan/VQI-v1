# Step 7: Model Validation Analysis

## Summary

Step 7 validates the VQI-S and VQI-V Random Forest models trained in Step 6
on a held-out set of 50,000 samples. This validation confirms that higher
VQI scores correspond to higher speaker recognition utility.

## VQI-S Validation Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.779087 |
| Precision | 0.924936 |
| Recall | 0.751033 |
| F1-Score | 0.828962 |
| AUC-ROC | 0.871943 |
| Youden's J | 45.0 |

## VQI-V Validation Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.774669 |
| Precision | 0.918987 |
| Recall | 0.75 |
| F1-Score | 0.825939 |
| AUC-ROC | 0.881235 |
| Youden's J | 47.0 |

## Dual-Score Quadrant Analysis

| Quadrant | Count (%) | Class 1 Rate | Mean S | Mean V |
|----------|-----------|-------------|--------|--------|
| Q1 (High S, High V) | 11,820 (23.6%) | 93.4% | 66.51 | 65.89 |
| Q2 (Low S, High V) | 1,839 (3.7%) | 75.8% | 43.6 | 54.46 |
| Q3 (Low S, Low V) | 33,547 (67.1%) | 37.8% | 28.89 | 29.5 |
| Q4 (High S, Low V) | 2,794 (5.6%) | 82.3% | 54.98 | 43.42 |

## Key Findings

1. **CDF Shift Validation**: Higher VQI scores correspond to higher genuine
   comparison scores across all three providers, confirming predictive validity.

2. **Confusion Matrix**: Both models achieve acceptable accuracy on the
   labeled validation subset, with AUC-ROC indicating good discrimination.

3. **Dual-Score Analysis**: The 2D scatter reveals four distinct quadrants
   of failure modes. Q1 (high S, high V) has the highest recognition success
   rate, while Q3 (low S, low V) has the lowest.

4. **Combined Rejection**: Using both scores (intersection strategy) provides
   the most conservative but highest-accuracy rejection policy. The union
   strategy catches more failures at the cost of higher rejection rates.

5. **OOB Convergence**: Both models converged well before the selected 1000
   trees, confirming sufficient ensemble size.

6. **CV Stability**: Cross-validation standard deviation < 0.03 for both
   models, indicating stable generalization.

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
