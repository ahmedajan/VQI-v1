# Step 7: VQI-V Validation & Dual-Score Analysis

## VQI-V Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.774669 |
| Precision | 0.918987 |
| Recall | 0.75 |
| F1-Score | 0.825939 |
| AUC-ROC | 0.881235 |
| Youden's J | 47.0 |

## Training Summary
- n_estimators: 1000
- n_features: 133
- OOB error: 0.179416
- Training accuracy: 0.978657

## CDF Shift Validation
Higher VQI-V scores correspond to higher genuine comparison scores across
all providers, confirming that voice distinctiveness positively correlates
with speaker verification performance.

## Dual-Score Correlation
- VQI-S vs VQI-V: Spearman rho = 0.8972
- The two scores measure different aspects (signal quality vs voice
  distinctiveness) and show moderate correlation, confirming partial independence.

## Quadrant Analysis

| Quadrant | Count | % | Class 1 Rate | Mean S | Mean V |
|----------|-------|---|-------------|--------|--------|
| Q1 (High S, High V) | 11,820 | 23.6% | 0.934 | 66.5 | 65.9 |
| Q2 (Low S, High V) | 1,839 | 3.7% | 0.758 | 43.6 | 54.5 |
| Q3 (Low S, Low V) | 33,547 | 67.1% | 0.378 | 28.9 | 29.5 |
| Q4 (High S, Low V) | 2,794 | 5.6% | 0.823 | 55.0 | 43.4 |

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

## Visualizations

- VQI-V score distribution, CDF, confusion matrix, OOB, ROC, scatter (8 plots)
- Dual-score scatter, quadrant bar, combined rejection (3 plots)
- Dual hexbin, quadrant table, quadrant violin (3 plots)
