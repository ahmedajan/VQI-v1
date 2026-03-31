# PCA Effective Dimensionality Analysis

**Date:** 2026-02-21
**Purpose:** Quantify redundancy in VQI-S (430 features) and VQI-V (133 features) spaces

## Summary Table

| Metric | VQI-S | VQI-V |
|--------|-------|-------|
| Total features | 430 | 133 |
| PCs for 90% variance | 99 | 47 |
| PCs for 95% variance | 156 | 60 |
| PCs for 99% variance | 268 | 86 |
| Effective dim ratio (95%) | 0.3628 | 0.4511 |
| PC1 variance share | 0.1633 | 0.1317 |
| Top-10 PCs variance | 0.5663 | 0.5686 |

## Key Finding

**VQI-S:** 156 principal components capture 95% of the variance in 430 features — an effective dimensionality ratio of 36.3%. The 90% threshold is reached at just 99 PCs (23.0% of features).

**VQI-V:** 60 principal components capture 95% of the variance in 133 features — an effective dimensionality ratio of 45.1%. The 90% threshold is reached at 47 PCs (35.3% of features).

## Interpretation

The PCA analysis reveals substantial redundancy in both feature spaces, even after the 0.95 correlation threshold removal in Step 5. For VQI-S, approximately 274 of 430 features (64%) contribute less than 5% of total variance — these features largely describe the same underlying dimensions as the top 156 PCs. For VQI-V, the picture is similar: 73 of 133 features (55%) are redundant at the 95% level.

This is consistent with the feature selection analysis showing accuracy plateaus: OOB 0.815 at k=200 vs 0.822 at k=430. The marginal features add noise rather than new information. However, Random Forests are robust to redundant features (random feature subsampling at each split naturally de-correlates trees), so the redundancy does not necessarily harm model performance — it just means a smaller feature set could achieve similar accuracy.

## Variance Explained Detail

### VQI-S — Top 10 Components

| PC | Variance Ratio | Cumulative |
|-----|---------------|------------|
| PC1 | 0.1633 | 0.1633 |
| PC2 | 0.1246 | 0.2879 |
| PC3 | 0.0882 | 0.3762 |
| PC4 | 0.0476 | 0.4238 |
| PC5 | 0.0340 | 0.4578 |
| PC6 | 0.0257 | 0.4835 |
| PC7 | 0.0238 | 0.5073 |
| PC8 | 0.0222 | 0.5294 |
| PC9 | 0.0185 | 0.5480 |
| PC10 | 0.0184 | 0.5663 |

### VQI-V — Top 10 Components

| PC | Variance Ratio | Cumulative |
|-----|---------------|------------|
| PC1 | 0.1317 | 0.1317 |
| PC2 | 0.1123 | 0.2440 |
| PC3 | 0.0714 | 0.3154 |
| PC4 | 0.0617 | 0.3771 |
| PC5 | 0.0472 | 0.4243 |
| PC6 | 0.0339 | 0.4582 |
| PC7 | 0.0322 | 0.4904 |
| PC8 | 0.0293 | 0.5196 |
| PC9 | 0.0254 | 0.5450 |
| PC10 | 0.0236 | 0.5686 |

## Output Files

- `variance_curve_s.png` — cumulative variance for VQI-S
- `variance_curve_v.png` — cumulative variance for VQI-V
- `variance_curve_combined.png` — both on one chart
- `scree_plot_s.png` — per-component variance for VQI-S (top 50)
- `scree_plot_v.png` — per-component variance for VQI-V (top 50)
- `analysis.md` — this file