# Step 7 Validation Analysis (v4.0)

## VQI-S (Signal Quality)
Ridge Regressor on full 430 features with StandardScaler.
- AUC-ROC: 0.8812
- F1: 0.8372
- Score range: [0, 100]
- Score mean: 56.1 +/- 29.1

## VQI-V (Voice Distinctiveness)
XGBoost Regressor on full 133 features with StandardScaler.
- AUC-ROC: 0.9122
- F1: 0.8819
- Score range: [0, 100]
- Score mean: 59.7 +/- 31.8

## Dual-Score Analysis
Both scores are computed independently. VQI-S captures signal quality,
VQI-V captures voice distinctiveness. The 2D scatter shows quadrant separation.

## Provider CDF Analysis
CDF plots show genuine similarity score distributions grouped by VQI quality
quartile (Bottom 25%, Middle 50%, Top 25%). Higher VQI scores should
correspond to higher genuine similarity, evidenced by rightward CDF shift.

## Notes
- v4.0 uses Ridge (S) + XGBoost (V) regressors — no OOB convergence plots
- Cross-validation stability was assessed during model selection (Step X1)
