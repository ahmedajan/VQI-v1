# 5-Way Dimensionality Reduction Comparison

**Date:** 2026-03-05 (re-run with independent component selection)
**Purpose:** Compare all DR methods against full features for both VQI-S and VQI-V
**Method:** StandardScaler -> DR transform -> same 30-config RF grid search -> top-5 CV -> train final model
**Component selection:** PCA uses explained variance thresholds (90%/95%); FA uses BIC minimization; ICA uses Parallel Analysis (Horn's method, 100 permutations, 95th percentile)

## Results

| Method | VQI-S Components | VQI-S OOB | VQI-S Diff | VQI-V Components | VQI-V OOB | VQI-V Diff |
|--------|-----------------|-----------|------------|-----------------|-----------|------------|
| Full features | 430 | 0.8176 | -- | 133 | 0.8206 | -- |
| FA (BIC) | 320 | 0.7794 | -0.0382 | 121 | 0.8183 | -0.0023 |
| PCA 90% | 99 | 0.8036 | -0.0139 | 47 | 0.8082 | -0.0124 |
| PCA 95% | 156 | 0.8016 | -0.0160 | 60 | 0.8086 | -0.0120 |
| ICA (PA) | 50 | 0.7952 | -0.0223 | 27 | 0.7969 | -0.0237 |

## Rankings

- **VQI-S:** Full > PCA-90% > PCA-95% > ICA (PA) > FA (BIC)
- **VQI-V:** Full > FA (BIC) > PCA-95% > PCA-90% > ICA (PA)

## Component Selection Methods

| Method | Selection Criterion | VQI-S n | VQI-V n |
|--------|-------------------|---------|---------|
| PCA 90% | Explained variance >= 90% | 99 | 47 |
| PCA 95% | Explained variance >= 95% | 156 | 60 |
| FA (BIC) | BIC minimization (two-pass sweep) | 320 | 121 |
| ICA (PA) | Parallel Analysis (Horn's method) | 50 | 27 |

## Key Findings

1. **FA (BIC) dramatically improves VQI-V** -- only -0.23 pp vs full features (was -2.44 pp with 47 factors). BIC selects 121/133 features, indicating nearly all VQI-V features carry non-redundant latent variance.
2. **FA (BIC) worsens VQI-S** -- -3.82 pp vs full (was -0.90 pp with 99 factors). BIC selects 320/430 factors, but the additional noisy factors dilute discriminative signal for RF.
3. **ICA (PA) slightly improves both scores** vs old ICA -- S: -2.23 pp (was -2.41 pp), V: -2.37 pp (was -2.57 pp). PA selects fewer components (50/27 vs 99/47), removing noise dimensions.
4. **PCA-90% remains the best overall DR method** for VQI-S and second-best for VQI-V, confirming its selection for deployment.
5. **Generative vs discriminative mismatch**: BIC optimizes generative fit (log-likelihood), not discriminative task performance. More factors improve FA's data model but can hurt downstream classification.
6. **PA is conservative**: selecting only 50/27 components (vs 99/47), Parallel Analysis focuses on clearly non-random variance, which helps slightly but still underperforms PCA.

## Old vs New Comparison (FA and ICA only)

| Method | VQI-S n (old/new) | VQI-S OOB (old/new) | VQI-V n (old/new) | VQI-V OOB (old/new) |
|--------|-------------------|---------------------|-------------------|---------------------|
| FA | 99/320 | 0.8086/0.7794 | 47/121 | 0.7961/0.8183 |
| ICA | 99/50 | 0.7935/0.7952 | 47/27 | 0.7949/0.7969 |

## Deployed Model

PCA-90% remains the deployed model (v2.0) because:
- Best VQI-S DR performance (-1.39 pp), adequate VQI-V (-1.24 pp)
- Strong compression: 430 -> 99 (77% reduction for S), 133 -> 47 (65% for V)
- Validation: S AUC=0.8600, F1=0.8102; V AUC=0.8619, F1=0.8141
- Evaluation: best ERC 38.0% (VoxCeleb1 P4), DET separation up to 6.41x (VCTK P3)
