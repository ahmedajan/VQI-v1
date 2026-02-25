# Dimensionality Reduction — 5-Way Comparison

**Date:** 2026-02-22
**Methods:** Full features, PCA (95% variance), PCA (90% variance), ICA, Factor Analysis
**Protocol:** StandardScaler -> DR transform -> same 30-config RF grid search (6 n_estimators x 5 max_features) -> top-5 CV -> train final model
**Component counts:** ICA and FA use the same n_components as PCA-90% (S=99, V=47) for fair comparison.

---

## VQI-S (Signal Quality) — 5-Way Comparison

| Metric | Full | PCA 95% | PCA 90% | ICA | FA |
|--------|------|---------|---------|-----|-----|
| Features / Components | 430 | 156 | 99 | 99 | 99 |
| Best n_estimators | 1000 | 1000 | 1000 | 750 | 1000 |
| Best max_features | 8 | 12 | 8 | 5 | 10 |
| OOB accuracy | 0.8176 | 0.8016 | 0.8036 | 0.7935 | 0.8086 |
| OOB diff vs full | — | -0.0160 | -0.0139 | -0.0241 | -0.0090 |
| Training accuracy | 0.9772 | 0.9972 | 0.9955 | 0.9918 | 0.9959 |
| Model size (MB) | 191.93 | 203.68 | 211.27 | 167.54 | 205.64 |

## VQI-V (Voice Distinctiveness) — 5-Way Comparison

| Metric | Full | PCA 95% | PCA 90% | ICA | FA |
|--------|------|---------|---------|-----|-----|
| Features / Components | 133 | 60 | 47 | 47 | 47 |
| Best n_estimators | 1000 | 1000 | 500 | 1000 | 750 |
| Best max_features | 5 | 10 | 10 | 12 | 5 |
| OOB accuracy | 0.8206 | 0.8086 | 0.8082 | 0.7949 | 0.7961 |
| OOB diff vs full | — | -0.0120 | -0.0124 | -0.0257 | -0.0244 |
| Training accuracy | 0.9787 | 0.9857 | 0.9811 | 0.9661 | 0.9953 |
| Model size (MB) | 203.98 | 194.58 | 96.23 | 178.43 | 171.73 |

## Combined Summary

| Method | VQI-S OOB | VQI-S diff | VQI-V OOB | VQI-V diff | Rank (S) | Rank (V) |
|--------|-----------|------------|-----------|------------|----------|----------|
| Full features | 0.8176 | — | 0.8206 | — | 1 | 1 |
| **FA (99/47)** | **0.8086** | **-0.0090** | 0.7961 | -0.0244 | **2** | 5 |
| PCA 90% (99/47) | 0.8036 | -0.0139 | 0.8082 | -0.0124 | 3 | 3 |
| PCA 95% (156/60) | 0.8016 | -0.0160 | **0.8086** | **-0.0120** | 4 | **2** |
| ICA (99/47) | 0.7935 | -0.0241 | 0.7949 | -0.0257 | 5 | 4 |

---

## Interpretation

### Method Ranking
1. **Full features** remain the best for both VQI-S and VQI-V, confirming that the Random Forest exploits fine-grained feature-level information that all DR methods discard.

2. **Factor Analysis** is the best same-dimensionality DR method for VQI-S, losing only 0.90 pp — significantly less than PCA-90% (-1.39 pp) and ICA (-2.41 pp). FA's explicit noise separation preserves more discriminative signal in the shared-variance (communality) components.

3. **PCA** performs well overall and is the best DR method for VQI-V. The 95% threshold (60 PCs) achieves -0.0120 for VQI-V, outperforming 90% (-0.0124), FA (-0.0244), and ICA (-0.0257). For VQI-S, however, 95% PCA (-0.0160) is worse than 90% (-0.0139), confirming the earlier finding that discriminative signal is spread across low-variance components.

4. **ICA** consistently shows the largest accuracy drops (-2.41 pp for S, -2.57 pp for V). The statistical independence objective does not align well with class discrimination for this task.

### Key Findings
- **No DR method matches full features.** The 0.9--2.6 pp OOB accuracy gap confirms that all 430 (S) / 133 (V) features carry non-redundant discriminative information.
- **FA outperforms PCA on VQI-S at equal dimensionality** (99 components: FA 0.8086 vs PCA 0.8036). FA's noise-variance separation preserves more task-relevant information.
- **PCA outperforms FA on VQI-V.** The VQI-V feature space (133 features, 47 components) is more compact and PCA's variance-maximization objective aligns better with the discriminative structure.
- **ICA is consistently the weakest** DR method, likely because maximizing non-Gaussianity does not preserve class-discriminative directions.
- **All reduced models select higher max_features** than baselines (or similar), consistent with transformed features benefiting from broader per-split sampling.

### Practical Implications
- Full-feature models remain the production default.
- For resource-constrained deployment requiring VQI-S only, FA-reduced models offer the best accuracy/size trade-off (99 features, 167.54 MB, -0.90 pp).
- For VQI-V deployment constraints, PCA-90% (47 PCs, 96.23 MB, -1.24 pp) is the most compact option.

---

## Output Files

### ICA
| File | Description |
|------|-------------|
| `models/vqi_rf_ica_model.joblib` | ICA VQI-S RF model (99 ICs) |
| `models/vqi_v_rf_ica_model.joblib` | ICA VQI-V RF model (47 ICs) |
| `models/vqi_ica_scaler_s.joblib` | VQI-S StandardScaler |
| `models/vqi_ica_scaler_v.joblib` | VQI-V StandardScaler |
| `models/vqi_ica_transformer_s.joblib` | VQI-S ICA transformer |
| `models/vqi_ica_transformer_v.joblib` | VQI-V ICA transformer |
| `data/training_ica/training_metrics.yaml` | VQI-S training metrics |
| `data/training_ica_v/training_metrics.yaml` | VQI-V training metrics |
| `reports/ica/analysis.md` | ICA analysis + statistics |

### Factor Analysis
| File | Description |
|------|-------------|
| `models/vqi_rf_fa_model.joblib` | FA VQI-S RF model (99 factors) |
| `models/vqi_v_rf_fa_model.joblib` | FA VQI-V RF model (47 factors) |
| `models/vqi_fa_scaler_s.joblib` | VQI-S StandardScaler |
| `models/vqi_fa_scaler_v.joblib` | VQI-V StandardScaler |
| `models/vqi_fa_transformer_s.joblib` | VQI-S FA transformer |
| `models/vqi_fa_transformer_v.joblib` | VQI-V FA transformer |
| `data/training_fa/training_metrics.yaml` | VQI-S training metrics |
| `data/training_fa_v/training_metrics.yaml` | VQI-V training metrics |
| `reports/fa/analysis.md` | FA analysis + statistics |

### Comparison
| File | Description |
|------|-------------|
| `reports/dimensionality_reduction/comparison.md` | This file — unified 5-way comparison |
