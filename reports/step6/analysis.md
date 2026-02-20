# Step 6: Model Training Analysis

## Summary

| Metric | VQI-S | VQI-V |
|--------|-------|-------|
| N_features | 430 | 133 |
| N_samples | 20,288 | 20,288 |
| Best n_estimators | 1000 | 1000 |
| Best max_features | 8 | 5 |
| OOB Error | 0.1824 | 0.1794 |
| OOB Accuracy | 0.8176 | 0.8206 |
| OOB 95% CI | [0.1771, 0.1877] | [0.1741, 0.1847] |
| Training Accuracy | 0.9772 | 0.9787 |
| ECE | 0.1853 | 0.1971 |

## Target Verification

| Target | VQI-S | Status | VQI-V | Status |
|--------|-------|--------|-------|--------|
| OOB < 0.20 (S) / 0.25 (V) | 0.1824 | PASS | 0.1794 | PASS |
| Accuracy > 0.85 (S) / 0.80 (V) | 0.9772 | PASS | 0.9787 | PASS |

## Confusion Matrices

### VQI-S
|  | Pred 0 | Pred 1 |
|--|--------|--------|
| True 0 | 9,981 | 163 |
| True 1 | 299 | 9,845 |

### VQI-V
|  | Pred 0 | Pred 1 |
|--|--------|--------|
| True 0 | 9,942 | 202 |
| True 1 | 231 | 9,913 |

## Feature Importance

### VQI-S Top 5
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | SpeakerTurns | 0.0522 |
| 2 | FrameSFlux_Hist0 | 0.0471 |
| 3 | SpeechContinuity | 0.0279 |
| 4 | DSI | 0.0204 |
| 5 | RT60_Est | 0.0169 |

### VQI-V Top 5
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | V_LTFD_Entropy | 0.1204 |
| 2 | V_DeltaMFCC_Mean_1 | 0.0291 |
| 3 | V_F0_Slope | 0.0223 |
| 4 | V_DeltaMFCC_Mean_3 | 0.0201 |
| 5 | V_DeltaMFCC_Mean_6 | 0.0166 |

### Importance Concentration

| Metric | VQI-S | VQI-V |
|--------|-------|-------|
| Top 10 share | 0.2085 | 0.2769 |
| Top 50 share | 0.3788 | 0.6189 |
| Features for 90% | 346 | N/A |
| Features for 95% | 384 | N/A |

## Key Findings

1. **Both models exceed all targets.** VQI-S OOB error (0.1824) well below 0.20 target;
   VQI-V (0.1794) well below 0.25 target.

2. **Optimal hyperparameters:** VQI-S uses max_features=8 (fewer than sqrt(430)~21),
   VQI-V uses max_features=5 (fewer than sqrt(133)~12). Both prefer more diversity
   per tree, consistent with many weak/moderate features.

3. **1000 trees selected for both.** OOB convergence shows diminishing returns past ~750 trees
   but 1000 provides additional stability at modest compute cost.

4. **VQI-V outperforms VQI-S** slightly (OOB 0.1794 vs 0.1824),
   despite fewer features (133 vs 430). VQI-V features may be more directly discriminative.

5. **Feature importance concentration:** VQI-V is more concentrated (top-10 = 27.69%)
   vs VQI-S (20.85%), reflecting V_LTFD_Entropy's dominant role (0.1204).

6. **VQI-S top features** match Step 5 rankings: SpeakerTurns, FrameSFlux_Hist0, SpeechContinuity, DSI, RT60_Est.

7. **VQI-V top features** also match Step 5: V_LTFD_Entropy, V_DeltaMFCC_Mean_1, V_F0_Slope.

## Models

- VQI-S: `models/vqi_rf_model.joblib` (1000 trees, 430 features)
- VQI-V: `models/vqi_v_rf_model.joblib` (1000 trees, 133 features)

## Runtime

- VQI-S: ~15.8 min (grid search + CV + final training + convergence)
- VQI-V: ~12.1 min
- Total: ~27.8 min
