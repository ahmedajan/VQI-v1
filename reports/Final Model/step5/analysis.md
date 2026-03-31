# Step 5: Feature Evaluation and Selection - Analysis

**Date:** 2026-02-16
**Pipeline runtime:** ~2.1 minutes

## VQI-S (Signal Quality)

- **Candidates:** 544 -> 513 valid (31 zero-variance: 8 DNSMOS/NISQA + 1 ClickRate + 22 histogram bins)
- **Post-redundancy:** 449 (64 pairs removed at |r| > 0.95)
- **Selected:** 430 features (RF pruning: 2 iterations)
- **OOB accuracy:** 0.8202

### Spearman Correlations
- Features with |rho| > 0.3: 3
- Features with |rho| > 0.2: 6
- Max |rho|: 0.4373 (SpeakerTurns)

### Top 10 Features by RF Importance
| Rank | Feature | Importance | |rho| |
|------|---------|------------|-------|
| 1 | SpeakerTurns | 0.068313 | 0.4373 |
| 2 | FrameSFlux_Hist0 | 0.067423 | 0.4064 |
| 3 | SpeechContinuity | 0.032193 | 0.1984 |
| 4 | DSI | 0.020647 | 0.2879 |
| 5 | RT60_Est | 0.019049 | 0.2380 |
| 6 | FrameSNR_Hist0 | 0.010194 | 0.1625 |
| 7 | C50_Est | 0.010135 | 0.1900 |
| 8 | FrameSC_Hist1 | 0.009348 | 0.1163 |
| 9 | DCOffset | 0.008429 | 0.1511 |
| 10 | Tremor_Intensity | 0.007064 | 0.1838 |

### Importance Concentration
- Top 10 features: 25.3% of total importance
- Top 30 features: 34.1% of total importance

### Top 5 ERC Features (best quality predictors)
| Rank | Feature | ERC AUC |
|------|---------|---------|
| 1 | SubbandSNR_Mid | 0.0149 |
| 2 | QOQ | 0.0178 |
| 3 | SubbandSNR_Low | 0.0185 |
| 4 | NoiseBandwidth | 0.0208 |
| 5 | GlobalVADRatio | 0.0215 |

## VQI-V (Voice Distinctiveness)

- **Candidates:** 161 -> 161 valid (0 zero-variance)
- **Post-redundancy:** 133 (28 pairs removed at |r| > 0.95)
- **Selected:** 133 features (RF pruning: 1 iterations)
- **OOB accuracy:** 0.8242

### Spearman Correlations
- Features with |rho| > 0.2: 1
- Max |rho|: 0.4235 (V_LTFD_Entropy)

### Top 10 Features by RF Importance
| Rank | Feature | Importance | |rho| |
|------|---------|------------|-------|
| 1 | V_LTFD_Entropy | 0.142375 | 0.4235 |
| 2 | V_DeltaMFCC_Mean_1 | 0.031306 | 0.1695 |
| 3 | V_F0_Slope | 0.020160 | 0.1094 |
| 4 | V_DeltaMFCC_Mean_3 | 0.019671 | 0.0870 |
| 5 | V_DeltaMFCC_Mean_6 | 0.017580 | 0.0732 |
| 6 | V_LTAS_500_1000 | 0.013749 | 0.1593 |
| 7 | V_MFCC_Std_1 | 0.013086 | 0.1968 |
| 8 | V_LTFD_Range | 0.013049 | 0.1638 |
| 9 | V_Rhythm_SpeechSegVar | 0.012406 | 0.1618 |
| 10 | V_DeltaMFCC_Mean_5 | 0.012062 | 0.0058 |

### Importance Concentration
- Top 10 features: 29.5% of total importance
- Top 30 features: 48.3% of total importance

### Top 5 ERC Features (best quality predictors)
| Rank | Feature | ERC AUC |
|------|---------|---------|
| 1 | V_Rhythm_SpeechSilenceRatio | 0.0215 |
| 2 | V_Rhythm_VoicedPct | 0.0215 |
| 3 | V_LSF_Mean_2 | 0.0480 |
| 4 | V_LAR_1 | 0.0481 |
| 5 | V_LSF_Mean_3 | 0.0482 |

## VQI-S vs VQI-V Comparison

| Metric | VQI-S | VQI-V |
|--------|-------|-------|
| Candidates | 544 | 161 |
| Zero-variance | 31 | 0 |
| Redundancy removed | 64 | 28 |
| Selected | 430 | 133 |
| OOB accuracy | 0.8202 | 0.8242 |
| Top-10 importance share | 25.3% | 29.5% |

### Key Findings

1. **Low pruning rate:** RF importance pruning removed very few features (19 for S, 0 for V) 
   because all post-redundancy features have importance above the 0.5% threshold. This suggests 
   the redundancy removal stage did most of the work in eliminating uninformative features.

2. **Spearman correlations are modest:** Only 3 VQI-S features and 1 VQI-V feature exceed 
   |rho| > 0.3. This is expected because Fisher d' is a per-speaker metric, while features 
   are per-utterance, and quality affects recognition through complex, non-linear pathways.

3. **VQI-V has higher importance concentration:** The top VQI-V feature (V_LTFD_Entropy) has 
   importance 0.142 vs VQI-S top (SpeakerTurns) at 0.069, suggesting VQI-V relies more 
   heavily on a few key features.

4. **Both scores have strong OOB:** ~82% OOB accuracy for both VQI-S and VQI-V indicates 
   good class separation from the feature sets, supporting viable model training in Step 6.

5. **Feature types:** VQI-S top features span signal (SNR, spectral flux), environment 
   (RT60, C50), temporal (SpeakerTurns, SpeechContinuity) and clinical (DSI). VQI-V top 
   features are dominated by dynamic cepstral features (DeltaMFCC) and long-term 
   distributional features.

## Output Files

### VQI-S (`data/evaluation/`)
- `spearman_correlations.csv` (544 rows)
- `feature_correlation_matrix.npy` (513 x 513)
- `removed_redundant_features.csv` (64 pairs)
- `rf_importance_rankings.csv` (430 features)
- `rf_pruning_history.csv` (iteration details)
- `selected_features.txt` (430 feature names)
- `feature_selection_summary.yaml`
- `erc_per_feature.csv` (430 features)

### VQI-V (`data/evaluation_v/`)
- `spearman_correlations.csv` (161 rows)
- `feature_correlation_matrix.npy` (161 x 161)
- `removed_redundant_features.csv` (28 pairs)
- `rf_importance_rankings.csv` (133 features)
- `rf_pruning_history.csv`
- `selected_features.txt` (133 feature names)
- `feature_selection_summary.yaml`
- `erc_per_feature.csv` (133 features)