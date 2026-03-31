# Step 4: Feature Extraction Analysis

## Overview
- **Training samples:** 20288 (Class 0: 10144, Class 1: 10144)
- **VQI-S features:** 544 (437 frame-level + 107 global)
- **VQI-V features:** 161
- **Total features:** 705

## Data Quality
- **VQI-S NaN/Inf:** 0 (0.0000%)
- **VQI-V NaN/Inf:** 0 (0.0000%)

## Top VQI-S Features by Class Separation (|Cohen's d|)

| Rank | Feature | |Cohen's d| | Direction |
|------|---------|-----------|-----------|
| 1 | SpeakerTurns | 1.123 | C1 > C0 |
| 2 | GlobalDuration | 1.113 | C1 > C0 |
| 3 | DSI | 0.894 | C1 > C0 |
| 4 | FrameSFlux_Hist0 | 0.790 | C0 > C1 |
| 5 | SpeechContinuity | 0.697 | C1 > C0 |
| 6 | FrameE_Mean | 0.606 | C0 > C1 |
| 7 | RT60_Est | 0.602 | C1 > C0 |
| 8 | GlobalEnergy | 0.593 | C0 > C1 |
| 9 | C50_Est | 0.582 | C0 > C1 |
| 10 | ModulationDepth | 0.572 | C1 > C0 |
| 11 | FrameE_P5 | 0.547 | C0 > C1 |
| 12 | FrameE_Median | 0.534 | C0 > C1 |
| 13 | FrameE_Hist8 | 0.532 | C0 > C1 |
| 14 | FrameSNR_Range | 0.519 | C1 > C0 |
| 15 | FrameE_P95 | 0.519 | C0 > C1 |
| 16 | FrameHNR_Hist0 | 0.496 | C1 > C0 |
| 17 | AGC_Activity | 0.482 | C1 > C0 |
| 18 | SubbandSNR_Mid | 0.431 | C0 > C1 |
| 19 | InterruptionCount | 0.424 | C1 > C0 |
| 20 | FrameSC_Hist0 | 0.420 | C0 > C1 |

## Top VQI-V Features by Class Separation (|Cohen's d|)

| Rank | Feature | |Cohen's d| | Direction |
|------|---------|-----------|-----------|
| 1 | V_LTFD_Entropy | 1.444 | C1 > C0 |
| 2 | V_DeltaMFCC_Mean_1 | 0.433 | C1 > C0 |
| 3 | V_LTAS_500_1000 | 0.430 | C0 > C1 |
| 4 | V_F0_Range | 0.420 | C1 > C0 |
| 5 | V_Rhythm_SpeechSegVar | 0.411 | C1 > C0 |
| 6 | V_LTAS_0_500 | 0.408 | C0 > C1 |
| 7 | V_Rhythm_TempoVar | 0.397 | C1 > C0 |
| 8 | V_LTFD_Range | 0.390 | C1 > C0 |
| 9 | V_MFCC_Std_1 | 0.375 | C1 > C0 |
| 10 | V_LTAS_1000_2000 | 0.372 | C0 > C1 |
| 11 | V_Rhythm_nPVI | 0.347 | C1 > C0 |
| 12 | V_Rhythm_SilenceSegVar | 0.331 | C1 > C0 |
| 13 | V_LFCC_Mean_11 | 0.328 | C1 > C0 |
| 14 | V_LTAS_2000_4000 | 0.300 | C0 > C1 |
| 15 | V_F2_Dynamics | 0.273 | C1 > C0 |
| 16 | V_MFCC_Std_5 | 0.262 | C1 > C0 |
| 17 | V_MFCC_Std_3 | 0.259 | C1 > C0 |
| 18 | V_MFCC_Std_6 | 0.255 | C1 > C0 |
| 19 | V_SpectralCentroid_Std | 0.232 | C1 > C0 |
| 20 | V_Rhythm_VoicedPct | 0.231 | C0 > C1 |

## Feature Statistics Summary

### VQI-S
- Mean of means: 103.8305
- Features with zero variance: 31
- Features with |Cohen's d| > 0.5: 15
- Features with |Cohen's d| > 0.2: 134

### VQI-V
- Features with |Cohen's d| > 0.5: 1
- Features with |Cohen's d| > 0.2: 26

## Computation
- Feature extraction time per file: see extraction_log_train.csv
- Modules: 23 frame-level + 32 global + 5 VQI-V = 60 feature modules
- Shared intermediates computed once per file
