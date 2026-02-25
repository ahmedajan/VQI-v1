# Step 8 (PCA-90%): VQI-V + Dual-Score Evaluation Analysis - vctk

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 7.1%
- Reject 20%: FNMR reduction = 15.6%
- Reject 30%: FNMR reduction = 19.6%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.8%
- Reject 20%: FNMR reduction = 13.3%
- Reject 30%: FNMR reduction = 16.4%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 6.7%
- Reject 20%: FNMR reduction = 13.6%
- Reject 30%: FNMR reduction = 18.0%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 10.3%
- Reject 20%: FNMR reduction = 18.3%
- Reject 30%: FNMR reduction = 22.0%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 12.1%
- Reject 20%: FNMR reduction = 21.3%
- Reject 30%: FNMR reduction = 24.0%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 18.2%
- v_only: FNMR reduction@20% = 15.6%
- union: FNMR reduction@20% = 15.6%
- intersection: FNMR reduction@20% = 18.6%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 18.9%
- v_only: FNMR reduction@20% = 13.3%
- union: FNMR reduction@20% = 13.3%
- intersection: FNMR reduction@20% = 19.3%

### ECAPA2
- s_only: FNMR reduction@20% = 15.4%
- v_only: FNMR reduction@20% = 13.6%
- union: FNMR reduction@20% = 13.6%
- intersection: FNMR reduction@20% = 15.6%

### x-vector
- s_only: FNMR reduction@20% = 7.9%
- v_only: FNMR reduction@20% = 18.3%
- union: FNMR reduction@20% = 18.3%
- intersection: FNMR reduction@20% = 8.2%

### WavLM-SV
- s_only: FNMR reduction@20% = 8.6%
- v_only: FNMR reduction@20% = 21.3%
- union: FNMR reduction@20% = 21.3%
- intersection: FNMR reduction@20% = 9.8%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0071
- Q2: EER = 0.0148
- Q3: EER = 0.0176
- Q4: EER = 0.0093

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0040
- Q2: EER = 0.0083
- Q3: EER = 0.0099
- Q4: EER = 0.0055

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0042
- Q2: EER = 0.0086
- Q3: EER = 0.0105
- Q4: EER = 0.0051

### x-vector
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0379
- Q2: EER = 0.0590
- Q3: EER = 0.0789
- Q4: EER = 0.0507

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.1154
- Q2: EER = 0.1373
- Q3: EER = 0.1518
- Q4: EER = 0.1319
