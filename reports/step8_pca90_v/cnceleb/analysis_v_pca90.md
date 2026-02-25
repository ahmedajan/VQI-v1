# Step 8 (PCA-90%): VQI-V + Dual-Score Evaluation Analysis - cnceleb

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.3%
- Reject 20%: FNMR reduction = 2.2%
- Reject 30%: FNMR reduction = 4.1%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.1%
- Reject 20%: FNMR reduction = 0.6%
- Reject 30%: FNMR reduction = 0.9%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.8%
- Reject 20%: FNMR reduction = 1.7%
- Reject 30%: FNMR reduction = 2.7%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = -1.8%
- Reject 20%: FNMR reduction = -2.4%
- Reject 30%: FNMR reduction = -3.1%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.5%
- Reject 20%: FNMR reduction = 2.1%
- Reject 30%: FNMR reduction = 2.8%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 6.0%
- v_only: FNMR reduction@20% = 2.2%
- union: FNMR reduction@20% = 1.9%
- intersection: FNMR reduction@20% = 6.8%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 4.0%
- v_only: FNMR reduction@20% = 0.6%
- union: FNMR reduction@20% = 0.3%
- intersection: FNMR reduction@20% = 4.5%

### ECAPA2
- s_only: FNMR reduction@20% = 6.0%
- v_only: FNMR reduction@20% = 1.7%
- union: FNMR reduction@20% = 2.1%
- intersection: FNMR reduction@20% = 6.9%

### x-vector
- s_only: FNMR reduction@20% = 0.7%
- v_only: FNMR reduction@20% = -2.4%
- union: FNMR reduction@20% = -2.4%
- intersection: FNMR reduction@20% = 1.0%

### WavLM-SV
- s_only: FNMR reduction@20% = 3.9%
- v_only: FNMR reduction@20% = 2.1%
- union: FNMR reduction@20% = 2.1%
- intersection: FNMR reduction@20% = 5.3%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.2028
- Q2: EER = 0.2179
- Q3: EER = 0.2274
- Q4: EER = 0.1996

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.1697
- Q2: EER = 0.2068
- Q3: EER = 0.2233
- Q4: EER = 0.1784

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.2008
- Q2: EER = 0.2038
- Q3: EER = 0.2161
- Q4: EER = 0.1896

### x-vector
- Q1 EER < Q3 EER: True
- Q1: EER = 0.2323
- Q2: EER = 0.2673
- Q3: EER = 0.2790
- Q4: EER = 0.2469

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.2661
- Q2: EER = 0.3184
- Q3: EER = 0.3413
- Q4: EER = 0.2971
