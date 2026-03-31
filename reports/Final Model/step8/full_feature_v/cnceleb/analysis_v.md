# Step 8: VQI-V + Dual-Score Evaluation Analysis - cnceleb

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 12.6%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.5%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 9.2%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 7.7%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 5.6%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 0.0%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 18.1%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 0.0%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 5.5%

### ECAPA2
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 0.0%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 17.1%

### x-vector
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 0.0%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 13.5%

### WavLM-SV
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 0.0%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 12.9%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: None
- Q1: EER = 0.2059
- Q2: EER = N/A
- Q3: EER = N/A
- Q4: EER = 0.2244

### ResNetSE34V2
- Q1 EER < Q3 EER: None
- Q1: EER = 0.1798
- Q2: EER = N/A
- Q3: EER = N/A
- Q4: EER = 0.2124

### ECAPA2
- Q1 EER < Q3 EER: None
- Q1: EER = 0.2027
- Q2: EER = N/A
- Q3: EER = N/A
- Q4: EER = 0.2125

### x-vector
- Q1 EER < Q3 EER: None
- Q1: EER = 0.2423
- Q2: EER = N/A
- Q3: EER = N/A
- Q4: EER = 0.2683

### WavLM-SV
- Q1 EER < Q3 EER: None
- Q1: EER = 0.2796
- Q2: EER = N/A
- Q3: EER = N/A
- Q4: EER = 0.3327
