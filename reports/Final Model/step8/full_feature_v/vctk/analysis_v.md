# Step 8: VQI-V + Dual-Score Evaluation Analysis - vctk

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 21.8%
- Reject 30%: FNMR reduction = 21.8%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 20.4%
- Reject 30%: FNMR reduction = 20.4%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 24.1%
- Reject 30%: FNMR reduction = 24.1%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 20.5%
- Reject 30%: FNMR reduction = 20.5%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 21.4%
- Reject 30%: FNMR reduction = 21.4%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 21.8%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 21.6%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 20.4%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 22.1%

### ECAPA2
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 24.1%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 23.4%

### x-vector
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 20.5%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 15.7%

### WavLM-SV
- s_only: FNMR reduction@20% = 0.0%
- v_only: FNMR reduction@20% = 21.4%
- union: FNMR reduction@20% = 0.0%
- intersection: FNMR reduction@20% = 17.6%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0069
- Q2: EER = 0.0144
- Q3: EER = 0.0159
- Q4: EER = 0.0110

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0038
- Q2: EER = 0.0081
- Q3: EER = 0.0097
- Q4: EER = 0.0053

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0032
- Q2: EER = 0.0074
- Q3: EER = 0.0103
- Q4: EER = 0.0061

### x-vector
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0363
- Q2: EER = 0.0615
- Q3: EER = 0.0733
- Q4: EER = 0.0482

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.1120
- Q2: EER = 0.1402
- Q3: EER = 0.1459
- Q4: EER = 0.1293
