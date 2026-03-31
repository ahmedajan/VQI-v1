# Step 8: VQI-V + Dual-Score Evaluation Analysis - voxceleb1

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.2%
- Reject 20%: FNMR reduction = 7.6%
- Reject 30%: FNMR reduction = 13.3%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 6.9%
- Reject 20%: FNMR reduction = 9.2%
- Reject 30%: FNMR reduction = 15.6%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.2%
- Reject 20%: FNMR reduction = 3.7%
- Reject 30%: FNMR reduction = 6.8%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.9%
- Reject 20%: FNMR reduction = 18.0%
- Reject 30%: FNMR reduction = 25.3%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.3%
- Reject 20%: FNMR reduction = 8.2%
- Reject 30%: FNMR reduction = 13.3%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 11.3%
- v_only: FNMR reduction@20% = 7.6%
- union: FNMR reduction@20% = 8.3%
- intersection: FNMR reduction@20% = 12.4%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 13.1%
- v_only: FNMR reduction@20% = 9.2%
- union: FNMR reduction@20% = 10.1%
- intersection: FNMR reduction@20% = 14.3%

### ECAPA2
- s_only: FNMR reduction@20% = 8.3%
- v_only: FNMR reduction@20% = 3.7%
- union: FNMR reduction@20% = 4.3%
- intersection: FNMR reduction@20% = 8.2%

### x-vector
- s_only: FNMR reduction@20% = 20.9%
- v_only: FNMR reduction@20% = 18.0%
- union: FNMR reduction@20% = 19.0%
- intersection: FNMR reduction@20% = 23.5%

### WavLM-SV
- s_only: FNMR reduction@20% = 12.8%
- v_only: FNMR reduction@20% = 8.2%
- union: FNMR reduction@20% = 8.4%
- intersection: FNMR reduction@20% = 15.5%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0399
- Q2: EER = 0.0390
- Q3: EER = 0.0591
- Q4: EER = 0.0520

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0403
- Q2: EER = 0.0394
- Q3: EER = 0.0583
- Q4: EER = 0.0513

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0401
- Q2: EER = 0.0383
- Q3: EER = 0.0576
- Q4: EER = 0.0515

### x-vector
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0740
- Q2: EER = 0.0839
- Q3: EER = 0.1135
- Q4: EER = 0.0912

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0872
- Q2: EER = 0.0789
- Q3: EER = 0.0937
- Q4: EER = 0.0830
