# Step 8 (PCA-90%): VQI-V + Dual-Score Evaluation Analysis - voxceleb1

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.8%
- Reject 20%: FNMR reduction = 1.4%
- Reject 30%: FNMR reduction = 4.7%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 3.8%
- Reject 20%: FNMR reduction = 5.8%
- Reject 30%: FNMR reduction = 8.1%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.3%
- Reject 20%: FNMR reduction = -1.2%
- Reject 30%: FNMR reduction = -1.0%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 6.7%
- Reject 20%: FNMR reduction = 9.3%
- Reject 30%: FNMR reduction = 15.9%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.2%
- Reject 20%: FNMR reduction = 1.7%
- Reject 30%: FNMR reduction = 5.4%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 12.9%
- v_only: FNMR reduction@20% = 1.4%
- union: FNMR reduction@20% = 2.9%
- intersection: FNMR reduction@20% = 10.9%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 15.9%
- v_only: FNMR reduction@20% = 5.8%
- union: FNMR reduction@20% = 7.2%
- intersection: FNMR reduction@20% = 13.9%

### ECAPA2
- s_only: FNMR reduction@20% = 8.3%
- v_only: FNMR reduction@20% = -1.2%
- union: FNMR reduction@20% = -0.1%
- intersection: FNMR reduction@20% = 6.9%

### x-vector
- s_only: FNMR reduction@20% = 25.1%
- v_only: FNMR reduction@20% = 9.3%
- union: FNMR reduction@20% = 11.2%
- intersection: FNMR reduction@20% = 23.4%

### WavLM-SV
- s_only: FNMR reduction@20% = 12.7%
- v_only: FNMR reduction@20% = 1.7%
- union: FNMR reduction@20% = 3.1%
- intersection: FNMR reduction@20% = 11.1%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0451
- Q2: EER = 0.0467
- Q3: EER = 0.0511
- Q4: EER = 0.0536

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0452
- Q2: EER = 0.0455
- Q3: EER = 0.0516
- Q4: EER = 0.0538

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0447
- Q2: EER = 0.0453
- Q3: EER = 0.0499
- Q4: EER = 0.0532

### x-vector
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0757
- Q2: EER = 0.0907
- Q3: EER = 0.1108
- Q4: EER = 0.0857

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.0872
- Q2: EER = 0.0883
- Q3: EER = 0.0908
- Q4: EER = 0.0874
