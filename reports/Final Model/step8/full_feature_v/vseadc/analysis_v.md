# Step 8: VQI-V + Dual-Score Evaluation Analysis - vseadc

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.9%
- Reject 20%: FNMR reduction = -0.4%
- Reject 30%: FNMR reduction = -7.3%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.8%
- Reject 20%: FNMR reduction = -6.7%
- Reject 30%: FNMR reduction = -10.8%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = -3.6%
- Reject 20%: FNMR reduction = -6.7%
- Reject 30%: FNMR reduction = -10.8%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.8%
- Reject 20%: FNMR reduction = -6.7%
- Reject 30%: FNMR reduction = -18.0%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = -3.6%
- Reject 20%: FNMR reduction = -6.7%
- Reject 30%: FNMR reduction = -14.4%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 17.9%
- v_only: FNMR reduction@20% = -0.4%
- union: FNMR reduction@20% = 3.7%
- intersection: FNMR reduction@20% = 17.9%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 8.8%
- v_only: FNMR reduction@20% = -6.7%
- union: FNMR reduction@20% = -5.7%
- intersection: FNMR reduction@20% = 8.8%

### ECAPA2
- s_only: FNMR reduction@20% = 5.7%
- v_only: FNMR reduction@20% = -6.7%
- union: FNMR reduction@20% = -15.0%
- intersection: FNMR reduction@20% = 5.7%

### x-vector
- s_only: FNMR reduction@20% = -3.4%
- v_only: FNMR reduction@20% = -6.7%
- union: FNMR reduction@20% = -11.9%
- intersection: FNMR reduction@20% = -3.4%

### WavLM-SV
- s_only: FNMR reduction@20% = 2.7%
- v_only: FNMR reduction@20% = -6.7%
- union: FNMR reduction@20% = -5.7%
- intersection: FNMR reduction@20% = 2.7%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: True
- Q1: EER = 0.3279
- Q2: EER = 0.3479
- Q3: EER = 0.4123
- Q4: EER = 0.3843

### ResNetSE34V2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.3149
- Q2: EER = 0.2957
- Q3: EER = 0.3947
- Q4: EER = 0.3808

### ECAPA2
- Q1 EER < Q3 EER: True
- Q1: EER = 0.3216
- Q2: EER = 0.3479
- Q3: EER = 0.4298
- Q4: EER = 0.3559

### x-vector
- Q1 EER < Q3 EER: False
- Q1: EER = 0.4740
- Q2: EER = 0.3913
- Q3: EER = 0.4298
- Q4: EER = 0.4627

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.3279
- Q2: EER = 0.3305
- Q3: EER = 0.4298
- Q4: EER = 0.4093
