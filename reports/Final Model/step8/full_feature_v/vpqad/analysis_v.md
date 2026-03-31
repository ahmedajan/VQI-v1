# Step 8: VQI-V + Dual-Score Evaluation Analysis - vpqad

## VQI-V ERC Results

### ECAPA-TDNN

**FNMR=10%:**
- Reject 10%: FNMR reduction = 11.5%
- Reject 20%: FNMR reduction = 2.3%
- Reject 30%: FNMR reduction = -8.7%

### ResNetSE34V2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.8%
- Reject 20%: FNMR reduction = 6.2%
- Reject 30%: FNMR reduction = -1.0%

### ECAPA2

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.8%
- Reject 20%: FNMR reduction = 8.9%
- Reject 30%: FNMR reduction = 0.5%

### x-vector

**FNMR=10%:**
- Reject 10%: FNMR reduction = 18.4%
- Reject 20%: FNMR reduction = 12.8%
- Reject 30%: FNMR reduction = 17.4%

### WavLM-SV

**FNMR=10%:**
- Reject 10%: FNMR reduction = 27.6%
- Reject 20%: FNMR reduction = 41.9%
- Reject 30%: FNMR reduction = 34.2%

## Combined ERC (Dual-Score)

### ECAPA-TDNN
- s_only: FNMR reduction@20% = 18.6%
- v_only: FNMR reduction@20% = 2.3%
- union: FNMR reduction@20% = 1.7%
- intersection: FNMR reduction@20% = 16.4%

### ResNetSE34V2
- s_only: FNMR reduction@20% = 16.1%
- v_only: FNMR reduction@20% = 6.2%
- union: FNMR reduction@20% = 7.0%
- intersection: FNMR reduction@20% = 13.8%

### ECAPA2
- s_only: FNMR reduction@20% = 14.8%
- v_only: FNMR reduction@20% = 8.9%
- union: FNMR reduction@20% = 8.3%
- intersection: FNMR reduction@20% = 7.4%

### x-vector
- s_only: FNMR reduction@20% = -0.7%
- v_only: FNMR reduction@20% = 12.8%
- union: FNMR reduction@20% = 10.9%
- intersection: FNMR reduction@20% = -0.3%

### WavLM-SV
- s_only: FNMR reduction@20% = 13.5%
- v_only: FNMR reduction@20% = 41.9%
- union: FNMR reduction@20% = 39.7%
- intersection: FNMR reduction@20% = 17.7%

## Quadrant Analysis

### ECAPA-TDNN
- Q1 EER < Q3 EER: False
- Q1: EER = 0.2103
- Q2: EER = 0.1465
- Q3: EER = 0.1988
- Q4: EER = 0.1633

### ResNetSE34V2
- Q1 EER < Q3 EER: False
- Q1: EER = 0.1230
- Q2: EER = 0.0945
- Q3: EER = 0.1217
- Q4: EER = 0.0761

### ECAPA2
- Q1 EER < Q3 EER: False
- Q1: EER = 0.1373
- Q2: EER = 0.0850
- Q3: EER = 0.1238
- Q4: EER = 0.0761

### x-vector
- Q1 EER < Q3 EER: False
- Q1: EER = 0.3422
- Q2: EER = 0.2742
- Q3: EER = 0.3407
- Q4: EER = 0.4295

### WavLM-SV
- Q1 EER < Q3 EER: True
- Q1: EER = 0.2103
- Q2: EER = 0.1772
- Q3: EER = 0.2576
- Q4: EER = 0.2371
