# VQI-V Evaluation Report — voxceleb1 (v4.0)

## Provider Results

### ECAPA-TDNN
- **fnmr_1pct:** ERC@20% = 1.8927762841375317%
- **fnmr_10pct:** ERC@20% = 7.5655076529973275%
- **DET separation:** nanx

### ResNet293
- **fnmr_1pct:** ERC@20% = 10.079037454691909%
- **fnmr_10pct:** ERC@20% = 9.200937453899705%
- **DET separation:** nanx

### ECAPA2
- **fnmr_1pct:** ERC@20% = 0.3141361256544406%
- **fnmr_10pct:** ERC@20% = 3.705893322867704%
- **DET separation:** nanx

### x-vector
- **fnmr_1pct:** ERC@20% = 20.571068450649587%
- **fnmr_10pct:** ERC@20% = 17.96684118673647%
- **DET separation:** nanx

### WavLM-SV
- **fnmr_1pct:** ERC@20% = -1.649783746870015%
- **fnmr_10pct:** ERC@20% = 8.219679573358274%
- **DET separation:** nanx

## Combined ERC Analysis

### ECAPA-TDNN
- s_only @20%: 11.307767820324543%
- v_only @20%: 7.5655076529973275%
- union @20%: 8.339710202472084%
- intersection @20%: 12.352824434995313%

### ResNet293
- s_only @20%: 13.094850110512034%
- v_only @20%: 9.200937453899705%
- union @20%: 10.13046839677978%
- intersection @20%: 14.337288787410507%

### ECAPA2
- s_only @20%: 8.26310910370881%
- v_only @20%: 3.705893322867704%
- union @20%: 4.293923170887981%
- intersection @20%: 8.185449294923398%

## Quadrant Analysis

### ECAPA-TDNN
- Q1<Q3 EER: True
- EER Q1: 0.039915600370373665, Q3: 0.05913931683381595

### ResNet293
- Q1<Q3 EER: True
- EER Q1: 0.04025421610923321, Q3: 0.05833518323935811

### ECAPA2
- Q1<Q3 EER: True
- EER Q1: 0.04011931906874026, Q3: 0.057623010528842916
