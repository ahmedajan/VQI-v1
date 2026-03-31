# VQI-V Evaluation Report — vpqad (v4.0)

## Provider Results

### ECAPA-TDNN
- **fnmr_1pct:** ERC@20% = 48.76165113182424%
- **fnmr_10pct:** ERC@20% = 2.277375869974063%
- **DET separation:** 1.0414185346269653x

### ResNet293
- **fnmr_1pct:** ERC@20% = 100.0%
- **fnmr_10pct:** ERC@20% = 6.239103875245389%
- **DET separation:** 1.651316384691939x

### ECAPA2
- **fnmr_1pct:** ERC@20% = 61.57123834886817%
- **fnmr_10pct:** ERC@20% = 8.880255878759602%
- **DET separation:** 1.633200472141916x

### x-vector
- **fnmr_1pct:** ERC@20% = 74.38082556591212%
- **fnmr_10pct:** ERC@20% = 12.841983884030927%
- **DET separation:** 1.356176919166243x

### WavLM-SV
- **fnmr_1pct:** ERC@20% = 100.0%
- **fnmr_10pct:** ERC@20% = 41.894655922687285%
- **DET separation:** 1.9886959759027991x

## Combined ERC Analysis

### ECAPA-TDNN
- s_only @20%: 18.645296391752588%
- v_only @20%: 2.277375869974063%
- union @20%: 1.7418186275177439%
- intersection @20%: 16.38921202882854%

### ResNet293
- s_only @20%: 16.062607388316152%
- v_only @20%: 6.239103875245389%
- union @20%: 6.9822549673834615%
- intersection @20%: 13.816572398946347%

### ECAPA2
- s_only @20%: 14.771262886597947%
- v_only @20%: 8.880255878759602%
- union @20%: 8.2923640523499%
- intersection @20%: 7.3849733242408515%

## Quadrant Analysis

### ECAPA-TDNN
- Q1<Q3 EER: False
- EER Q1: 0.21033934926283682, Q3: 0.19881422924901188

### ResNet293
- Q1<Q3 EER: False
- EER Q1: 0.1229982206405694, Q3: 0.12168148880105402

### ECAPA2
- Q1<Q3 EER: False
- EER Q1: 0.13725216065073714, Q3: 0.12376482213438736
