# VQI-V Evaluation Report — vseadc (v4.0)

## Provider Results

### ECAPA-TDNN
- **fnmr_1pct:** ERC@20% = 48.55345911949686%
- **fnmr_10pct:** ERC@20% = -0.38349440098175425%
- **DET separation:** 1.0949684964221091x

### ResNet293
- **fnmr_1pct:** ERC@20% = 22.83018867924529%
- **fnmr_10pct:** ERC@20% = -6.6574628010431125%
- **DET separation:** 1.2765858672228916x

### ECAPA2
- **fnmr_1pct:** ERC@20% = -28.616352201257843%
- **fnmr_10pct:** ERC@20% = -6.6574628010431125%
- **DET separation:** 1.0949684964221091x

### x-vector
- **fnmr_1pct:** ERC@20% = -2.8930817610062887%
- **fnmr_10pct:** ERC@20% = -6.6574628010431125%
- **DET separation:** 1.116444312188993x

### WavLM-SV
- **fnmr_1pct:** ERC@20% = -2.8930817610062887%
- **fnmr_10pct:** ERC@20% = -6.6574628010431125%
- **DET separation:** 1.4023751270030025x

## Combined ERC Analysis

### ECAPA-TDNN
- s_only @20%: 17.8837001784652%
- v_only @20%: -0.38349440098175425%
- union @20%: 3.6623356887774516%
- intersection @20%: 17.8837001784652%

### ResNet293
- s_only @20%: 8.759666864961336%
- v_only @20%: -6.6574628010431125%
- union @20%: -5.6606640832763455%
- intersection @20%: 8.759666864961336%

### ECAPA2
- s_only @20%: 5.71832242712671%
- v_only @20%: -6.6574628010431125%
- union @20%: -14.983663855330143%
- intersection @20%: 5.71832242712671%

## Quadrant Analysis

### ECAPA-TDNN
- Q1<Q3 EER: True
- EER Q1: 0.3278902953586498, Q3: 0.4122536945812808

### ResNet293
- Q1<Q3 EER: True
- EER Q1: 0.3148945147679325, Q3: 0.3947044334975369

### ECAPA2
- Q1<Q3 EER: True
- EER Q1: 0.32156118143459916, Q3: 0.4298029556650246
