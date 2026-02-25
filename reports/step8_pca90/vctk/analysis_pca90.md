# Step 8 (PCA-90%): VQI-S Evaluation Analysis - vctk

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 30.2%
- Reject 20%: FNMR reduction = 35.9%
- Reject 30%: FNMR reduction = 42.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.2%
- Reject 20%: FNMR reduction = 18.2%
- Reject 30%: FNMR reduction = 23.4%

**Ranked DET:**
- bottom: EER = 0.0229
- middle: EER = 0.0122
- top: EER = 0.0060
- EER separation (bottom/top) = 3.81x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 25.4%
- Reject 20%: FNMR reduction = 31.1%
- Reject 30%: FNMR reduction = 37.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.8%
- Reject 20%: FNMR reduction = 18.9%
- Reject 30%: FNMR reduction = 24.5%

**Ranked DET:**
- bottom: EER = 0.0134
- middle: EER = 0.0067
- top: EER = 0.0032
- EER separation (bottom/top) = 4.26x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 22.5%
- Reject 20%: FNMR reduction = 30.9%
- Reject 30%: FNMR reduction = 38.3%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 11.0%
- Reject 20%: FNMR reduction = 15.4%
- Reject 30%: FNMR reduction = 20.3%

**Ranked DET:**
- bottom: EER = 0.0127
- middle: EER = 0.0075
- top: EER = 0.0020
- EER separation (bottom/top) = 6.41x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 16.3%
- Reject 20%: FNMR reduction = 21.1%
- Reject 30%: FNMR reduction = 24.9%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.9%
- Reject 20%: FNMR reduction = 7.9%
- Reject 30%: FNMR reduction = 9.7%

**Ranked DET:**
- bottom: EER = 0.0832
- middle: EER = 0.0573
- top: EER = 0.0336
- EER separation (bottom/top) = 2.47x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 11.0%
- Reject 20%: FNMR reduction = 15.9%
- Reject 30%: FNMR reduction = 20.5%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 6.1%
- Reject 20%: FNMR reduction = 8.6%
- Reject 30%: FNMR reduction = 11.6%

**Ranked DET:**
- bottom: EER = 0.1535
- middle: EER = 0.1356
- top: EER = 0.1163
- EER separation (bottom/top) = 1.32x

## Cross-System Generalization

- **Verdict:** PASS
- All monotonic: True
- Mean train reduction@20%: 17.5%
- Mean test reduction@20%: 8.3%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
