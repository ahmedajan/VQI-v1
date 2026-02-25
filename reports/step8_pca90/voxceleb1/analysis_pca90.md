# Step 8 (PCA-90%): VQI-S Evaluation Analysis - voxceleb1

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 14.6%
- Reject 20%: FNMR reduction = 4.7%
- Reject 30%: FNMR reduction = 8.5%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 7.3%
- Reject 20%: FNMR reduction = 12.9%
- Reject 30%: FNMR reduction = 15.5%

**Ranked DET:**
- bottom: EER = 0.0616
- middle: EER = 0.0474
- top: EER = 0.0426
- EER separation (bottom/top) = 1.44x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 23.2%
- Reject 20%: FNMR reduction = 12.8%
- Reject 30%: FNMR reduction = 20.3%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 9.1%
- Reject 20%: FNMR reduction = 15.9%
- Reject 30%: FNMR reduction = 19.3%

**Ranked DET:**
- bottom: EER = 0.0614
- middle: EER = 0.0471
- top: EER = 0.0418
- EER separation (bottom/top) = 1.47x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 13.0%
- Reject 20%: FNMR reduction = 4.3%
- Reject 30%: FNMR reduction = 9.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 4.2%
- Reject 20%: FNMR reduction = 8.3%
- Reject 30%: FNMR reduction = 12.1%

**Ranked DET:**
- bottom: EER = 0.0603
- middle: EER = 0.0470
- top: EER = 0.0420
- EER separation (bottom/top) = 1.44x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 26.7%
- Reject 20%: FNMR reduction = 38.0%
- Reject 30%: FNMR reduction = 42.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 15.2%
- Reject 20%: FNMR reduction = 25.1%
- Reject 30%: FNMR reduction = 30.2%

**Ranked DET:**
- bottom: EER = 0.1253
- middle: EER = 0.0900
- top: EER = 0.0749
- EER separation (bottom/top) = 1.67x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 10.8%
- Reject 20%: FNMR reduction = 2.8%
- Reject 30%: FNMR reduction = 6.9%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 8.0%
- Reject 20%: FNMR reduction = 12.7%
- Reject 30%: FNMR reduction = 16.0%

**Ranked DET:**
- bottom: EER = 0.0988
- middle: EER = 0.0887
- top: EER = 0.0771
- EER separation (bottom/top) = 1.28x

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 12.4%
- Mean test reduction@20%: 18.9%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
