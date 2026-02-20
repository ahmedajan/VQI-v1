# Step 8: VQI-S Evaluation Analysis - voxceleb1

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 13.6%
- Reject 20%: FNMR reduction = 15.2%
- Reject 30%: FNMR reduction = 11.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 4.1%
- Reject 20%: FNMR reduction = 8.5%
- Reject 30%: FNMR reduction = 13.2%

**Ranked DET:**
- bottom: EER = 0.0440
- middle: EER = 0.0536
- top: EER = 0.0263
- EER separation (bottom/top) = 1.67x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 23.7%
- Reject 20%: FNMR reduction = 26.6%
- Reject 30%: FNMR reduction = 20.6%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.7%
- Reject 20%: FNMR reduction = 11.0%
- Reject 30%: FNMR reduction = 15.5%

**Ranked DET:**
- bottom: EER = 0.0427
- middle: EER = 0.0547
- top: EER = 0.0267
- EER separation (bottom/top) = 1.60x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 12.1%
- Reject 20%: FNMR reduction = 12.8%
- Reject 30%: FNMR reduction = 11.6%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.6%
- Reject 20%: FNMR reduction = 3.8%
- Reject 30%: FNMR reduction = 5.8%

**Ranked DET:**
- bottom: EER = 0.0415
- middle: EER = 0.0540
- top: EER = 0.0258
- EER separation (bottom/top) = 1.61x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 29.4%
- Reject 20%: FNMR reduction = 42.1%
- Reject 30%: FNMR reduction = 47.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 12.5%
- Reject 20%: FNMR reduction = 20.5%
- Reject 30%: FNMR reduction = 27.5%

**Ranked DET:**
- bottom: EER = 0.1208
- middle: EER = 0.0953
- top: EER = 0.0593
- EER separation (bottom/top) = 2.04x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 7.8%
- Reject 20%: FNMR reduction = 7.7%
- Reject 30%: FNMR reduction = 10.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.9%
- Reject 20%: FNMR reduction = 6.6%
- Reject 30%: FNMR reduction = 10.6%

**Ranked DET:**
- bottom: EER = 0.0915
- middle: EER = 0.0893
- top: EER = 0.0828
- EER separation (bottom/top) = 1.10x

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 7.7%
- Mean test reduction@20%: 13.6%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
