# Step 8: VQI-S Evaluation Analysis - voxceleb1

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = -8.0%
- Reject 20%: FNMR reduction = 0.7%
- Reject 30%: FNMR reduction = 2.6%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 4.5%
- Reject 20%: FNMR reduction = 11.3%
- Reject 30%: FNMR reduction = 20.6%

**Ranked DET:**
- bottom: EER = 0.0573
- middle: EER = 0.0482
- top: EER = 0.0430
- EER separation (bottom/top) = 1.33x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = -6.7%
- Reject 20%: FNMR reduction = 4.2%
- Reject 30%: FNMR reduction = 5.8%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.8%
- Reject 20%: FNMR reduction = 13.1%
- Reject 30%: FNMR reduction = 22.2%

**Ranked DET:**
- bottom: EER = 0.0574
- middle: EER = 0.0480
- top: EER = 0.0420
- EER separation (bottom/top) = 1.37x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = -8.2%
- Reject 20%: FNMR reduction = 0.7%
- Reject 30%: FNMR reduction = 3.3%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.5%
- Reject 20%: FNMR reduction = 8.3%
- Reject 30%: FNMR reduction = 15.3%

**Ranked DET:**
- bottom: EER = 0.0565
- middle: EER = 0.0471
- top: EER = 0.0428
- EER separation (bottom/top) = 1.32x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 17.2%
- Reject 20%: FNMR reduction = 27.5%
- Reject 30%: FNMR reduction = 34.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 11.7%
- Reject 20%: FNMR reduction = 20.9%
- Reject 30%: FNMR reduction = 32.4%

**Ranked DET:**
- bottom: EER = 0.1197
- middle: EER = 0.0937
- top: EER = 0.0690
- EER separation (bottom/top) = 1.74x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = -8.6%
- Reject 20%: FNMR reduction = -0.1%
- Reject 30%: FNMR reduction = 1.3%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.6%
- Reject 20%: FNMR reduction = 12.8%
- Reject 30%: FNMR reduction = 23.2%

**Ranked DET:**
- bottom: EER = 0.0957
- middle: EER = 0.0867
- top: EER = 0.0821
- EER separation (bottom/top) = 1.17x

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 10.9%
- Mean test reduction@20%: 16.8%

## Speed Benchmarks
- 3s: 870.1ms (target: <50ms) [FAIL]
- 10s: 2501.2ms (target: <100ms) [FAIL]
- 60s: 14503.3ms (target: <300ms) [FAIL]
