# Step 8: VQI-S Evaluation Analysis - vctk

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 41.5%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 33.3%

**Ranked DET:**
- middle: EER = 0.0140
- top: EER = 0.0055
- EER separation (bottom/top) = nanx

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 43.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 35.7%

**Ranked DET:**
- middle: EER = 0.0079
- top: EER = 0.0033
- EER separation (bottom/top) = nanx

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 47.8%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 32.6%

**Ranked DET:**
- middle: EER = 0.0081
- top: EER = 0.0032
- EER separation (bottom/top) = nanx

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 31.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 18.2%

**Ranked DET:**
- middle: EER = 0.0612
- top: EER = 0.0316
- EER separation (bottom/top) = nanx

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 34.6%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 22.6%

**Ranked DET:**
- middle: EER = 0.1390
- top: EER = 0.1085
- EER separation (bottom/top) = nanx

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: True
- Mean train reduction@20%: 0.0%
- Mean test reduction@20%: 0.0%

## Speed Benchmarks
- 3s: 870.1ms (target: <50ms) [FAIL]
- 10s: 2501.2ms (target: <100ms) [FAIL]
- 60s: 14503.3ms (target: <300ms) [FAIL]
