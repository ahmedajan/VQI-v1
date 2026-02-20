# Step 8: VQI-S Evaluation Analysis - vctk

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 19.2%
- Reject 20%: FNMR reduction = 27.1%
- Reject 30%: FNMR reduction = 36.3%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.4%
- Reject 20%: FNMR reduction = 19.4%
- Reject 30%: FNMR reduction = 25.8%

**Ranked DET:**
- bottom: EER = 0.0176
- middle: EER = 0.0123
- top: EER = 0.0067
- EER separation (bottom/top) = 2.62x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 23.7%
- Reject 20%: FNMR reduction = 30.5%
- Reject 30%: FNMR reduction = 41.9%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.8%
- Reject 20%: FNMR reduction = 19.9%
- Reject 30%: FNMR reduction = 27.2%

**Ranked DET:**
- bottom: EER = 0.0109
- middle: EER = 0.0068
- top: EER = 0.0033
- EER separation (bottom/top) = 3.27x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 23.5%
- Reject 20%: FNMR reduction = 33.2%
- Reject 30%: FNMR reduction = 41.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 13.0%
- Reject 20%: FNMR reduction = 18.7%
- Reject 30%: FNMR reduction = 26.3%

**Ranked DET:**
- bottom: EER = 0.0094
- middle: EER = 0.0071
- top: EER = 0.0043
- EER separation (bottom/top) = 2.21x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 38.1%
- Reject 20%: FNMR reduction = 47.6%
- Reject 30%: FNMR reduction = 57.8%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 17.8%
- Reject 20%: FNMR reduction = 23.8%
- Reject 30%: FNMR reduction = 30.3%

**Ranked DET:**
- bottom: EER = 0.0753
- middle: EER = 0.0535
- top: EER = 0.0400
- EER separation (bottom/top) = 1.88x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 15.0%
- Reject 20%: FNMR reduction = 20.7%
- Reject 30%: FNMR reduction = 31.8%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 12.0%
- Reject 20%: FNMR reduction = 17.8%
- Reject 30%: FNMR reduction = 26.7%

**Ranked DET:**
- bottom: EER = 0.1551
- middle: EER = 0.1339
- top: EER = 0.1101
- EER separation (bottom/top) = 1.41x

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 19.3%
- Mean test reduction@20%: 20.8%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
