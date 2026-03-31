# Step 8: VQI-S Evaluation Analysis - vpqad

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 44.5%
- Reject 20%: FNMR reduction = 49.9%
- Reject 30%: FNMR reduction = 44.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 15.4%
- Reject 20%: FNMR reduction = 18.6%
- Reject 30%: FNMR reduction = 20.6%

**Ranked DET:**
- bottom: EER = 0.2398
- middle: EER = 0.1807
- EER separation (bottom/top) = nanx

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 100.0%
- Reject 20%: FNMR reduction = 100.0%
- Reject 30%: FNMR reduction = 100.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 16.5%
- Reject 20%: FNMR reduction = 16.1%
- Reject 30%: FNMR reduction = 11.9%

**Ranked DET:**
- bottom: EER = 0.1819
- middle: EER = 0.0940
- EER separation (bottom/top) = nanx

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 66.7%
- Reject 20%: FNMR reduction = 62.4%
- Reject 30%: FNMR reduction = 58.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 17.6%
- Reject 20%: FNMR reduction = 14.8%
- Reject 30%: FNMR reduction = 6.2%

**Ranked DET:**
- bottom: EER = 0.1926
- middle: EER = 0.0928
- EER separation (bottom/top) = nanx

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 66.7%
- Reject 20%: FNMR reduction = 62.4%
- Reject 30%: FNMR reduction = 72.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 5.1%
- Reject 20%: FNMR reduction = -0.7%
- Reject 30%: FNMR reduction = 0.4%

**Ranked DET:**
- bottom: EER = 0.3346
- middle: EER = 0.3457
- EER separation (bottom/top) = nanx

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 100.0%
- Reject 20%: FNMR reduction = 100.0%
- Reject 30%: FNMR reduction = 100.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 15.4%
- Reject 20%: FNMR reduction = 13.5%
- Reject 30%: FNMR reduction = 14.8%

**Ranked DET:**
- bottom: EER = 0.2329
- middle: EER = 0.2329
- EER separation (bottom/top) = nanx

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 16.5%
- Mean test reduction@20%: 6.4%

## Speed Benchmarks
- 3s: 870.1ms (target: <50ms) [FAIL]
- 10s: 2501.2ms (target: <100ms) [FAIL]
- 60s: 14503.3ms (target: <300ms) [FAIL]
