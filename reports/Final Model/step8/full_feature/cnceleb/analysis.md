# Step 8: VQI-S Evaluation Analysis - cnceleb

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**Ranked DET:**
- middle: EER = 0.2283
- top: EER = 0.0865
- EER separation (bottom/top) = nanx

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**Ranked DET:**
- middle: EER = 0.2137
- top: EER = 0.0715
- EER separation (bottom/top) = nanx

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**Ranked DET:**
- middle: EER = 0.2191
- top: EER = 0.0788
- EER separation (bottom/top) = nanx

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**Ranked DET:**
- middle: EER = 0.2723
- top: EER = 0.1402
- EER separation (bottom/top) = nanx

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = 0.0%
- Reject 30%: FNMR reduction = 0.0%

**Ranked DET:**
- middle: EER = 0.3245
- top: EER = 0.1745
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
