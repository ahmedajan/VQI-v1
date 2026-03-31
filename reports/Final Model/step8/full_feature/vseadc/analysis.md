# Step 8: VQI-S Evaluation Analysis - vseadc

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = -11.4%
- Reject 20%: FNMR reduction = 25.2%
- Reject 30%: FNMR reduction = 17.1%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.1%
- Reject 20%: FNMR reduction = 17.9%
- Reject 30%: FNMR reduction = 15.7%

**Ranked DET:**
- bottom: EER = 0.3782
- middle: EER = 0.3591
- EER separation (bottom/top) = nanx

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 10.8%
- Reject 20%: FNMR reduction = 25.2%
- Reject 30%: FNMR reduction = 17.1%

**FNMR=10%:**
- Reject 10%: FNMR reduction = -3.3%
- Reject 20%: FNMR reduction = 8.8%
- Reject 30%: FNMR reduction = 5.6%

**Ranked DET:**
- bottom: EER = 0.3614
- middle: EER = 0.3333
- EER separation (bottom/top) = nanx

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 10.8%
- Reject 20%: FNMR reduction = 50.1%
- Reject 30%: FNMR reduction = 44.7%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.1%
- Reject 20%: FNMR reduction = 5.7%
- Reject 30%: FNMR reduction = 12.4%

**Ranked DET:**
- bottom: EER = 0.3614
- middle: EER = 0.3448
- EER separation (bottom/top) = nanx

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = -11.4%
- Reject 20%: FNMR reduction = 0.2%
- Reject 30%: FNMR reduction = -10.5%

**FNMR=10%:**
- Reject 10%: FNMR reduction = -8.7%
- Reject 20%: FNMR reduction = -3.4%
- Reject 30%: FNMR reduction = -4.5%

**Ranked DET:**
- bottom: EER = 0.4454
- middle: EER = 0.4764
- EER separation (bottom/top) = nanx

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 10.8%
- Reject 20%: FNMR reduction = 50.1%
- Reject 30%: FNMR reduction = 44.7%

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.6%
- Reject 20%: FNMR reduction = 2.7%
- Reject 30%: FNMR reduction = 2.3%

**Ranked DET:**
- bottom: EER = 0.3532
- middle: EER = 0.3634
- EER separation (bottom/top) = nanx

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: False
- Mean train reduction@20%: 10.8%
- Mean test reduction@20%: -0.4%

## Speed Benchmarks
- 3s: 870.1ms (target: <50ms) [FAIL]
- 10s: 2501.2ms (target: <100ms) [FAIL]
- 60s: 14503.3ms (target: <300ms) [FAIL]
