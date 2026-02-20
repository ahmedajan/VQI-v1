# Step 8: VQI-S Evaluation Analysis - cnceleb

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 3.4%
- Reject 20%: FNMR reduction = 4.9%
- Reject 30%: FNMR reduction = 6.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 4.6%
- Reject 20%: FNMR reduction = 7.4%
- Reject 30%: FNMR reduction = 10.3%

**Ranked DET:**
- bottom: EER = 0.2410
- middle: EER = 0.2261
- top: EER = 0.0849
- EER separation (bottom/top) = 2.84x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 8.0%
- Reject 20%: FNMR reduction = 11.9%
- Reject 30%: FNMR reduction = 9.4%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 3.7%
- Reject 20%: FNMR reduction = 6.2%
- Reject 30%: FNMR reduction = 7.9%

**Ranked DET:**
- bottom: EER = 0.2520
- middle: EER = 0.2070
- top: EER = 0.0699
- EER separation (bottom/top) = 3.61x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 4.0%
- Reject 20%: FNMR reduction = 7.4%
- Reject 30%: FNMR reduction = 8.1%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 4.6%
- Reject 20%: FNMR reduction = 7.5%
- Reject 30%: FNMR reduction = 10.2%

**Ranked DET:**
- bottom: EER = 0.2247
- middle: EER = 0.2187
- top: EER = 0.0817
- EER separation (bottom/top) = 2.75x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = -3.5%
- Reject 20%: FNMR reduction = -6.7%
- Reject 30%: FNMR reduction = -9.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 0.0%
- Reject 20%: FNMR reduction = -0.8%
- Reject 30%: FNMR reduction = 0.5%

**Ranked DET:**
- bottom: EER = 0.2849
- middle: EER = 0.2693
- top: EER = 0.1429
- EER separation (bottom/top) = 1.99x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 2.7%
- Reject 20%: FNMR reduction = 5.7%
- Reject 30%: FNMR reduction = 7.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 2.6%
- Reject 20%: FNMR reduction = 5.0%
- Reject 30%: FNMR reduction = 7.4%

**Ranked DET:**
- bottom: EER = 0.3599
- middle: EER = 0.3180
- top: EER = 0.1732
- EER separation (bottom/top) = 2.08x

## Cross-System Generalization

- **Verdict:** FAIL
- All monotonic: True
- Mean train reduction@20%: 7.0%
- Mean test reduction@20%: 2.1%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
