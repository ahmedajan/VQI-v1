# Step 8 (PCA-90%): VQI-S Evaluation Analysis - cnceleb

## ERC Results

### ECAPA-TDNN

**FNMR=1%:**
- Reject 10%: FNMR reduction = 2.8%
- Reject 20%: FNMR reduction = 3.4%
- Reject 30%: FNMR reduction = 5.5%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 3.5%
- Reject 20%: FNMR reduction = 6.0%
- Reject 30%: FNMR reduction = 12.0%

**Ranked DET:**
- bottom: EER = 0.2352
- middle: EER = 0.2278
- top: EER = 0.0779
- EER separation (bottom/top) = 3.02x

### ResNetSE34V2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 2.6%
- Reject 20%: FNMR reduction = 2.3%
- Reject 30%: FNMR reduction = 5.0%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.8%
- Reject 20%: FNMR reduction = 4.0%
- Reject 30%: FNMR reduction = 7.1%

**Ranked DET:**
- bottom: EER = 0.2398
- middle: EER = 0.2085
- top: EER = 0.0644
- EER separation (bottom/top) = 3.72x

### ECAPA2

**FNMR=1%:**
- Reject 10%: FNMR reduction = 5.5%
- Reject 20%: FNMR reduction = 7.1%
- Reject 30%: FNMR reduction = 10.9%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 3.9%
- Reject 20%: FNMR reduction = 6.0%
- Reject 30%: FNMR reduction = 11.2%

**Ranked DET:**
- bottom: EER = 0.2234
- middle: EER = 0.2192
- top: EER = 0.0699
- EER separation (bottom/top) = 3.20x

### x-vector

**FNMR=1%:**
- Reject 10%: FNMR reduction = -3.9%
- Reject 20%: FNMR reduction = -2.9%
- Reject 30%: FNMR reduction = -1.2%

**FNMR=10%:**
- Reject 10%: FNMR reduction = -0.1%
- Reject 20%: FNMR reduction = 0.7%
- Reject 30%: FNMR reduction = 4.1%

**Ranked DET:**
- bottom: EER = 0.2851
- middle: EER = 0.2705
- top: EER = 0.1378
- EER separation (bottom/top) = 2.07x

### WavLM-SV

**FNMR=1%:**
- Reject 10%: FNMR reduction = 0.8%
- Reject 20%: FNMR reduction = 1.0%
- Reject 30%: FNMR reduction = 6.6%

**FNMR=10%:**
- Reject 10%: FNMR reduction = 1.9%
- Reject 20%: FNMR reduction = 3.9%
- Reject 30%: FNMR reduction = 7.4%

**Ranked DET:**
- bottom: EER = 0.3536
- middle: EER = 0.3188
- top: EER = 0.1713
- EER separation (bottom/top) = 2.06x

## Cross-System Generalization

- **Verdict:** PASS
- All monotonic: True
- Mean train reduction@20%: 5.3%
- Mean test reduction@20%: 2.3%

## Speed Benchmarks
- 3s: 1927.0ms (target: <50ms) [FAIL]
- 10s: 6001.7ms (target: <100ms) [FAIL]
- 60s: 36734.7ms (target: <300ms) [FAIL]
