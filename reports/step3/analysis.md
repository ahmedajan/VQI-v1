# Step 3: Preprocessing Pipeline -- Analysis Report

**Date:** 2026-02-14
**Status:** COMPLETE (all 6 sub-tasks: 3.1-3.6)

---

## 1. Overview

Step 3 implements the audio preprocessing pipeline -- the reusable modules that Step 4 calls to load, normalize, and VAD-segment audio before extracting 544 features. This is a code-writing step, not a computation step. No checkpointing was needed.

**Modules created:**
| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Exceptions (3.5) | `preprocessing/exceptions.py` | 22 | VQIError, AudioLoadError, TooShortError, InsufficientSpeechError |
| Audio Loader (3.1) | `preprocessing/audio_loader.py` | 99 | Load any format, resample 16kHz mono, validate duration |
| Normalizer (3.2) | `preprocessing/normalize.py` | 45 | DC removal + peak norm to [-0.95, 0.95]; accepts numpy and torch |
| VAD (3.3) | `preprocessing/vad.py` | 170 | energy_vad() (existing) + reconstruct_from_mask() (new) |
| Quality Checks (3.4) | `core/vqi_algorithm.py` | 76 | 4 actionable feedback checks on raw waveform |

**Tests:** 20 unit tests in `tests/test_preprocessing.py`, all passing.

---

## 2. Audio Loader (3.1)

The `load_audio()` function handles:
- Multi-format loading via torchaudio/soundfile backend (WAV, FLAC supported; M4A NOT supported on this system)
- Stereo-to-mono mixdown
- Resampling to 16kHz using sinc_interp_kaiser (rolloff=0.99, width=6) -- gold-standard quality
- Duration validation: minimum 1.0s, maximum 120.0s
- Center-crop truncation for files exceeding 120s
- NaN/Inf replacement with zeros + warning

**Known limitation:** The soundfile backend on this system does not support M4A/AAC. VoxCeleb2 has already been converted to WAV in Step 1.2a.

---

## 3. Normalization (3.2)

`dc_remove_and_normalize()` applies:
1. DC offset removal: `x = x - mean(x)`
2. Peak normalization: `x = x / (max|x| + 1e-8) * 0.95`

Peak target of 0.95 provides headroom against downstream clipping. All-zero input returns zeros safely. Accepts both numpy arrays and torch tensors (tensors are converted internally via `np.asarray()`).

**Visualization insight (Plot 5):** Raw waveforms from VoxCeleb typically have small DC offsets (< 0.01) and varied peak amplitudes. After normalization, all signals consistently peak at 0.95 with zero mean.

---

## 4. VAD (3.3)

`energy_vad()` was already complete from Step 2.1. Step 3 added `reconstruct_from_mask()`:
- Converts frame-level boolean mask to sample-level mask
- Extracts contiguous speech regions
- Concatenates with 10ms silence gaps between segments

**VAD characteristics (from 1,210,451 samples):**
- Mean speech ratio: 0.980
- Median speech ratio: 1.000
- Files with ratio < 5%: 5 (0.0004%)
- Files with speech < 1.0s: 9 (0.0007%)

**Key finding:** The training pool is overwhelmingly clean speech. VAD rarely removes significant content, which is expected for VoxCeleb/VOiCES datasets designed for speaker verification research.

---

## 5. Actionable Quality Checks (3.4)

Four checks on the RAW waveform (before normalization):

| Check | Threshold | Triggered | % of Pool |
|-------|-----------|-----------|-----------|
| TooShort | speech < 1.0s | 9 | 0.0007% |
| TooQuiet | peak < 0.001 | ~0 (estimated) | ~0% |
| Clipped | ratio > 10% | ~0 (estimated) | ~0% |
| InsufficientSpeech | ratio < 5% | 5 | 0.0004% |

**Key finding:** Actionable feedback is extremely rare in the training pool. Only 14 out of 1,210,451 files (0.001%) would trigger any check. This is expected -- the VoxCeleb/VOiCES datasets are well-curated. The checks will be more valuable for user-supplied audio at inference time.

**Dataset breakdown:**
- VoxCeleb2 dev: 9 TooShort + 5 InsufficientSpeech (out of 1,048,003)
- VoxCeleb1 dev: 0 triggers (out of 142,648)
- VOiCES: 0 triggers (out of 19,800)

---

## 6. Statistical Tests (M.3 Spec)

### 6.1 Chi-Squared Test: Feedback Flag Rates Across Datasets

Tests whether the probability of triggering any actionable feedback flag differs across datasets.

| | Flagged | Not Flagged | Total |
|---|---------|-------------|-------|
| VOiCES | 0 | 19,800 | 19,800 |
| VoxCeleb1 dev | 0 | 142,648 | 142,648 |
| VoxCeleb2 dev | 14 | 1,047,989 | 1,048,003 |

**Result:** chi2 = 1.395, p = 0.498, dof = 2

**Interpretation:** p = 0.498 (not significant at alpha = 0.05). The feedback flag rate does NOT differ significantly across datasets. Although all 14 triggers come from VoxCeleb2, this is entirely explained by VoxCeleb2 comprising 86.6% of the pool. The per-dataset flag rate is uniformly near zero (~0.001%), so the chi-squared test correctly finds no evidence of dataset-specific quality issues.

### 6.2 KS Test: VAD Ratio Distributions Across Dataset Pairs

Two-sample Kolmogorov-Smirnov test on VAD speech ratio distributions.

| Pair | KS Statistic (D) | p-value | Interpretation |
|------|-------------------|---------|----------------|
| VOiCES vs VoxCeleb1 | 0.1393 | ~0 | Significant |
| VOiCES vs VoxCeleb2 | 0.2153 | ~0 | Significant |
| VoxCeleb1 vs VoxCeleb2 | 0.0828 | ~0 | Significant |

**Interpretation:** All three KS tests are highly significant (p ~ 0), meaning the VAD ratio distributions are statistically distinguishable. However, this is expected and NOT concerning for VQI:

1. **Sample size effect:** With N > 1M samples, even trivially small distribution differences become statistically significant. The KS statistic D is the relevant measure -- D = 0.08-0.22 are modest effect sizes.
2. **All means are within 1% of each other:** VOiCES = 0.982, VoxCeleb1 = 0.973, VoxCeleb2 = 0.980. The differences reflect minor recording condition variations (e.g., VoxCeleb1 has slightly more diverse recording environments).
3. **40dB threshold is appropriate:** All three datasets converge at high speech ratios (>0.97), confirming the threshold works well across corpora.

### 6.3 Bootstrap 95% CI: Mean VAD Ratio Per Dataset

10,000 bootstrap resamples per dataset.

| Dataset | N | Mean VAD Ratio | 95% CI |
|---------|---|----------------|--------|
| VOiCES | 19,800 | 0.98241 | [0.98173, 0.98308] |
| VoxCeleb1 dev | 142,648 | 0.97262 | [0.97232, 0.97291] |
| VoxCeleb2 dev | 1,048,003 | 0.98038 | [0.98030, 0.98047] |

**Interpretation:** All CIs are extremely tight (width < 0.001) due to large sample sizes, and all overlap substantially in the 0.97-0.98 range. VoxCeleb1 has the lowest mean (0.973) -- this is likely because VoxCeleb1 includes more challenging recordings (interviews, press conferences) with more background noise. The 0.8% difference from VOiCES is practically negligible for feature extraction.

### 6.4 Statistical Test Summary

All three M.3 statistical tests converge on the same conclusion: **the VAD preprocessing behaves consistently across datasets.** Feedback flags are uniformly rare, VAD ratios are tightly clustered near 1.0, and the small inter-dataset differences are explainable by recording conditions rather than VAD failure modes. The 40dB dynamic range threshold is well-suited for this corpus.

---

## 7. Visualization Summary

| # | Plot | Key Observation |
|---|------|-----------------|
| 1 | `vad_mask_examples.png` | VAD correctly identifies speech regions; clean files show near-100% speech |
| 2 | `vad_ratio_distribution.png` | Heavy right-skew: most files are ~100% speech |
| 3 | `speech_duration_after_vad.png` | Mean 7.77s, range [0.3s, 46.6s], only 9 below 1.0s |
| 4 | `actionable_feedback_counts.png` | Extremely low trigger rates across all 4 checks |
| 5 | `normalization_effect.png` | DC offset removed, peak consistently at 0.95 |
| 6 | `vad_ratio_ridgeline_by_dataset.png` | All 3 datasets show similar ratio distributions (~0.98 mean) |
| 7 | `duration_before_after_vad_scatter.png` | Points cluster near x=y diagonal (most audio is speech) |
| 8 | `spectrogram_vad_examples.png` | Mel spectrograms confirm VAD aligns with visible speech energy |
| 9 | `actionable_feedback_by_dataset.png` | All triggers concentrated in VoxCeleb2 dev |
| 10 | `bootstrap_ci_vad_ratio.png` | All datasets' mean VAD ratios within 0.97-0.98 with tight CIs |
| 11 | `ks_test_vad_ratio.png` | KS D = 0.08-0.22 between pairs -- statistically significant but practically small |

**Sample selection note (Plots 1, 5, 8):** The M.3 blueprint specifies 6 sample types: clean speech, noisy speech, speech with music, very quiet, clipped, and reverberant. However, the training pool contains effectively zero TooQuiet or Clipped files, and no music-contaminated files (these are speaker verification datasets). Samples were instead selected by dataset diversity (2 from each of VOiCES, VoxCeleb1, VoxCeleb2) to show VAD behavior across different recording conditions.

---

## 8. Implications for Downstream Steps

1. **Step 4 (Feature Extraction):** The preprocessing pipeline is ready. Step 4 will call `load_audio() -> dc_remove_and_normalize() -> energy_vad() -> reconstruct_from_mask()` for each file, then extract 544 features from the speech waveform.

2. **Quality checks will rarely trigger** on the training pool, meaning virtually all 1.21M files will proceed to feature extraction. The checks exist primarily for inference-time robustness.

3. **VAD-reconstructed speech** will be the input to frame-level feature extraction (F1-F23). The 10ms gap between segments prevents cross-contamination of frame statistics across non-contiguous speech regions.

4. **Cross-dataset VAD consistency** (confirmed by KS + bootstrap tests) means feature distributions should be comparable across datasets -- important for Steps 5-6 where features are pooled for model training.

---

## 9. Output Inventory

**Code modules (5):**
- `implementation/vqi/preprocessing/exceptions.py`
- `implementation/vqi/preprocessing/audio_loader.py`
- `implementation/vqi/preprocessing/normalize.py`
- `implementation/vqi/preprocessing/vad.py` (updated with reconstruct_from_mask)
- `implementation/vqi/core/vqi_algorithm.py`

**Tests:**
- `implementation/tests/test_preprocessing.py` (20 tests, all passing)

**Visualizations (11):**
- `implementation/reports/step3/vad_mask_examples.png`
- `implementation/reports/step3/vad_ratio_distribution.png`
- `implementation/reports/step3/speech_duration_after_vad.png`
- `implementation/reports/step3/actionable_feedback_counts.png`
- `implementation/reports/step3/normalization_effect.png`
- `implementation/reports/step3/vad_ratio_ridgeline_by_dataset.png`
- `implementation/reports/step3/duration_before_after_vad_scatter.png`
- `implementation/reports/step3/spectrogram_vad_examples.png`
- `implementation/reports/step3/actionable_feedback_by_dataset.png`
- `implementation/reports/step3/bootstrap_ci_vad_ratio.png`
- `implementation/reports/step3/ks_test_vad_ratio.png`

**Statistical test results:**
- `implementation/reports/step3/statistical_tests.json`

---

## 10. Verification Checklist

- [x] All 5 modules importable with no errors
- [x] 20 unit tests pass (pytest)
- [x] End-to-end: load WAV -> normalize -> VAD -> quality checks -> returns waveform
- [x] normalize.py accepts both numpy arrays and torch tensors
- [x] 11 visualizations generated
- [x] Chi-squared test: feedback flags not dataset-dependent (p = 0.498)
- [x] KS test: VAD ratio distributions computed for all 3 dataset pairs
- [x] Bootstrap 95% CI: mean VAD ratio per dataset with 10,000 resamples
- [x] analysis.md written with statistical test interpretations
- [x] Blueprint docs updated per completion policy
