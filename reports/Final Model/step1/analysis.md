# Step 1: Data Collection -- Completion Analysis

## Overview

Step 1 assembled and organized the speech corpus for training, validation, and testing, and set up the speaker recognition provider systems for comparison score generation. All 9 sub-steps completed successfully with zero errors across all operations.

**Completion date:** 2026-02-14
**Total elapsed time:** ~8 days (2026-02-07 to 2026-02-14), dominated by embedding extraction (1.6a, ~7 days)

---

## Sub-Step Results Summary

| Sub-step | Description | Duration | Key Output | Errors |
|----------|-------------|----------|------------|--------|
| 1.1 | Dataset Verification | <1 min | 8/8 datasets present | 0 |
| 1.2 | Inventory Script | ~5 min | dataset_inventory.csv | 0 |
| 1.2a | VoxCeleb2 M4A->WAV | 4h 5m | 1,128,246 WAV files | 0 |
| 1.3 | VoxCeleb1-test Split | <1 min | 40 speakers, 4,874 utts | 0 |
| 1.4 | Train/Val/Test Splits | ~2 min | 7 CSV manifests | 0 |
| 1.5 | Provider Setup | ~1h | 5/5 providers verified | 0 |
| 1.7 | S-Norm Cohort | 4h 2m | 1,000 speakers x 5 providers | 0 |
| 1.6a | Embedding Extraction | ~7 days | 1,210,451 x 3 memmaps | 0 |
| 1.6b | Score Computation | 2.5 min | 3 score CSVs + 6 impostor .npy | 0 |

---

## Dataset Verification (1.1-1.2)

### Corpus Composition

| Dataset | Speakers | Utterances | Hours | Format | Role |
|---------|----------|------------|-------|--------|------|
| VoxCeleb1 | 1,251 | 153,516 | ~328h | WAV 16kHz | Train/Test |
| VoxCeleb2 | 6,112 | 1,128,246 | ~2,572h | WAV 16kHz (converted) | Train/Val |
| VCTK | 110 | 88,328 | ~85h | FLAC 48kHz | Test |
| VOiCES | 749 | 20,248 | ~366h | WAV 16kHz | Train |
| LibriSpeech | 73 | 5,559 | ~11h | FLAC 16kHz | Test |
| CN-Celeb1 | 997 | 126,532 | ~232h | FLAC 16kHz | Test |
| MUSAN | -- | 2,016 | ~93h | WAV 16kHz | Augmentation |
| RIR | -- | 61,260 | ~21h | WAV 16kHz | Augmentation |

### Key Observations

- **VoxCeleb2 dominates the training pool** (90.2% of train_pool.csv), which is expected given it is the largest speaker verification dataset available. This means VQI will primarily learn quality patterns from YouTube-sourced interview/talk show audio.
- **VoxCeleb2 M4A conversion** was error-free: all 1,128,246 files converted to 16kHz mono WAV.
- **VCTK is 48kHz** -- requires resampling to 16kHz in Step 3 preprocessing.
- **VOiCES includes both clean (source-16k) and reverberant (distant-16k) recordings**, which is intentional: the RF must learn that reverberation degrades recognition.

### Implications for VQI

The heavy VoxCeleb2 representation means VQI will be most calibrated for conversational speech in moderately noisy environments (YouTube source). Performance on clean studio recordings (VCTK) and read speech (LibriSpeech) will be validated as out-of-distribution generalization tests in Step 8.

---

## Split Construction (1.3-1.4)

### Split Sizes

| Split | Files | Source | Purpose |
|-------|-------|--------|---------|
| train_pool.csv | 1,210,451 | VC1-dev + VC2-dev + VOiCES | Model training (after label selection) |
| val_set.csv | 50,000 | VC1-dev + VC2-dev (random, seed=42) | Hyperparameter tuning, validation |
| test_voxceleb1.csv | 4,874 | VC1-test (40 speakers) | Conformance testing |
| test_vctk.csv | 44,455 | VCTK mic1 | Cross-domain evaluation |
| test_librispeech_clean.csv | 2,620 | LS test-clean | Clean speech evaluation |
| test_librispeech_other.csv | 2,939 | LS test-other | Noisy speech evaluation |
| test_cnceleb.csv | 126,532 | CN-Celeb1 | Cross-language evaluation |
| **Total** | **1,441,871** | | |

### Integrity Checks

- **Zero overlap** between train and val filenames
- **Zero test speaker leaks** into train/val
- **Zero file overlap** between any splits
- **Validation set** (50,000) is ~4% of training pool, providing tight confidence intervals (CI width ~0.01 at 95% for binary accuracy)

---

## Provider Verification (1.5)

### Verification Results

| Provider | Architecture | Dim | EER (Vox1-O) | Genuine | Impostor | Gap | Status |
|----------|-------------|-----|-------------|---------|----------|-----|--------|
| P1 ECAPA-TDNN | TDNN+SE+Attention | 192 | 0.87% | +0.728 | -0.124 | 0.851 | PASS |
| P2 ResNet34 | CNN+SE+ASP | 256 | 1.05% | +0.758 | -0.081 | 0.839 | PASS |
| P3 ECAPA2 | Hybrid 1D+2D Conv | 192 | 0.17% | +0.778 | -0.026 | 0.804 | PASS |
| P4 x-vector | Classical TDNN | 512 | 3.13% | +0.976 | +0.862 | 0.114 | PASS |
| P5 WavLM-SV | SSL Transformer | 512 | ~0.6% | +0.955 | +0.416 | 0.539 | PASS |

### Key Findings

- **P1-P3 (training providers)** show excellent separation with near-zero impostor similarity -- ideal for label generation.
- **P4 (x-vector)** has compressed score range (genuine ~0.98, impostor ~0.86) -- reserved for testing only.
- **P5 (WavLM-SV)** has moderate separation -- reserved for testing to evaluate SSL-based system.
- **Provider diversity** is confirmed: 3 different architecture families (TDNN, CNN, Hybrid) for training ensures labels capture consensus quality rather than single-model artifacts.

### Platform Workarounds

- SpeechBrain `local_strategy=LocalStrategy.COPY` (Windows symlink limitation)
- ECAPA2 TorchScript: `_jit_override_can_fuse_on_gpu(False)` + CPU load then `.to(device)` (NVFuser STFT kernel issue)

---

## S-Norm Cohort (1.7)

### Cohort Quality

- **1,000 speakers** selected from VoxCeleb2-dev (seed=42)
- All embeddings properly L2-normalized (norm = 1.0 for all 5 providers)
- **Zero NaN/Inf** in any cohort embedding

### Inter-Speaker Similarity (Cohort)

| Provider | Mean Pairwise Cosine | Assessment |
|----------|---------------------|------------|
| P1 ECAPA | 0.0168 | Excellent (near zero) |
| P2 ResNet | 0.0136 | Excellent (near zero) |
| P3 ECAPA2 | 0.0140 | Excellent (near zero) |
| P4 x-vector | 0.9329 | High (compressed range) |
| P5 WavLM | 0.6443 | Moderate |

P1-P3 show well-separated embedding spaces (mean pairwise cosine near zero), confirming the cohort will produce effective s-norm normalization. P4's high inter-speaker similarity explains its narrow genuine-impostor gap.

---

## Embedding Extraction (1.6a)

### Extraction Results

| Provider | Memmap Shape | Size | Norms | NaN/Inf | Zero Rows |
|----------|-------------|------|-------|---------|-----------|
| P1_ECAPA | (1,210,451, 192) | 887 MB | All 1.0 | 0 | 0 |
| P2_RESNET | (1,210,451, 256) | 1,182 MB | All 1.0 | 0 | 0 |
| P3_ECAPA2 | (1,210,451, 192) | 887 MB | All 1.0 | 0 | 0 |

- **Total runtime:** ~7 days at ~2.3-2.4 files/sec
- **GPU utilization:** 96-100% throughout, 12 GB VRAM
- **Bottleneck:** Python/IO overhead, not GPU compute (batch/concurrency optimization was attempted but yielded no significant speedup due to small model sizes)

### Embedding Quality Assessment

- All embeddings are properly L2-normalized (norm = 1.000000)
- t-SNE visualizations show clear speaker clustering across all 3 providers
- No visible dataset bias in embedding space (VoxCeleb1 and VoxCeleb2 embeddings intermix)
- Inter-speaker cosine similarity approximately centered at zero for P1-P3

---

## Score Computation (1.6b)

### Score Statistics

| Provider | Genuine Mean | Genuine Std | Impostor Mean | Impostor Std | d' (Raw) | d' (S-Norm) | Cohen's d |
|----------|-------------|-------------|---------------|-------------|----------|-------------|-----------|
| P1_ECAPA | 0.8060 | 0.0985 | 0.0169 | 0.0816 | 4.38 | 4.27 | 8.72 |
| P2_RESNET | 0.7844 | 0.1075 | 0.0141 | 0.0867 | 3.97 | 3.85 | 7.89 |
| P3_ECAPA2 | 0.8582 | 0.0963 | 0.0140 | 0.1019 | 4.26 | 4.02 | 8.51 |

### Key Findings

- **All d-prime values exceed 3.9** (raw), indicating excellent genuine-impostor separation. This is well above the d'=3.0 threshold for "good" separation.
- **Cohen's d > 7.5** for all providers -- massive effect sizes confirming that genuine and impostor scores are drawn from very different distributions.
- **S-norm effectively centers impostor distributions near zero** (all impostor means < 0.01 after normalization).
- **S-norm slightly reduces d-prime** (raw > s-norm for all providers) because it increases genuine score variance. However, the normalized scores are more comparable across speakers.

### Cross-Provider Score Correlation

| Pair | Pearson r | Spearman rho |
|------|-----------|-------------|
| P1 vs P2 | 0.938 | 0.911 |
| P1 vs P3 | 0.899 | 0.882 |
| P2 vs P3 | 0.900 | 0.871 |

Cross-provider correlations are high (0.87-0.94), indicating strong provider agreement on which utterances have high/low genuine scores. This is desirable for label generation: samples that all providers agree are "high quality" will form Class 1, providing robust training targets.

### Bland-Altman Analysis

Provider pairs show minimal systematic bias (mean differences < 0.03) with limits of agreement approximately +/-0.15. This means providers can disagree by up to ~0.15 cosine similarity for individual utterances, but on average they agree closely.

---

## Implications for Downstream Steps

### Step 2 (Utility/Label Definition)

- **Label quality will be HIGH.** With d' > 3.9, the 90th percentile (Class 1) and FMR=0.001 (Class 0) thresholds will select clearly separated samples.
- **Expected labeled sample count:** ~8,000 per class (rough estimate based on the 90th percentile of 1.2M genuine scores and the FMR threshold).
- **All three providers will be used for consensus labeling** -- a sample must pass all three providers' thresholds to be labeled, reducing noise from individual provider quirks.

### Step 3 (Preprocessing)

- **VCTK resampling** from 48kHz to 16kHz will be needed for test set feature extraction.
- All training audio is already 16kHz mono WAV -- no further conversion needed.

### Step 4 (Feature Computation)

- **1,210,451 training pool files** need feature extraction (544 VQI-S + 161 VQI-V candidates).
- At estimated ~541ms per utterance, feature extraction will take approximately **7.6 days** on a single thread. Parallelization should be considered.

### Overall Assessment

**Step 1 is a success.** All data is assembled, all providers verified, all embeddings extracted without error, and score distributions show the strong separation needed for reliable label generation in Step 2. The foundation is solid for proceeding to the labeling phase.

---

## Visualization Inventory

### Sub-step level (in step1/ subfolders)

| Subfolder | Plots | Analysis |
|-----------|-------|----------|
| 1.1_dataset_verification/ | 6 | analysis.md |
| 1.2_inventory/ | 2 | analysis.md |
| 1.4_splits/ | 4 | analysis.md |
| 1.5_providers/ | 7 | analysis.md |
| 1.6_embeddings/ | 11 | analysis.md |
| 1.6_scores/ | 12 | analysis.md |
| 1.7_snorm_cohort/ | 6 | analysis.md |

### Step level (in reports/ root)

| File | Description |
|------|-------------|
| 1_step1_summary_dashboard.png | Comprehensive 12-panel dashboard |
| 1_step1_summary.csv | Machine-readable summary |
| 1_6a_*.png (6 files) | 1.6a embedding extraction plots |
| 1_6b_*.png (6 files) | 1.6b score computation plots |
| 1_6a_summary_statistics.csv | 1.6a summary stats |
| 1_6b_summary_statistics.csv | 1.6b summary stats |

**Total visualization outputs:** 48 plots + 7 sub-step analysis files + 1 step-level analysis + 4 CSVs
