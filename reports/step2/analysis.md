# Step 2: Binary Label Definition & Continuous d-prime — Analysis Report

**Date:** 2026-02-14
**Sub-steps:** 2.1 (durations), 2.2 (thresholds), 2.3 (labels), 2.4 (balancing), 2.5 (d-prime), 2.6 (validation)
**Total runtime:** ~5h 10m (2.1 VAD) + 19s (2.2-2.6) = ~5h 10m
**Outputs:** 5 data files + 21 plots (6 from 2.1, 15 from 2.2-2.6)

---

## 1. Overview

Step 2 defines two independent label systems:
- **Binary labels** (Class 0/1): Training targets for the RF classifier in Step 6. Driven by provider scores + duration thresholds.
- **Continuous d-prime** (per-speaker): Fisher Discriminant Ratio for feature evaluation in Step 5. Measures genuine-vs-impostor separation quality per speaker.

---

## 2. Threshold Computation (2.2)

S-normalized scores used throughout (genuine_norm, impostor_norm).

| Provider | 90th Percentile | FMR=0.001 | Gap |
|----------|---------------:|----------:|----:|
| P1 (ECAPA)  | 11.1714 | 4.5827 | 6.5887 |
| P2 (ResNet)  | 10.6319 | 4.6758 | 5.9560 |
| P3 (ECAPA2)  | 9.5537 | 4.2563 | 5.2974 |

The large gaps (5.3-6.6) between Class 1 gate (90th percentile) and Class 0 gate (FMR=0.001) mean there is a wide "excluded" zone. This is by design -- we only label the extremes for clean training signal.

P3 (ECAPA2) has slightly lower thresholds because its S-norm distribution is tighter (std=1.11 vs 1.30-1.38).

---

## 3. Binary Label Assignment (2.3)

| Category | Count | Percentage |
|----------|------:|----------:|
| Class 1 (high-performing) | 39,096 | 3.23% |
| Class 0 (low-performing) | 10,144 | 0.84% |
| Excluded | 1,161,211 | 95.93% |
| **Total** | **1,210,451** | **100%** |

**Class 0 breakdown:**
- Duration < 1.5s only: 24 (0.24% of Class 0)
- All 3 providers < FMR: 10,120 (99.76% of Class 0)
- Both conditions: 0

**Key observations:**
1. **Duration is not discriminative** -- 99.9% of samples pass the >= 3.0s threshold. Binary labels are driven almost entirely by provider scores.
2. **Class 0 is 3.9x smaller than Class 1** (10,144 vs 39,096). Failing ALL 3 providers simultaneously is quite rare, confirming that most samples perform well on at least one provider.
3. **95.9% excluded** -- the vast majority of samples fall in the ambiguous zone between the two extremes. This is expected and desirable for clean training labels.

**Dataset composition within labels:**

| Dataset | Class 0 | Class 1 | Total |
|---------|--------:|--------:|------:|
| voxceleb1_dev | 324 | 1,107 | 1,431 |
| voxceleb2_dev | 9,785 | 35,817 | 45,602 |
| voices | 35 | 2,172 | 2,207 |

VoxCeleb2 dominates both classes (92.6% of all labeled samples), which is expected given it contains ~80% of the training pool.

---

## 4. Balanced Training Set (2.4)

| Metric | Value |
|--------|------:|
| Minority class | Class 0 (10,144) |
| Total balanced | 20,288 (2 x 10,144) |
| Downsampled | Class 1: 39,096 -> 10,144 (74.1% removed) |

The balanced set is relatively small (20,288 / 1,210,451 = 1.7% of the full pool). This is adequate for RF training with ~500 features but worth monitoring -- if feature extraction reveals that more training data is needed, we could revisit the threshold strategy (e.g., relaxing FMR to 0.01).

**Random seed:** 42 (reproducible)

---

## 5. Continuous d-prime for Feature Evaluation (2.5)

Per-speaker d-prime computed for 7,505 unique speakers across all 1,210,451 samples.

| Statistic | P1 (ECAPA) | P2 (ResNet) | P3 (ECAPA2) | Mean |
|-----------|----------:|----------:|----------:|------:|
| Mean | 5.011 | 4.614 | 5.018 | 4.881 |
| Std | 0.952 | 0.915 | 1.065 | 0.956 |
| Min | 1.403 | 1.460 | 1.354 | 1.406 |
| P5 | 3.274 | 2.919 | 2.933 | 3.052 |
| Median | 5.095 | 4.703 | 5.181 | 5.002 |
| P95 | 6.436 | 5.977 | 6.479 | 6.248 |
| Max | 7.946 | 7.646 | 7.566 | 7.549 |

**Key observations:**
1. **All d-prime values positive** (min 1.35-1.46), meaning every speaker has some genuine-impostor separation. No catastrophically bad speakers.
2. **Range is tighter than expected** (plan predicted 0-15, actual is ~1.3-8.0). This is because the denominator uses sum-of-sigmas (sigma_genuine + sigma_impostor), which is larger than a pooled RMS denominator.
3. **P1 and P3 are very similar** (mean ~5.0), while P2 is slightly lower (4.6). This aligns with Step 1 findings where P2 (ResNet) showed slightly lower d-prime.
4. **d-prime is per-speaker, not per-sample.** In Step 5, features will be correlated with these speaker-level values to identify which audio characteristics predict speaker verification quality.

**P4/P5 gap:** fisher_P4 and fisher_P5 columns are NaN because Step 1.6 only extracted P1-P3 embeddings for the training pool. P4/P5 extraction is deferred to before Step 7.

---

## 6. Validation Set (2.6)

| Metric | Value |
|--------|------:|
| Total samples | 50,000 |
| Overlap with training set | **0** (verified) |
| Samples with P1-P3 scores | 0 / 50,000 |

**Composition:**
| Dataset | Count |
|---------|------:|
| voxceleb1_dev | 5,994 |
| voxceleb2_dev | 44,006 |

The validation set has no P1-P3 scores because it was split from the full pool BEFORE Step 1.6 embedding extraction (which only processed the training pool). Validation scores will be computed in Step 7 when all 5 providers (P1-P5) are run on the validation set.

---

## 7. Implications for Downstream Steps

### Step 3 (Feature Extraction)
- No dependency on Step 2 (independent). Can proceed.

### Step 5 (Feature Evaluation)
- Uses `fisher_values.csv` (1,210,451 rows with per-speaker d-prime)
- Spearman correlation between extracted features and d-prime identifies quality-predictive features
- d-prime range [1.3, 8.0] provides good spread for correlation analysis

### Step 6 (Model Training)
- Uses `training_set_final.csv` (20,288 rows, 1:1 balanced)
- 20K samples with ~500 features is adequate for RF but on the smaller side
- If model performance is insufficient, consider relaxing Class 0 threshold

### Step 7 (Validation)
- Uses `validation_set.csv` (50,000 rows, no scores yet)
- P1-P5 scores must be computed before validation

---

## 8. Output Inventory

| File | Path | Rows | Size |
|------|------|-----:|-----:|
| label_thresholds.yaml | `implementation/data/` | 9 lines | <1 KB |
| training_labels.csv | `implementation/data/` | 49,240 | ~6 MB |
| training_set_final.csv | `implementation/data/` | 20,288 | ~2.5 MB |
| fisher_values.csv | `implementation/data/` | 1,210,451 | ~180 MB |
| validation_set.csv | `implementation/data/` | 50,000 | ~6 MB |

**Plots (21 total):**
- 2.1_durations/: 6 plots
- 2.2_thresholds/: 3 plots (genuine distributions, impostor distributions, summary table)
- 2.3_labels/: 4 plots (label counts, score distributions, duration by label, dataset composition)
- 2.4_balanced/: 2 plots (before/after, pie chart)
- 2.5_fisher/: 4 plots (distributions, correlation matrix, d-prime vs score, mean distribution)
- 2.6_validation/: 2 plots (overlap check, composition)

---

## 9. Verification Checklist

- [x] `label_thresholds.yaml`: 6 values, all percentile_90 > fmr_001
- [x] `training_labels.csv`: no NaN, labels exactly {0, 1}, all row_idx unique
- [x] `training_set_final.csv`: exact 1:1 class ratio, subset of training_labels.csv
- [x] `fisher_values.csv`: no NaN in P1-P3 columns, d-prime range [1.3, 8.0], P4/P5 all NaN
- [x] `validation_set.csv`: exactly 50,000 rows, zero overlap with training_set_final.csv
- [x] All 21 visualizations generated
- [x] Blueprint docs updated per completion policy
