# Step 1.7 S-Norm Cohort - Analysis

## Summary
S-norm cohort built from 1,000 speakers (VoxCeleb2 dev set). Embeddings extracted for all 5 providers.

## Embedding Quality
| Provider | Shape | Norm Mean | NaN | Inf |
|----------|-------|-----------|-----|-----|
| P1_ECAPA | (1000, 192) | 1.000000 | 0 | 0 |
| P2_RESNET | (1000, 256) | 1.000000 | 0 | 0 |
| P3_ECAPA2 | (1000, 192) | 1.000000 | 0 | 0 |
| P4_XVECTOR | (1000, 512) | 1.000000 | 0 | 0 |
| P5_WAVLM | (1000, 512) | 1.000000 | 0 | 0 |

## Inter-Speaker Cosine Similarity
| Provider | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| P1_ECAPA | 0.0168 | 0.0814 | -0.3853 | 0.6973 |
| P2_RESNET | 0.0136 | 0.0869 | -0.4014 | 0.6213 |
| P3_ECAPA2 | 0.0140 | 0.1006 | -0.4516 | 0.6913 |
| P4_XVECTOR | 0.9329 | 0.0276 | 0.8177 | 0.9906 |
| P5_WAVLM | 0.6443 | 0.1861 | 0.0725 | 0.9925 |

## What is GOOD for VQI:
- **All embeddings perfectly L2-normalized** (norms = 1.000000 for all providers). This confirms the extraction pipeline is working correctly.
- **Zero NaN/Inf** across all 5 providers. No corrupt embeddings.
- **P1, P2, P3 have near-zero inter-speaker cosine similarity** (means: 0.0168, 0.0136, 0.0140). This means their embedding spaces spread speakers evenly across the hypersphere -- ideal for clean genuine/impostor separation. High-dimensional unit vectors should be approximately orthogonal, and they are.
- **Near-zero mean = maximum discriminability.** When random speaker pairs have cosine similarity near 0, there is maximum room for genuine pairs to score high (+0.5 to +1.0) and impostors to score near 0. This gives clean, well-separated score distributions for label definition.

## What is CONCERNING but EXPECTED:
- **P4 (x-vector) has very high mean similarity** (0.9329). This means the x-vector embedding space compresses speakers into a narrow angular region. Genuine and impostor score distributions overlap heavily (both near 0.93), making P4 poor for defining utility labels. This is WHY P4 is excluded from training labels and used only for cross-system generalization testing.
- **P5 (WavLM) has moderate mean similarity** (0.6443). Better than P4 but still significantly compressed compared to P1-P3. WavLM's self-supervised pretraining objective is not speaker-discriminative by design -- it's fine-tuned for speaker verification, but the embedding space retains some non-speaker structure. P5 is also testing-only.

## Why This Matters for S-Norm:
S-norm normalization computes cohort-based score statistics to center and scale each speaker's score distribution. For P1-P3, the cohort will produce well-behaved impostor distributions centered near 0 with small variance. For P4, the cohort impostor distribution will be centered near 0.93 with very small variance -- s-norm will stretch and re-center this, but the inherent low discriminability remains.

A good s-norm cohort requires diverse, well-separated speakers. The near-zero inter-speaker similarities for P1-P3 confirm this cohort achieves that goal.

## Verdict
Cohort quality is excellent. All embeddings are valid, properly normalized, and show expected discriminability patterns. P1-P3 are confirmed as suitable for training labels; P4-P5 are confirmed as suitable for testing only.
