# Step 1.4 Train/Val/Test Splits - Analysis

## Summary
7 split manifests created with 1,441,871 total utterances:
- **Training pool:** 1,210,451 utterances (7,505 speakers)
- **Validation set:** 50,000 utterances
- **Test sets:** 181,420 utterances across 5 test splits

## Split Details
| Split | Utterances | Speakers |
|-------|-----------|----------|
| train_pool | 1,210,451 | 7,505 |
| val_set | 50,000 | 6,853 |
| test_voxceleb1 | 4,874 | 40 |
| test_vctk | 44,455 | 110 |
| test_librispeech_clean | 2,620 | 40 |
| test_librispeech_other | 2,939 | 33 |
| test_cnceleb | 126,532 | 997 |

## Speaker Distribution (Train Pool)
- Total speakers: 7,505
- Median utterances/speaker: 113
- Mean utterances/speaker: 161
- Min: 18, Max: 951
- Speakers with >= 20 utterances: 7,495

## What is GOOD for VQI:
- **Large training pool** (1,210,451) provides ample candidates for label selection. Even at ~1% label rate, we get ~12K labeled samples (more than the target 8K).
- **No overlap between splits** -- splits are mutually exclusive by construction (validation sampled first, then training pool from remainder, test sets from separate datasets).
- **Diverse test coverage** -- 5 test datasets covering in-the-wild (VoxCeleb1-test), studio (VCTK), clean read (LibriSpeech-clean), noisy read (LibriSpeech-other), and cross-language (CN-Celeb). This ensures VQI is evaluated across the full range of acoustic conditions.
- **Most speakers have multiple utterances** (median=113) -- essential for computing within-speaker genuine scores for label definition.

## What to WATCH:
- **Train pool is dominated by VoxCeleb2** (1,048,003/1,210,451 = 86.6%). The label selection process should monitor whether labels are proportionally distributed across dataset sources.
- **Some speakers may have very few utterances** (min=18). Speakers with <5 utterances may produce unreliable genuine score statistics for label computation.
- **VOiCES contribution** (19,800 utterances) is small relative to VoxCeleb but crucial for reverberant/noisy quality diversity.

## Verdict
Splits are correctly constructed, mutually exclusive, and provide good coverage for training, validation, and testing across diverse conditions.
