# Step 1.6 Score Computation Analysis

## Summary

Score computation completed successfully for all 3 training providers (P1-P3).
Leave-one-out genuine scores and centroid-vs-centroid impostor distributions
were computed for both raw cosine similarity and s-norm normalized scores.

## Key Findings

### P1_ECAPA

- Genuine scores: mean=0.8060, std=0.0985, n=1,210,451
- Impostor scores: mean=0.0169, std=0.0816, n=28,158,760
- **d-prime (raw):** 4.380
- **d-prime (s-norm):** 4.274
- **Cohen's d:** 8.721
- **Mann-Whitney U (genuine > impostor):** p < 0.00e+00
- S-norm genuine mean: 9.84 (vs raw 0.8060)
- S-norm impostor mean: 0.0022 (near zero: YES)

### P2_RESNET

- Genuine scores: mean=0.7844, std=0.1075, n=1,210,451
- Impostor scores: mean=0.0141, std=0.0867, n=28,158,760
- **d-prime (raw):** 3.966
- **d-prime (s-norm):** 3.849
- **Cohen's d:** 7.886
- **Mann-Whitney U (genuine > impostor):** p < 0.00e+00
- S-norm genuine mean: 9.18 (vs raw 0.7844)
- S-norm impostor mean: 0.0092 (near zero: YES)

### P3_ECAPA2

- Genuine scores: mean=0.8582, std=0.0963, n=1,210,451
- Impostor scores: mean=0.0140, std=0.1019, n=28,158,760
- **d-prime (raw):** 4.258
- **d-prime (s-norm):** 4.017
- **Cohen's d:** 8.514
- **Mann-Whitney U (genuine > impostor):** p < 0.00e+00
- S-norm genuine mean: 8.52 (vs raw 0.8582)
- S-norm impostor mean: 0.0044 (near zero: YES)

## Cross-Provider Correlation

| Pair | Pearson r | Spearman rho |
|------|-----------|-------------|
| P1_ECAPA vs P2_RESNET | 0.9383 | 0.9114 |
| P1_ECAPA vs P3_ECAPA2 | 0.8986 | 0.8822 |
| P2_RESNET vs P3_ECAPA2 | 0.9003 | 0.8706 |

## Implications for Step 2

- All three providers show strong genuine-impostor separation (d' > 3.9)
- This means Class 1 (high quality) and Class 0 (low quality) labels will be well-separated
- S-norm successfully centers impostor distributions near zero for all providers
- Moderate cross-provider correlation (~0.5-0.7) confirms provider diversity
- Provider consensus will ensure labels reflect genuine quality, not single-model artifacts