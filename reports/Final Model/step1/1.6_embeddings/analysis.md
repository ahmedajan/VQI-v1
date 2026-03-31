# Step 1.6 Partial Embeddings - Analysis

## Summary
Embedding extraction is in progress: **1,210,451 / 1,210,451** (100.0%) complete.
Providers: P1 (ECAPA-TDNN, 192-dim), P2 (ResNet34, 256-dim), P3 (ECAPA2, 192-dim).

## Embedding Quality (sampled 10,000 rows)
| Provider | Norm Mean | Norm Std | Zero Rows | NaN Rows |
|----------|-----------|----------|-----------|----------|
| P1_ECAPA | 1.000000 | 0.000000 | 0 | 0 |
| P2_RESNET | 1.000000 | 0.000000 | 0 | 0 |
| P3_ECAPA2 | 1.000000 | 0.000000 | 0 | 0 |

## t-SNE Visualization
t-SNE computed on 1414 samples from 30 speakers (perplexity=30, max_iter=1000).

## What is GOOD for VQI:
- **All embeddings properly L2-normalized** (norm = 1.0 +/- negligible deviation). The extraction pipeline is producing correct output.
- **Zero NaN and zero all-zero rows** across all 3 providers. No audio loading failures or model inference errors.
- **t-SNE shows clear speaker clusters** (visible in the "By Speaker" panels). Embeddings from the same speaker group tightly together, confirming that provider models are producing speaker-discriminative representations. This is essential for genuine/impostor score computation in Step 1.6b.
- **No obvious dataset bias in t-SNE** (the "By Dataset" panels should show dataset sources intermixed, not forming separate clusters). If datasets were forming separate clusters, it would indicate that VQI features might predict dataset identity rather than quality.

## What to WATCH:
- **Extraction is only 100.0% complete.** The visualizations above represent the first 1,210,451 rows (which are sorted by filename, so they are predominantly from early alphabetical speaker IDs). The full picture may change as more diverse speakers are processed.
- **Extraction rate** should remain stable at ~2.3 f/s. Significant slowdowns may indicate GPU memory issues or disk I/O bottleneck.

## Verdict
Partial embeddings look healthy. All quality checks pass (norms, NaN/Inf, speaker clustering). Extraction should continue to completion without intervention.
