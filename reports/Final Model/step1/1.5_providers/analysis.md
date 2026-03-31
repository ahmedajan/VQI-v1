# Step 1.5 Provider Setup - Analysis

## Summary
5 speaker recognition providers installed, verified, and tested on CUDA. All produce correct embedding dimensions and pass the genuine > impostor sanity check.

## Provider Details
| ID | Model | Architecture | Embedding Dim | EER (VoxCeleb1-O) | Role |
|----|-------|-------------|---------------|-------------------|------|
| P1 | ECAPA-TDNN | TDNN + SE + Attention | 192 | 0.87% | Training |
| P2 | ResNet34 | CNN + SE + ASP | 256 | 1.05% | Training |
| P3 | ECAPA2 | Hybrid 1D+2D Conv | 192 | 0.17% | Training |
| P4 | x-vector | Classical TDNN | 512 | 3.13% | Testing only |
| P5 | WavLM-SV | SSL Transformer | 512 | ~0.6% | Testing only |

## What is GOOD for VQI:
- **Three architecturally diverse training providers.** P1 (TDNN-based), P2 (CNN-based), and P3 (Hybrid) represent fundamentally different approaches to speaker embedding extraction. When all three agree that a sample is high/low quality, the label reflects a genuine quality consensus rather than one model's idiosyncrasy.
- **P1-P3 are all state-of-the-art** (<1.1% EER on VoxCeleb1-O). High-performing providers produce cleaner score distributions with larger genuine/impostor gaps, leading to less ambiguous labels.
- **P3 (ECAPA2) achieves 0.17% EER** -- essentially the best publicly available speaker verification model. Including it ensures that even the most discriminative system contributes to label definition.
- **P4 and P5 provide complementary test perspectives.** P4 (x-vector, 3.13% EER) represents older/weaker systems -- if VQI helps even weak systems, it proves the quality metric captures fundamental signal properties, not just features that help modern systems. P5 (WavLM, ~0.6% EER) represents the self-supervised paradigm -- generalization to this confirms VQI is architecture-agnostic.

## What is CONCERNING but ACCEPTABLE:
- **P2 and P3 have the same embedding dimension change from blueprint.** P2 was changed from 512-dim (voxceleb_trainer) to 256-dim (SpeechBrain ResNet34), and P3 from TitaNet to ECAPA2, due to Windows compatibility issues. These substitutions are equivalent or superior in performance.
- **P5 actual dimension is 512, not 256 as originally stated in blueprint.** This was corrected after verification.

## Platform Notes:
- SpeechBrain models require `LocalStrategy.COPY` on Windows (symlinks need admin).
- ECAPA2 TorchScript requires CPU load -> .to(cuda) + `_jit_override_can_fuse_on_gpu(False)`.
- All providers verified on CUDA with correct output dimensions and score ranges.

## Verdict
All 5 providers operational and verified. The 3-training/2-testing split provides both label quality (diverse consensus from strong models) and rigorous generalization testing (weak + different-paradigm models).
