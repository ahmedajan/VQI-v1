# Changelog

## v2.0 (February 2026)

### PCA-90% Pipeline
- PCA-90% dimensionality reduction: 430→99 features (VQI-S), 133→47 features (VQI-V)
- 5-way DR comparison: Full vs PCA-90% vs PCA-95% vs ICA vs Factor Analysis
- PCA-90% selected as deployed model (best balanced OOB loss: -0.0263)
- StandardScaler → PCA → Random Forest scoring pipeline

### Updated Evaluation
- PCA-90% Step 7 validation on 50,000 samples (VQI-S AUC=0.8600, VQI-V AUC=0.8619)
- PCA-90% Step 8 evaluation on 3 test datasets with 5 providers
- ERC reduction up to 38.0% at 20% rejection (PCA-90% model)
- DET EER separation ratios 1.04x–6.41x

### Software v2.0
- Desktop application updated to PCA-90% pipeline
- 200-file conformance test suite (202/202 PASS)

## v1.0 (February 2026)

### Initial Release

- Dual quality scoring: VQI-S (Signal Quality) and VQI-V (Voice Distinctiveness)
- File upload with drag-and-drop support (WAV, FLAC, MP3, M4A, OGG)
- Microphone recording with live VU meter and device selection
- Animated score gauge widgets with color-coded quality bands
- Plain-language feedback with actionable improvement suggestions
- Expert diagnostics with per-feature percentile analysis
- Waveform, spectrogram, and mel spectrogram visualization
- Export detailed quality reports as text files
- Standalone Windows executable (no Python installation required)
- Trained on 1.2 million speech samples from 8 datasets
- Evaluated against 5 speaker recognition systems
