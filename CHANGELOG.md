# Changelog

## v4.0 (March 2026)

### DR Optimization
- Systematic evaluation of 8 dimensionality reduction configurations (Full, PCA-80/85/90/95/99%, FA-BIC, ICA-PA)
- Full features (no DR) confirmed optimal for both VQI-S and VQI-V
- All DR methods degrade AUC and ERC compared to full features

### Model Architecture (v4.0)
- VQI-S: StandardScaler + Ridge Regressor (20K balanced training, 430 features)
- VQI-V: StandardScaler + XGBoost Regressor (58K expanded training, 133 features)
- Model size: 6.7MB total (down from 705MB in v2.0)
- Standardized model file naming (`vqi_v4_*`)

### Metrics
- VQI-S: AUC = 0.8812, mean ERC@20% = 7.7%
- VQI-V: AUC = 0.9122, mean ERC@20% = 6.1%
- 200/200 conformance PASS, scores S=[13-99], V=[22-88]

### Report Regeneration
- Complete report regeneration for v4.0: 427 files in `reports/Final Model/`
- Full 5-provider evaluation on 5 test datasets (25 plot types per dataset)
- X1 comparison: v4.0 added to 56-model comprehensive comparison

## v3.0 (March 2026)

### Model Selection (Step X1)
- 56-model comprehensive comparison (7 families x 2 paradigms x 2 data sizes x 2 score types)
- Ridge Regressor selected for VQI-S (best ERC@20% = 20.2%, AUC = 0.8803)
- XGBoost Regressor selected for VQI-V (best ERC@20% = 14.3%, AUC = 0.9130)
- SVM highest AUC but extreme bimodal 0/100 scores unsuitable for production

### Architecture Change
- Replaced Random Forest classifiers with Ridge (S) + XGBoost (V) regressors
- Removed PCA dimensionality reduction (full features)
- Regression output: `model.predict() * 100` -> [0-100] score
- Model size reduced from 705MB to 6.7MB (100x smaller)

### Extended Evaluation
- 5 test datasets: VoxCeleb1, VCTK, CN-Celeb, VPQAD, VSEA DC
- 140 comparison plots in `reports/x1/comparison/`

## v2.0 (February 2026)

### PCA-90% Pipeline
- PCA-90% dimensionality reduction: 430->99 features (VQI-S), 133->47 features (VQI-V)
- 5-way DR comparison: Full vs PCA-90% vs PCA-95% vs ICA vs Factor Analysis
- PCA-90% selected as deployed model (best balanced OOB loss: -0.0263)
- StandardScaler -> PCA -> Random Forest scoring pipeline

### Updated Evaluation
- PCA-90% Step 7 validation on 50,000 samples (VQI-S AUC=0.8600, VQI-V AUC=0.8619)
- PCA-90% Step 8 evaluation on 3 test datasets with 5 providers
- ERC reduction up to 38.0% at 20% rejection (PCA-90% model)
- DET EER separation ratios 1.04x-6.41x

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
