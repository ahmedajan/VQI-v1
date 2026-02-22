# VQI: Voice Quality Index for Speaker Recognition

A biometric sample quality metric that predicts whether a speech recording will produce reliable speaker verification results. VQI follows the NIST NFIQ 2 methodology (NIST.IR.8382) adapted for the speech domain.

## Overview

VQI produces two complementary scores from 0 to 100:

- **VQI-S (Signal Quality):** Measures recording quality -- noise, reverberation, spectral distortion, dynamics, and channel artifacts. 430 features extracted from the audio signal.
- **VQI-V (Voice Distinctiveness):** Measures how well-suited a voice sample is for speaker recognition -- cepstral stability, formant clarity, and prosodic consistency. 133 features.

Higher scores indicate better quality for speaker recognition systems.

**Key Results:**
- Trained on 1.2 million speech samples from 8 datasets
- Evaluated against 5 speaker recognition systems (ECAPA-TDNN, ResNetSE34, ECAPA2, x-vector, WavLM)
- VQI-S: AUC = 0.8719, best ERC = 47.6% FNMR reduction at 20% rejection
- VQI-V: AUC = 0.8812, best ERC = 26.9% FNMR reduction at 20% rejection
- Cross-system generalization confirmed (trained on 3 providers, evaluated on 5)

## Desktop Application

A standalone Windows desktop application is available with a graphical interface, animated score gauges, and plain-language feedback.

### Download

**[Download VQI v1.0 from Google Drive](https://drive.google.com/drive/folders/1C9p9ENf_eA-GmDh--iXlX6bwLdupAkh6)**

1. Download **`VQI-v1.0-windows.zip`** from the link above
2. Extract the ZIP to any folder on your computer
3. Open the extracted folder and run **`VQI.exe`**

No installation or Python required.

### System Requirements

- Windows 10 or Windows 11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- ~5 GB free disk space
- Audio input device (for microphone recording feature)

### Screenshot

![VQI Desktop Application](screenshots/main_window.png)

### Features

- **File Upload:** Drag-and-drop or browse for audio files (WAV, FLAC, MP3, M4A, OGG)
- **Microphone Recording:** Record directly with device selection and live VU meter
- **Dual Quality Scores:** Animated color-coded 0-100 gauge displays for VQI-S and VQI-V
- **Plain-Language Feedback:** Actionable suggestions to improve recording quality
- **Expert Diagnostics:** Per-feature percentile analysis for technical users
- **Visualization:** Waveform, spectrogram, and mel spectrogram display
- **Export Reports:** Save detailed quality reports as text files

### Troubleshooting

- **VQI.exe does not start:** Ensure the entire folder is extracted, including the `_internal` subfolder. Do not move VQI.exe out of its folder.
- **Slow first scoring:** The application loads machine learning models at startup (~1-2 seconds). Subsequent scores are faster.
- **No audio devices found:** Check that your microphone is connected and Windows recognizes it.
- **Unsupported format:** Convert your audio to WAV or FLAC for best compatibility.

## Python Library

### Quick Start: Score a Single File

```python
from vqi.preprocessing.audio_loader import load_audio
from vqi.preprocessing.normalize import dc_remove_and_normalize
from vqi.preprocessing.vad import energy_vad, reconstruct_from_mask
from vqi.core.feature_orchestrator import compute_all_features
from vqi.core.feature_orchestrator_v import compute_all_features_v
from vqi.prediction.random_forest import load_model, predict_score

# Load and preprocess
waveform, sr = load_audio("path/to/audio.wav")
normalized = dc_remove_and_normalize(waveform)
mask = energy_vad(normalized, sr)
speech = reconstruct_from_mask(normalized, mask)

# Extract features
features_s, intermediates = compute_all_features(speech, sr, mask, waveform)
features_v = compute_all_features_v(speech, sr, mask, intermediates)

# Load model and predict
clf = load_model("models/vqi_rf_model.joblib")
# Select only the 430 trained features (see data/evaluation/selected_features.txt)
# score = predict_score(clf, selected_feature_vector)
```

### Installation

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/VQI.git
cd VQI

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if not using Git LFS)
python scripts/download_models.py
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA recommended for embedding extraction)
- ~16GB RAM for feature extraction
- ~50GB disk for datasets (not included)

## Datasets Required

VQI is trained and evaluated on 8 publicly available speech datasets. You must obtain these independently (see [DATASETS.md](DATASETS.md) for download links and expected directory structure):

| Dataset | Speakers | Utterances | Used For |
|---------|----------|------------|----------|
| VoxCeleb1 | 1,251 | 153,516 | Training + Testing |
| VoxCeleb2 | 6,112 | 1,128,246 | Training |
| LibriSpeech | 2,484 | 292,367 | Training + Testing |
| VCTK | 110 | 44,455 | Testing |
| VOiCES | 300 | 22,000+ | Training |
| CN-Celeb1 | 1,000 | 126,532 | Testing |
| MUSAN | -- | -- | Noise augmentation (training labels) |
| RIR | -- | -- | Reverberation augmentation (training labels) |

## Reproducing the Full Pipeline

The VQI pipeline consists of 8 steps. Each step has dedicated scripts and produces reports in `reports/stepN/`.

### Step 1: Data Collection and Embedding Extraction
Assemble datasets, extract speaker embeddings using 5 providers (P1-P5), compute comparison scores.
```bash
python scripts/inventory_datasets.py
python scripts/create_splits.py
python scripts/extract_embeddings_batched.py
python scripts/compute_scores.py
```

### Step 2: Label Computation
Compute speech durations, set quality thresholds, generate binary labels (Class 0/1), create balanced training set.
```bash
python scripts/run_step2.py
```

### Step 3: Preprocessing Pipeline
Audio loading, DC removal, peak normalization, VAD, quality checks.
```bash
pytest tests/test_preprocessing.py
```

### Step 4: Feature Extraction
Extract 544 VQI-S candidate features and 161 VQI-V candidate features from all training and validation samples.
```bash
python scripts/extract_features.py
```

### Step 5: Feature Evaluation and Selection
Evaluate features using Spearman correlation, ERC contribution, and Random Forest importance. Select top features.
```bash
python scripts/run_step5.py
```
Result: 430 VQI-S features selected, 133 VQI-V features selected.

### Step 6: Random Forest Training
Train Random Forest classifiers for VQI-S and VQI-V.
```bash
python scripts/run_step6.py
```
Result: VQI-S OOB error = 0.1824, VQI-V OOB error = 0.1794.

### Step 7: Model Validation
Validate on 50,000 held-out samples. Compute AUC, CDF shift tests, cross-validation.
```bash
python scripts/run_step7.py
```
Result: VQI-S AUC = 0.8719, VQI-V AUC = 0.8812.

### Step 8: Evaluation of Predictive Power
Evaluate on 3 test datasets (VoxCeleb1-test, VCTK, CN-Celeb) using ERC, Ranked DET, cross-system analysis.
```bash
python scripts/run_step8.py
```

## Pre-trained Models

Pre-trained Random Forest models are available:
- `models/vqi_rf_model.joblib` -- VQI-S model (1000 trees, 430 features, ~192MB)
- `models/vqi_v_rf_model.joblib` -- VQI-V model (1000 trees, 133 features, ~204MB)

These files exceed GitHub's 100MB limit. If not tracked via Git LFS, download them from the [Releases](https://github.com/YOUR_USERNAME/VQI/releases) page.

## Repository Structure

```
VQI/
|-- vqi/                    # Core Python package
|   |-- preprocessing/      # Audio loading, normalization, VAD
|   |-- core/               # Feature orchestration, quality algorithm
|   |-- features/           # 23 frame-level + 32 global feature modules (VQI-S)
|   |-- features_v/         # 5 voice distinctiveness feature modules (VQI-V)
|   |-- prediction/         # Random Forest model loading and prediction
|   |-- evaluation/         # ERC, DET, cross-system evaluation
|   |-- training/           # Feature selection, model training, validation
|   |-- providers/          # Speaker verification system wrappers (P1-P5)
|-- scripts/                # Step execution and visualization scripts
|-- tests/                  # Unit and integration tests
|-- models/                 # Pre-trained RF models (.joblib)
|-- data/                   # Split manifests, labels, selected features
|-- reports/                # Visualizations and analysis for each step
|-- screenshots/            # Desktop application screenshots
|-- CHANGELOG.md            # Application release history
|-- DATASETS.md             # Dataset download links and directory structure
|-- requirements.txt        # Python dependencies
```

## Results Summary

### Error vs. Reject Curves (ERC)

Best FNMR reduction at 20% rejection (FNMR=1% operating point):

| Dataset | Best Provider | VQI-S | VQI-V |
|---------|--------------|-------|-------|
| VoxCeleb1-test | x-vector | 42.1% | -- |
| VCTK | x-vector | 47.6% | 26.9% |
| CN-Celeb | ResNet | 11.9% | 5.2% |

### Ranked DET Separation

EER separation ratio (highest-quality / lowest-quality group):

| Dataset | ResNet | ECAPA | ECAPA2 | x-vector | WavLM |
|---------|--------|-------|--------|----------|-------|
| VoxCeleb1 | 2.04x | 1.72x | 1.41x | 1.10x | 1.29x |
| VCTK | 3.27x | 2.62x | 2.21x | 1.88x | 1.41x |

## Citation

```bibtex
@article{vqi2026,
  title={VQI: Voice Quality Index for Speaker Recognition},
  author={[Author Names]},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
