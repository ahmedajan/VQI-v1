# Datasets Required for VQI

VQI is trained and evaluated using 8 publicly available speech datasets. These are **not included** in this repository due to size and licensing constraints. You must obtain each dataset independently.

## Dataset Summary

| # | Dataset | Download | Size | Format | Sample Rate |
|---|---------|----------|------|--------|-------------|
| 1 | **VoxCeleb1** | [robots.ox.ac.uk/~vgg/data/voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) | ~30GB | WAV (16kHz) | 16kHz |
| 2 | **VoxCeleb2** | [robots.ox.ac.uk/~vgg/data/voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) | ~70GB (M4A) | M4A -> WAV | 16kHz |
| 3 | **LibriSpeech** | [openslr.org/12](https://www.openslr.org/12/) | ~60GB | FLAC | 16kHz |
| 4 | **VCTK** | [datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | ~11GB | WAV | 48kHz |
| 5 | **VOiCES** | [iarpa.gov/voices](https://lab41.org/datasets/) | ~15GB | WAV | 16kHz |
| 6 | **CN-Celeb1** | [openslr.org/82](https://www.openslr.org/82/) | ~8GB | WAV | 16kHz |
| 7 | **MUSAN** | [openslr.org/17](https://www.openslr.org/17/) | ~11GB | WAV | 16kHz |
| 8 | **RIR** | [openslr.org/28](https://www.openslr.org/28/) | ~5GB | WAV | Various |

## Expected Directory Structure

Place datasets under a `Datasets/` directory (sibling to this repo or configure paths):

```
Datasets/
|-- VoxCeleb1/
|   |-- vox1_dev_wav/
|   |   |-- id10001/
|   |   |   |-- 1zcIwhmdeo4/
|   |   |   |   |-- 00001.wav
|   |-- vox1_test_wav/
|-- VoxCeleb2/
|   |-- dev/
|   |   |-- wav/           # After M4A -> WAV conversion
|-- LibriSpeech/
|   |-- train-clean-100/
|   |-- train-clean-360/
|   |-- train-other-500/
|   |-- test-clean/
|   |-- test-other/
|-- VCTK/
|   |-- wav48_silence_trimmed/
|-- VOiCES/
|   |-- source-16k/
|-- CN-Celeb/
|   |-- data/
|-- MUSAN/
|   |-- noise/
|   |-- speech/
|   |-- music/
|-- RIR/
|   |-- simulated_rirs/
```

## VoxCeleb2 M4A to WAV Conversion

VoxCeleb2 ships in M4A (AAC) format which is not directly supported by torchaudio's soundfile backend. Convert to WAV using ffmpeg:

```bash
python scripts/convert_voxceleb2_to_wav.py --input Datasets/VoxCeleb2/dev/aac --output Datasets/VoxCeleb2/dev/wav
```

Requires ffmpeg installed and accessible on PATH.

## Dataset Roles

- **Training pool (Steps 1-6):** VoxCeleb1 (dev), VoxCeleb2 (dev), LibriSpeech (train), VCTK, VOiCES, CN-Celeb
- **Validation (Step 7):** 50,000 samples randomly drawn from training pool (mutually exclusive with training set)
- **Testing (Step 8):** VoxCeleb1-test, VCTK (eval), CN-Celeb (test), LibriSpeech (test-clean, test-other)
- **Augmentation labels:** MUSAN (noise), RIR (reverberation) -- used during provider score computation, not directly for VQI training

## Split Manifests

Pre-computed split manifests are provided in `data/splits/`:
- `train_pool.csv` -- 1,210,451 files in the training pool
- `val_set.csv` -- 50,000 files for validation
- `test_voxceleb1.csv` -- 4,874 files for VoxCeleb1-test evaluation
- `test_vctk.csv` -- 44,455 files for VCTK evaluation
- `test_cnceleb.csv` -- 126,532 files for CN-Celeb evaluation
- `test_librispeech_clean.csv` / `test_librispeech_other.csv` -- LibriSpeech test sets

These manifests list file paths relative to your Datasets directory. Adjust paths if your directory structure differs.
