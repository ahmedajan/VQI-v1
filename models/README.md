# Pre-trained VQI Models

This directory contains the pre-trained Random Forest models for VQI scoring.

## Models

| File | Description | Features | Trees | Size |
|------|-------------|----------|-------|------|
| `vqi_rf_model.joblib` | VQI-S (Signal Quality) | 430 | 1000 | ~192MB |
| `vqi_v_rf_model.joblib` | VQI-V (Voice Distinctiveness) | 133 | 1000 | ~204MB |

## Download

These files exceed GitHub's 100MB limit. If they are not present (not using Git LFS), download from:

**GitHub Releases:** [https://github.com/YOUR_USERNAME/VQI/releases](https://github.com/YOUR_USERNAME/VQI/releases)

Or use the download script:
```bash
python scripts/download_models.py
```

## Training

To retrain the models from scratch, follow Steps 1-6 in the main README:
```bash
python scripts/run_step6.py
```

This will generate new model files in this directory.
