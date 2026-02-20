"""G4: Global Clipping (fraction of samples >= 0.99 on RAW waveform)."""

import numpy as np

CLIPPING_THRESHOLD = 0.99


def compute_clipping_features(waveform, sr, vad_mask, intermediates, raw_waveform=None):
    # Use raw waveform (before normalization) if available
    wav = raw_waveform if raw_waveform is not None else waveform
    n_clipped = int(np.sum(np.abs(wav) >= CLIPPING_THRESHOLD))
    ratio = n_clipped / len(wav) if len(wav) > 0 else 0.0
    return {"GlobalClipping": float(ratio)}
