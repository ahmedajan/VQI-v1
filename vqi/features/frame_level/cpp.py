"""F6: Frame-level Cepstral Peak Prominence (CPP). Features 95-113.

CPP = height of the dominant cepstral peak above a regression line
through the cepstrum.  Higher CPP = more periodic / less noisy voice.
"""

import numpy as np
from ..histogram import aggregate_frame_features

BIN_BOUNDARIES = [2, 4, 6, 8, 10, 12, 15, 18, 22]
PREFIX = "FrameCPP"


def compute_cpp_features(waveform, sr, vad_mask, intermediates):
    """Compute CPP per frame from the power cepstrum."""
    frames = intermediates["frames"]  # (n_fft, n_frames)
    n_fft = intermediates["n_fft"]
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    cpp_values = np.zeros(n_frames)
    # Pitch range for peak search: 60-500 Hz -> quefrency 2ms-16.7ms
    min_quefrency = int(sr / 500)  # ~32 samples at 16kHz
    max_quefrency = int(sr / 60)   # ~267 samples at 16kHz

    for i in range(n_frames):
        frame = frames[:, i] * np.hanning(n_fft)
        # Power spectrum
        spec = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
        # Cepstrum = IFFT of log power spectrum
        log_spec = np.log(spec)
        cepstrum = np.fft.irfft(log_spec)

        # Search range
        lo = min(min_quefrency, len(cepstrum) - 1)
        hi = min(max_quefrency, len(cepstrum))
        if hi <= lo:
            cpp_values[i] = 0.0
            continue

        cep_region = cepstrum[lo:hi]

        # Regression line through cepstrum
        x = np.arange(len(cepstrum))
        coeffs = np.polyfit(x, cepstrum, 1)
        baseline = np.polyval(coeffs, np.arange(lo, hi))

        # CPP = peak above baseline
        peak_val = np.max(cep_region)
        peak_idx = np.argmax(cep_region)
        cpp_values[i] = peak_val - baseline[peak_idx]

    speech_cpp = cpp_values[vad]
    return aggregate_frame_features(speech_cpp, BIN_BOUNDARIES, PREFIX)


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
