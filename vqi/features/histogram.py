"""
Shared aggregation module for frame-level features (Sub-task 4.1).

Converts a per-frame value array into 19 features:
  - 10 histogram bins (normalized proportions)
  - 9 distribution statistics (mean, std, skew, kurt, median, IQR, P5, P95, range)

Bins: [-inf, b0, b1, ..., b8, +inf]  (9 boundaries -> 10 bins)
"""

import numpy as np
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis


def aggregate_frame_features(data, bin_boundaries, prefix):
    """Aggregate per-frame values into 19 features.

    Parameters
    ----------
    data : np.ndarray
        1-D array of per-frame values (speech frames only).
    bin_boundaries : list of float
        9 boundary values defining 10 histogram bins.
    prefix : str
        Feature name prefix (e.g. "FrameSNR").

    Returns
    -------
    dict[str, float]
        19 features: {prefix}_Hist0..Hist9, {prefix}_Mean, _Std, _Skew,
        _Kurt, _Median, _IQR, _P5, _P95, _Range.
    """
    features = {}

    # Degenerate case: fewer than 10 speech frames
    if data is None or len(data) < 10:
        for i in range(10):
            features[f"{prefix}_Hist{i}"] = 0.0
        for stat in ("Mean", "Std", "Skew", "Kurt", "Median", "IQR",
                      "P5", "P95", "Range"):
            features[f"{prefix}_{stat}"] = 0.0
        return features

    data = np.asarray(data, dtype=np.float64)

    # Replace NaN/Inf with 0 for robustness
    mask = np.isfinite(data)
    if not mask.all():
        data = data.copy()
        data[~mask] = 0.0

    # ----- Histogram (10 bins) -----
    edges = [-np.inf] + list(bin_boundaries) + [np.inf]
    counts, _ = np.histogram(data, bins=edges)
    total = counts.sum()
    proportions = counts / total if total > 0 else np.zeros(10)
    for i in range(10):
        features[f"{prefix}_Hist{i}"] = float(proportions[i])

    # ----- Distribution statistics -----
    features[f"{prefix}_Mean"] = float(np.mean(data))
    features[f"{prefix}_Std"] = float(np.std(data, ddof=0))
    skew_val = sp_skew(data, bias=True)
    kurt_val = sp_kurtosis(data, bias=True, fisher=True)
    features[f"{prefix}_Skew"] = float(skew_val) if np.isfinite(skew_val) else 0.0
    features[f"{prefix}_Kurt"] = float(kurt_val) if np.isfinite(kurt_val) else 0.0
    features[f"{prefix}_Median"] = float(np.median(data))
    p5, p25, p75, p95 = np.percentile(data, [5, 25, 75, 95])
    features[f"{prefix}_IQR"] = float(p75 - p25)
    features[f"{prefix}_P5"] = float(p5)
    features[f"{prefix}_P95"] = float(p95)
    features[f"{prefix}_Range"] = float(np.max(data) - np.min(data))

    return features
