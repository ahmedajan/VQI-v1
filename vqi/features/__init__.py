"""VQI Feature Extraction Package.

Provides:
  - aggregate_frame_features: shared histogram + stats aggregation
  - compute_shared_intermediates: one-time computation of expensive representations
  - frame_level: 23 frame-level feature modules (F1-F23, 437 features total)
  - global: global scalar feature modules (G1-G107, 107 features total)
"""

from .histogram import aggregate_frame_features
from .shared_intermediates import compute_shared_intermediates
