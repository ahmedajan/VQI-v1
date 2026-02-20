"""VQI-V (Voice Distinctiveness) feature modules. 161 features total."""

from .cepstral import compute_cepstral_features
from .lp_derived import compute_lp_derived_features
from .formant_identity import compute_formant_identity_features
from .prosodic_voice import compute_prosodic_voice_features
from .distributional_phase import compute_distributional_phase_features

# Ordered list for orchestrator
VQIV_FEATURE_MODULES = [
    (compute_cepstral_features, 65),
    (compute_lp_derived_features, 33),
    (compute_formant_identity_features, 11),
    (compute_prosodic_voice_features, 17),
    (compute_distributional_phase_features, 35),
]

VQIV_FEATURE_COUNT = sum(count for _, count in VQIV_FEATURE_MODULES)
assert VQIV_FEATURE_COUNT == 161, f"Expected 161 VQI-V features, got {VQIV_FEATURE_COUNT}"
