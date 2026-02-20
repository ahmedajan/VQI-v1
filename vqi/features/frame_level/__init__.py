"""Frame-level feature modules (F1-F23). Each produces 19 aggregated features."""

from .snr import compute_snr_features
from .spectral_flatness import compute_spectral_flatness_features
from .pitch_confidence import compute_pitch_confidence_features
from .hnr import compute_hnr_features
from .mfcc_variance import compute_mfcc_variance_features
from .cpp import compute_cpp_features
from .spectral_entropy import compute_spectral_entropy_features
from .spectral_rolloff import compute_spectral_rolloff_features
from .spectral_slope import compute_spectral_slope_features
from .spectral_contrast import compute_spectral_contrast_features
from .spectral_bandwidth import compute_spectral_bandwidth_features
from .spectral_flux import compute_spectral_flux_features
from .frame_energy import compute_frame_energy_features
from .autocorrelation_peak import compute_autocorrelation_peak_features
from .spectral_skewness import compute_spectral_skewness_features
from .spectral_kurtosis import compute_spectral_kurtosis_features
from .spectral_crest import compute_spectral_crest_features
from .shr import compute_shr_features
from .fundamental_frequency import compute_fundamental_frequency_features
from .zcr_frame import compute_zcr_frame_features
from .gne import compute_gne_features
from .delta_mfcc import compute_delta_mfcc_features
from .delta_delta_mfcc import compute_delta_delta_mfcc_features

# Ordered list of (function, prefix) for the orchestrator
FRAME_FEATURE_MODULES = [
    (compute_snr_features, "FrameSNR"),
    (compute_spectral_flatness_features, "FrameSF"),
    (compute_pitch_confidence_features, "FramePC"),
    (compute_hnr_features, "FrameHNR"),
    (compute_mfcc_variance_features, "FrameMFCCVar"),
    (compute_cpp_features, "FrameCPP"),
    (compute_spectral_entropy_features, "FrameSE"),
    (compute_spectral_rolloff_features, "FrameSR"),
    (compute_spectral_slope_features, "FrameSS"),
    (compute_spectral_contrast_features, "FrameSC"),
    (compute_spectral_bandwidth_features, "FrameSBW"),
    (compute_spectral_flux_features, "FrameSFlux"),
    (compute_frame_energy_features, "FrameE"),
    (compute_autocorrelation_peak_features, "FrameAC"),
    (compute_spectral_skewness_features, "FrameSSkew"),
    (compute_spectral_kurtosis_features, "FrameSKurt"),
    (compute_spectral_crest_features, "FrameSCF"),
    (compute_shr_features, "FrameSHR"),
    (compute_fundamental_frequency_features, "FrameF0"),
    (compute_zcr_frame_features, "FrameZCR"),
    (compute_gne_features, "FrameGNE"),
    (compute_delta_mfcc_features, "FrameDeltaMFCC"),
    (compute_delta_delta_mfcc_features, "FrameDeltaDeltaMFCC"),
]
