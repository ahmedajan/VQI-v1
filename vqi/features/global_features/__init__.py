"""Global scalar feature modules (G1-G107). Each returns a dict of named features."""

from .duration import compute_duration_features
from .vad_ratio import compute_vad_ratio_features
from .energy import compute_energy_features
from .clipping import compute_clipping_features
from .bandwidth import compute_bandwidth_features
from .reverberation import compute_reverberation_features
from .spectral_centroid import compute_spectral_centroid_features
from .zcr import compute_zcr_features
from .jitter_shimmer import compute_jitter_shimmer_features
from .speech_rate import compute_speech_rate_features
from .speaker_turns import compute_speaker_turns_features
from .noise_estimation import compute_noise_estimation_features
from .spectral_quality import compute_spectral_quality_features
from .voice_quality import compute_voice_quality_features
from .energy_dynamics import compute_energy_dynamics_features
from .temporal_dynamics import compute_temporal_dynamics_features
from .distortion_detection import compute_distortion_detection_features
from .channel_quality import compute_channel_quality_features
from .subband_analysis import compute_subband_analysis_features
from .speech_models import compute_speech_models_features
from .signal_integrity import compute_signal_integrity_features
from .intelligibility import compute_intelligibility_features
from .glottal_source import compute_glottal_source_features
from .voice_quality_mdvp import compute_voice_quality_mdvp_features
from .tremor import compute_tremor_features
from .formant import compute_formant_features
from .egemaps_spectral import compute_egemaps_spectral_features
from .prosody_rhythm import compute_prosody_rhythm_features
from .fw_snr import compute_fw_snr_features
from .srmr import compute_srmr_features
from .dnn_quality import compute_dnn_quality_features
from .clinical_composites import compute_clinical_composites_features

# Ordered list matching blueprint G1-G107
# Each entry: (function, [feature_names])
GLOBAL_FEATURE_MODULES = [
    (compute_duration_features, ["GlobalDuration"]),
    (compute_vad_ratio_features, ["GlobalVADRatio"]),
    (compute_energy_features, ["GlobalEnergy"]),
    (compute_clipping_features, ["GlobalClipping"]),
    (compute_bandwidth_features, ["GlobalBandwidth"]),
    (compute_reverberation_features, ["GlobalReverb", "RT60_Est", "C50_Est", "ModulationDepth"]),
    (compute_spectral_centroid_features, ["SpectralCentroid"]),
    (compute_zcr_features, ["GlobalZCR"]),
    (compute_jitter_shimmer_features, ["Jitter", "Shimmer", "JitterPPQ5", "JitterRAP",
                                        "ShimmerAPQ3", "ShimmerAPQ5", "ShimmerAPQ11"]),
    (compute_speech_rate_features, ["SpeechRate"]),
    (compute_speaker_turns_features, ["SpeakerTurns"]),
    (compute_noise_estimation_features, ["SegmentalSNR", "WADA_SNR", "NoiseFloorLevel",
                                          "NoiseBandwidth", "NoiseStationarity"]),
    (compute_spectral_quality_features, ["LTAS_Slope", "LTAS_Tilt", "SpectralFluxMean",
                                          "SpectralFluxStd", "SpectralRolloff",
                                          "SpectralEntropy", "SpectralSkewness",
                                          "SpectralKurtosis", "SpectralCrest"]),
    (compute_voice_quality_features, ["CPP_Mean", "CPP_Std", "NHR", "H1H2", "H1A3",
                                       "UnvoicedFrameRatio"]),
    (compute_energy_dynamics_features, ["EnergyRange", "EnergyContourVariance"]),
    (compute_temporal_dynamics_features, ["PauseDurationMean", "PauseRate", "LongestPause",
                                           "SpeechContinuity", "OnsetStrengthMean",
                                           "OnsetStrengthStd"]),
    (compute_distortion_detection_features, ["ClickRate", "DropoutRate", "SaturationRatio",
                                              "MusicalNoiseLevel", "QuantizationNoise"]),
    (compute_channel_quality_features, ["DCOffset", "PowerLineHum", "AGC_Activity"]),
    (compute_subband_analysis_features, ["SubbandSNR_Low", "SubbandSNR_Mid", "SubbandSNR_High",
                                          "LowToHighEnergyRatio"]),
    (compute_speech_models_features, ["LPCResidualEnergy", "VocalTractRegularity"]),
    (compute_signal_integrity_features, ["InterruptionCount"]),
    (compute_intelligibility_features, ["SII_Estimate", "ModulationSpectrumArea"]),
    (compute_glottal_source_features, ["NAQ", "QOQ", "HRF", "PSP", "GCI_Rate", "GOI_Regularity"]),
    (compute_voice_quality_mdvp_features, ["MDVP_Fo", "MDVP_Jitter", "MDVP_Shimmer",
                                            "MDVP_NHR", "MDVP_VTI", "MDVP_SPI", "MDVP_DVB"]),
    (compute_tremor_features, ["Tremor_Freq", "Tremor_Intensity", "Tremor_CycleVariation",
                                "Tremor_Regularity"]),
    (compute_formant_features, ["F1_Mean", "F2_Mean", "F3_Mean", "FormantDispersion",
                                 "F1_BW", "F2_BW", "F3_BW"]),
    (compute_egemaps_spectral_features, ["AlphaRatio", "HammarbergIndex", "SpectralSlope0500_1500"]),
    (compute_prosody_rhythm_features, ["MeanF0", "F0_StdDev"]),
    (compute_fw_snr_features, ["FrequencyWeightedSNR"]),
    (compute_srmr_features, ["SRMR"]),
    (compute_dnn_quality_features, ["DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL",
                                     "NISQA_MOS", "NISQA_NOI", "NISQA_DIS",
                                     "NISQA_COL", "NISQA_LOUD"]),
    (compute_clinical_composites_features, ["AVQI", "DSI", "CSID"]),
]

# Total expected global features
GLOBAL_FEATURE_COUNT = sum(len(names) for _, names in GLOBAL_FEATURE_MODULES)
assert GLOBAL_FEATURE_COUNT == 107, f"Expected 107 global features, got {GLOBAL_FEATURE_COUNT}"
