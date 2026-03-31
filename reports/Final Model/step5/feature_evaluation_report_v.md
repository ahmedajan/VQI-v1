# VQI-V Feature Evaluation Report

**Generated:** 2026-02-16
**N_candidates:** 161
**N_selected_V:** 133
**OOB accuracy:** 0.8242

## Selection Funnel
| Stage | Count | Removed |
|-------|-------|---------|
| Candidates | 161 | - |
| Zero-variance removed | 161 | 0 |
| Redundancy removed | 133 | 28 |
| RF pruned | 133 | 0 |

## All 161 Features

| # | Feature | |rho| | Retained | Reason | Importance | ERC AUC |
|---|---------|-------|----------|--------|------------|---------|
| 1 | V_MFCC_Mean_1 | 0.0067 | Yes | Selected | 0.003794 | 0.0582 |
| 2 | V_MFCC_Mean_2 | 0.0707 | Yes | Selected | 0.004938 | 0.0610 |
| 3 | V_MFCC_Mean_3 | 0.0297 | Yes | Selected | 0.004539 | 0.0574 |
| 4 | V_MFCC_Mean_4 | 0.0693 | Yes | Selected | 0.005510 | 0.0603 |
| 5 | V_MFCC_Mean_5 | 0.0083 | Yes | Selected | 0.005106 | 0.0550 |
| 6 | V_MFCC_Mean_6 | 0.0232 | Yes | Selected | 0.005018 | 0.0581 |
| 7 | V_MFCC_Mean_7 | 0.0173 | Yes | Selected | 0.004766 | 0.0548 |
| 8 | V_MFCC_Mean_8 | 0.0904 | Yes | Selected | 0.006854 | 0.0552 |
| 9 | V_MFCC_Mean_9 | 0.0127 | Yes | Selected | 0.005011 | 0.0548 |
| 10 | V_MFCC_Mean_10 | 0.1201 | Yes | Selected | 0.006896 | 0.0582 |
| 11 | V_MFCC_Mean_11 | 0.0093 | Yes | Selected | 0.004545 | 0.0513 |
| 12 | V_MFCC_Mean_12 | 0.0098 | Yes | Selected | 0.004961 | 0.0588 |
| 13 | V_MFCC_Mean_13 | 0.0328 | Yes | Selected | 0.005688 | 0.0576 |
| 14 | V_MFCC_Std_1 | 0.1968 | Yes | Selected | 0.013086 | 0.0599 |
| 15 | V_MFCC_Std_2 | 0.0823 | Yes | Selected | 0.005921 | 0.0561 |
| 16 | V_MFCC_Std_3 | 0.1200 | Yes | Selected | 0.005897 | 0.0559 |
| 17 | V_MFCC_Std_4 | 0.0823 | Yes | Selected | 0.006875 | 0.0547 |
| 18 | V_MFCC_Std_5 | 0.1203 | Yes | Selected | 0.007637 | 0.0549 |
| 19 | V_MFCC_Std_6 | 0.1126 | Yes | Selected | 0.006213 | 0.0532 |
| 20 | V_MFCC_Std_7 | 0.0705 | Yes | Selected | 0.005247 | 0.0544 |
| 21 | V_MFCC_Std_8 | 0.0992 | Yes | Selected | 0.006790 | 0.0559 |
| 22 | V_MFCC_Std_9 | 0.0277 | Yes | Selected | 0.004868 | 0.0584 |
| 23 | V_MFCC_Std_10 | 0.0485 | Yes | Selected | 0.005213 | 0.0560 |
| 24 | V_MFCC_Std_11 | 0.0379 | Yes | Selected | 0.004592 | 0.0589 |
| 25 | V_MFCC_Std_12 | 0.0058 | Yes | Selected | 0.005038 | 0.0576 |
| 26 | V_MFCC_Std_13 | 0.0180 | Yes | Selected | 0.004653 | 0.0576 |
| 27 | V_DeltaMFCC_Mean_1 | 0.1695 | Yes | Selected | 0.031306 | 0.0547 |
| 28 | V_DeltaMFCC_Mean_2 | 0.0470 | Yes | Selected | 0.010713 | 0.0566 |
| 29 | V_DeltaMFCC_Mean_3 | 0.0870 | Yes | Selected | 0.019671 | 0.0558 |
| 30 | V_DeltaMFCC_Mean_4 | 0.0465 | Yes | Selected | 0.011173 | 0.0597 |
| 31 | V_DeltaMFCC_Mean_5 | 0.0058 | Yes | Selected | 0.012062 | 0.0592 |
| 32 | V_DeltaMFCC_Mean_6 | 0.0732 | Yes | Selected | 0.017580 | 0.0618 |
| 33 | V_DeltaMFCC_Mean_7 | 0.0323 | Yes | Selected | 0.010759 | 0.0601 |
| 34 | V_DeltaMFCC_Mean_8 | 0.0093 | Yes | Selected | 0.008299 | 0.0583 |
| 35 | V_DeltaMFCC_Mean_9 | 0.0093 | Yes | Selected | 0.009498 | 0.0602 |
| 36 | V_DeltaMFCC_Mean_10 | 0.0315 | Yes | Selected | 0.009936 | 0.0616 |
| 37 | V_DeltaMFCC_Mean_11 | 0.0290 | Yes | Selected | 0.008170 | 0.0611 |
| 38 | V_DeltaMFCC_Mean_12 | 0.0281 | Yes | Selected | 0.009828 | 0.0600 |
| 39 | V_DeltaMFCC_Mean_13 | 0.0092 | Yes | Selected | 0.009661 | 0.0600 |
| 40 | V_LPCC_Mean_1 | 0.0351 | No | Redundant | - | - |
| 41 | V_LPCC_Mean_2 | 0.0198 | Yes | Selected | 0.004600 | 0.0612 |
| 42 | V_LPCC_Mean_3 | 0.0705 | No | Redundant | - | - |
| 43 | V_LPCC_Mean_4 | 0.0327 | No | Redundant | - | - |
| 44 | V_LPCC_Mean_5 | 0.0432 | No | Redundant | - | - |
| 45 | V_LPCC_Mean_6 | 0.0704 | Yes | Selected | 0.005827 | 0.0610 |
| 46 | V_LPCC_Mean_7 | 0.0625 | No | Redundant | - | - |
| 47 | V_LPCC_Mean_8 | 0.0149 | Yes | Selected | 0.004684 | 0.0595 |
| 48 | V_LPCC_Mean_9 | 0.0472 | Yes | Selected | 0.004530 | 0.0555 |
| 49 | V_LPCC_Mean_10 | 0.0913 | Yes | Selected | 0.005389 | 0.0568 |
| 50 | V_LPCC_Mean_11 | 0.0415 | Yes | Selected | 0.005320 | 0.0594 |
| 51 | V_LPCC_Mean_12 | 0.0053 | Yes | Selected | 0.004460 | 0.0609 |
| 52 | V_LPCC_Mean_13 | 0.0109 | Yes | Selected | 0.004747 | 0.0588 |
| 53 | V_LFCC_Mean_1 | 0.0374 | Yes | Selected | 0.003962 | 0.0552 |
| 54 | V_LFCC_Mean_2 | 0.0181 | No | Redundant | - | - |
| 55 | V_LFCC_Mean_3 | 0.0777 | Yes | Selected | 0.006153 | 0.0559 |
| 56 | V_LFCC_Mean_4 | 0.0509 | Yes | Selected | 0.004969 | 0.0619 |
| 57 | V_LFCC_Mean_5 | 0.0512 | Yes | Selected | 0.005049 | 0.0540 |
| 58 | V_LFCC_Mean_6 | 0.0552 | No | Redundant | - | - |
| 59 | V_LFCC_Mean_7 | 0.0931 | Yes | Selected | 0.005245 | 0.0585 |
| 60 | V_LFCC_Mean_8 | 0.0165 | Yes | Selected | 0.004901 | 0.0615 |
| 61 | V_LFCC_Mean_9 | 0.0808 | Yes | Selected | 0.005332 | 0.0544 |
| 62 | V_LFCC_Mean_10 | 0.0582 | Yes | Selected | 0.006071 | 0.0590 |
| 63 | V_LFCC_Mean_11 | 0.1120 | Yes | Selected | 0.008567 | 0.0541 |
| 64 | V_LFCC_Mean_12 | 0.0429 | Yes | Selected | 0.006329 | 0.0645 |
| 65 | V_LFCC_Mean_13 | 0.0436 | Yes | Selected | 0.004653 | 0.0536 |
| 66 | V_LSF_Mean_1 | 0.0488 | Yes | Selected | 0.004408 | 0.0493 |
| 67 | V_LSF_Mean_2 | 0.0310 | Yes | Selected | 0.004164 | 0.0480 |
| 68 | V_LSF_Mean_3 | 0.0170 | Yes | Selected | 0.004198 | 0.0482 |
| 69 | V_LSF_Mean_4 | 0.0095 | Yes | Selected | 0.004084 | 0.0516 |
| 70 | V_LSF_Mean_5 | 0.0147 | Yes | Selected | 0.003968 | 0.0516 |
| 71 | V_LSF_Mean_6 | 0.0169 | Yes | Selected | 0.004061 | 0.0539 |
| 72 | V_LSF_Mean_7 | 0.0225 | Yes | Selected | 0.004083 | 0.0544 |
| 73 | V_LSF_Mean_8 | 0.0081 | Yes | Selected | 0.004873 | 0.0594 |
| 74 | V_LSF_Mean_9 | 0.0636 | Yes | Selected | 0.005778 | 0.0581 |
| 75 | V_LSF_Mean_10 | 0.0607 | Yes | Selected | 0.004969 | 0.0596 |
| 76 | V_LSF_Mean_11 | 0.0453 | Yes | Selected | 0.004409 | 0.0583 |
| 77 | V_LSF_Mean_12 | 0.0093 | Yes | Selected | 0.005762 | 0.0606 |
| 78 | V_LSF_Mean_13 | 0.0298 | Yes | Selected | 0.005624 | 0.0604 |
| 79 | V_LSF_Mean_14 | 0.0620 | Yes | Selected | 0.010363 | 0.0642 |
| 80 | V_RC_1 | 0.0976 | Yes | Selected | 0.004142 | 0.0529 |
| 81 | V_RC_2 | 0.0176 | Yes | Selected | 0.004063 | 0.0551 |
| 82 | V_RC_3 | 0.0269 | Yes | Selected | 0.005456 | 0.0603 |
| 83 | V_RC_4 | 0.0303 | Yes | Selected | 0.004533 | 0.0532 |
| 84 | V_RC_5 | 0.0391 | No | Redundant | - | - |
| 85 | V_RC_6 | 0.0479 | No | Redundant | - | - |
| 86 | V_RC_7 | 0.0894 | No | Redundant | - | - |
| 87 | V_RC_8 | 0.0099 | Yes | Selected | 0.004794 | 0.0535 |
| 88 | V_LAR_1 | 0.0131 | Yes | Selected | 0.004021 | 0.0481 |
| 89 | V_LAR_2 | 0.0430 | Yes | Selected | 0.004429 | 0.0565 |
| 90 | V_LAR_3 | 0.0234 | No | Redundant | - | - |
| 91 | V_LAR_4 | 0.0274 | No | Redundant | - | - |
| 92 | V_LAR_5 | 0.0398 | Yes | Selected | 0.005705 | 0.0580 |
| 93 | V_LAR_6 | 0.0502 | Yes | Selected | 0.004852 | 0.0545 |
| 94 | V_LAR_7 | 0.0900 | Yes | Selected | 0.005389 | 0.0620 |
| 95 | V_LAR_8 | 0.0084 | No | Redundant | - | - |
| 96 | V_LPCGain_Mean | 0.0314 | No | Redundant | - | - |
| 97 | V_LPCGain_Std | 0.0410 | Yes | Selected | 0.004617 | 0.0601 |
| 98 | V_LPCGain_Range | 0.0183 | Yes | Selected | 0.004571 | 0.0579 |
| 99 | V_F2F1_Ratio | 0.0167 | Yes | Selected | 0.004163 | 0.0622 |
| 100 | V_F3F2_Ratio | 0.0167 | Yes | Selected | 0.004529 | 0.0582 |
| 101 | V_F3F1_Ratio | 0.0170 | Yes | Selected | 0.004240 | 0.0620 |
| 102 | V_VTL | 0.0401 | Yes | Selected | 0.004663 | 0.0519 |
| 103 | V_F1_Dynamics | 0.1235 | Yes | Selected | 0.005774 | 0.0569 |
| 104 | V_F2_Dynamics | 0.1455 | Yes | Selected | 0.007191 | 0.0566 |
| 105 | V_F3_Dynamics | 0.1064 | Yes | Selected | 0.005395 | 0.0561 |
| 106 | V_FormantCentralization | 0.0592 | Yes | Selected | 0.004557 | 0.0520 |
| 107 | V_F1F2_Corr | 0.0774 | Yes | Selected | 0.004620 | 0.0580 |
| 108 | V_F2F3_Corr | 0.0582 | Yes | Selected | 0.004468 | 0.0605 |
| 109 | V_F1F3_Corr | 0.0306 | Yes | Selected | 0.004473 | 0.0605 |
| 110 | V_ArticulationRate | 0.0267 | Yes | Selected | 0.005206 | 0.0609 |
| 111 | V_F0_Mean | 0.0112 | Yes | Selected | 0.005677 | 0.0513 |
| 112 | V_F0_Std | 0.0555 | Yes | Selected | 0.005767 | 0.0489 |
| 113 | V_F0_Range | 0.1315 | Yes | Selected | 0.010140 | 0.0486 |
| 114 | V_F0_Slope | 0.1094 | Yes | Selected | 0.020160 | 0.0543 |
| 115 | V_F0_Median | 0.0099 | No | Redundant | - | - |
| 116 | V_OQ | 0.0035 | Yes | Selected | 0.005385 | 0.0575 |
| 117 | V_CQ | 0.0035 | No | Redundant | - | - |
| 118 | V_SQ | 0.0164 | Yes | Selected | 0.004222 | 0.0490 |
| 119 | V_Rd | 0.0035 | No | Redundant | - | - |
| 120 | V_Ra | 0.0035 | No | Redundant | - | - |
| 121 | V_Rk | 0.0164 | No | Redundant | - | - |
| 122 | V_ModalProportion | 0.0195 | Yes | Selected | 0.005699 | 0.0559 |
| 123 | V_BreathyProportion | 0.0266 | Yes | Selected | 0.006090 | 0.0570 |
| 124 | V_AlphaRatio | 0.0377 | Yes | Selected | 0.004594 | 0.0628 |
| 125 | V_HarmonicRichness | 0.0468 | Yes | Selected | 0.005626 | 0.0502 |
| 126 | V_SpectralTilt | 0.0545 | Yes | Selected | 0.004736 | 0.0600 |
| 127 | V_LTFD_Flatness | 0.0191 | Yes | Selected | 0.004645 | 0.0502 |
| 128 | V_LTFD_Entropy | 0.4235 | Yes | Selected | 0.142375 | 0.0506 |
| 129 | V_LTFD_Kurtosis | 0.0433 | Yes | Selected | 0.005114 | 0.0510 |
| 130 | V_LTFD_Range | 0.1638 | Yes | Selected | 0.013049 | 0.0582 |
| 131 | V_LTAS_0_500 | 0.1080 | Yes | Selected | 0.010094 | 0.0655 |
| 132 | V_LTAS_500_1000 | 0.1593 | Yes | Selected | 0.013749 | 0.0603 |
| 133 | V_LTAS_1000_2000 | 0.1423 | Yes | Selected | 0.011366 | 0.0591 |
| 134 | V_LTAS_2000_4000 | 0.0943 | Yes | Selected | 0.006379 | 0.0530 |
| 135 | V_LTAS_4000_8000 | 0.0169 | Yes | Selected | 0.004286 | 0.0613 |
| 136 | V_LTAS_LowMidRatio | 0.0134 | Yes | Selected | 0.004306 | 0.0641 |
| 137 | V_LTAS_MidHighRatio | 0.0731 | Yes | Selected | 0.004429 | 0.0512 |
| 138 | V_MGDCC_1 | 0.0095 | No | Redundant | - | - |
| 139 | V_MGDCC_2 | 0.0046 | No | Redundant | - | - |
| 140 | V_MGDCC_3 | 0.0105 | Yes | Selected | 0.004882 | 0.0590 |
| 141 | V_MGDCC_4 | 0.0057 | No | Redundant | - | - |
| 142 | V_MGDCC_5 | 0.0082 | No | Redundant | - | - |
| 143 | V_MGDCC_6 | 0.0064 | No | Redundant | - | - |
| 144 | V_MGDCC_7 | 0.0057 | No | Redundant | - | - |
| 145 | V_MGDCC_8 | 0.0070 | Yes | Selected | 0.004840 | 0.0587 |
| 146 | V_MGDCC_9 | 0.0100 | No | Redundant | - | - |
| 147 | V_MGDCC_10 | 0.0067 | Yes | Selected | 0.004814 | 0.0585 |
| 148 | V_MGDCC_11 | 0.0085 | Yes | Selected | 0.004933 | 0.0597 |
| 149 | V_MGDCC_12 | 0.0071 | Yes | Selected | 0.004676 | 0.0588 |
| 150 | V_MGDCC_13 | 0.0055 | Yes | Selected | 0.005213 | 0.0588 |
| 151 | V_Rhythm_nPVI | 0.1476 | Yes | Selected | 0.007561 | 0.0547 |
| 152 | V_Rhythm_VoicedPct | 0.1017 | Yes | Selected | 0.005078 | 0.0215 |
| 153 | V_Rhythm_SpeechSegVar | 0.1618 | Yes | Selected | 0.012406 | 0.0540 |
| 154 | V_Rhythm_SilenceSegVar | 0.1461 | Yes | Selected | 0.008708 | 0.0526 |
| 155 | V_Rhythm_SpeechSilenceRatio | 0.0997 | Yes | Selected | 0.005017 | 0.0215 |
| 156 | V_Rhythm_TempoVar | 0.1172 | Yes | Selected | 0.011365 | 0.0500 |
| 157 | V_SpectralCentroid_Mean | 0.0745 | No | Redundant | - | - |
| 158 | V_SpectralCentroid_Std | 0.1401 | Yes | Selected | 0.005724 | 0.0587 |
| 159 | V_SpectralBW_Mean | 0.0919 | No | Redundant | - | - |
| 160 | V_SpectralEntropy_Mean | 0.0134 | Yes | Selected | 0.004416 | 0.0496 |
| 161 | V_SpectralEntropy_Std | 0.1158 | Yes | Selected | 0.006789 | 0.0585 |

## Redundancy Pairs Removed

| Removed | Kept | Pearson r |
|---------|------|-----------|
| V_Rk | V_SQ | 1.0000 |
| V_Ra | V_OQ | 1.0000 |
| V_CQ | V_OQ | 1.0000 |
| V_Rd | V_OQ | 1.0000 |
| V_RC_7 | V_LAR_7 | 0.9995 |
| V_RC_5 | V_LAR_5 | 0.9988 |
| V_LAR_8 | V_RC_8 | 0.9984 |
| V_MGDCC_1 | V_MGDCC_3 | 0.9976 |
| V_RC_6 | V_LAR_6 | 0.9970 |
| V_LPCC_Mean_1 | V_LFCC_Mean_1 | 0.9968 |
| V_LAR_3 | V_RC_3 | 0.9957 |
| V_LFCC_Mean_2 | V_LPCC_Mean_2 | 0.9952 |
| V_LAR_4 | V_RC_4 | 0.9947 |
| V_MGDCC_2 | V_MGDCC_4 | 0.9941 |
| V_LPCC_Mean_3 | V_LFCC_Mean_3 | 0.9883 |
| V_MGDCC_5 | V_MGDCC_3 | 0.9881 |
| V_LPCC_Mean_4 | V_LFCC_Mean_4 | 0.9819 |
| V_MGDCC_4 | V_MGDCC_6 | 0.9802 |
| V_MGDCC_7 | V_MGDCC_3 | 0.9767 |
| V_LPCC_Mean_5 | V_LFCC_Mean_5 | 0.9760 |
| V_LFCC_Mean_6 | V_LPCC_Mean_6 | 0.9723 |
| V_SpectralCentroid_Mean | V_RC_1 | 0.9720 |
| V_F0_Median | V_F0_Mean | 0.9714 |
| V_SpectralBW_Mean | V_RC_1 | 0.9603 |
| V_MGDCC_9 | V_MGDCC_3 | 0.9600 |
| V_MGDCC_6 | V_MGDCC_8 | 0.9577 |
| V_LPCGain_Mean | V_LPCGain_Std | 0.9560 |
| V_LPCC_Mean_7 | V_LFCC_Mean_7 | 0.9504 |

## Top 10 by RF Importance

| Feature | Importance |
|---------|------------|
| V_LTFD_Entropy | 0.142375 |
| V_DeltaMFCC_Mean_1 | 0.031306 |
| V_F0_Slope | 0.020160 |
| V_DeltaMFCC_Mean_3 | 0.019671 |
| V_DeltaMFCC_Mean_6 | 0.017580 |
| V_LTAS_500_1000 | 0.013749 |
| V_MFCC_Std_1 | 0.013086 |
| V_LTFD_Range | 0.013049 |
| V_Rhythm_SpeechSegVar | 0.012406 |
| V_DeltaMFCC_Mean_5 | 0.012062 |

## Top 10 by ERC AUC

| Feature | ERC AUC |
|---------|---------|
| V_Rhythm_SpeechSilenceRatio | 0.0215 |
| V_Rhythm_VoicedPct | 0.0215 |
| V_LSF_Mean_2 | 0.0480 |
| V_LAR_1 | 0.0481 |
| V_LSF_Mean_3 | 0.0482 |
| V_F0_Range | 0.0486 |
| V_F0_Std | 0.0489 |
| V_SQ | 0.0490 |
| V_LSF_Mean_1 | 0.0493 |
| V_SpectralEntropy_Mean | 0.0496 |

## VQI-S vs VQI-V Feature Overlap

- VQI-S selected: 430 features
- VQI-V selected: 133 features
- Overlap: 0 features

No overlap (VQI-S and VQI-V use completely disjoint feature sets).
This is expected since VQI-S features (Frame*/Global) and VQI-V features (V_*) have different naming conventions.