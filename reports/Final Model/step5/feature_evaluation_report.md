# VQI-S Feature Evaluation Report

**Generated:** 2026-02-16
**N_candidates:** 544
**N_selected:** 430
**OOB accuracy:** 0.8202

## Selection Funnel
| Stage | Count | Removed |
|-------|-------|---------|
| Candidates | 544 | - |
| Zero-variance removed | 513 | 31 |
| Redundancy removed | 449 | 64 |
| RF pruned | 430 | 19 |

## All 544 Features

| # | Feature | |rho| | Retained | Reason | Importance | ERC AUC |
|---|---------|-------|----------|--------|------------|---------|
| 1 | FrameSNR_Hist0 | 0.1625 | Yes | Selected | 0.010194 | 0.1349 |
| 2 | FrameSNR_Hist1 | 0.1183 | Yes | Selected | 0.001565 | 0.0552 |
| 3 | FrameSNR_Hist2 | 0.1142 | Yes | Selected | 0.001951 | 0.0518 |
| 4 | FrameSNR_Hist3 | 0.0857 | Yes | Selected | 0.001894 | 0.0548 |
| 5 | FrameSNR_Hist4 | 0.0229 | Yes | Selected | 0.001814 | 0.0590 |
| 6 | FrameSNR_Hist5 | 0.1390 | Yes | Selected | 0.002341 | 0.0586 |
| 7 | FrameSNR_Hist6 | 0.1583 | Yes | Selected | 0.002430 | 0.0561 |
| 8 | FrameSNR_Hist7 | 0.1407 | Yes | Selected | 0.002277 | 0.0564 |
| 9 | FrameSNR_Hist8 | 0.1019 | Yes | Selected | 0.000620 | 0.0559 |
| 10 | FrameSNR_Hist9 | 0.0505 | No | RF pruned | - | - |
| 11 | FrameSNR_Mean | 0.1513 | No | Redundant | - | - |
| 12 | FrameSNR_Std | 0.1393 | Yes | Selected | 0.002146 | 0.0590 |
| 13 | FrameSNR_Skew | 0.0137 | Yes | Selected | 0.001592 | 0.0502 |
| 14 | FrameSNR_Kurt | 0.0030 | Yes | Selected | 0.001535 | 0.0542 |
| 15 | FrameSNR_Median | 0.1424 | No | Redundant | - | - |
| 16 | FrameSNR_IQR | 0.1044 | Yes | Selected | 0.001739 | 0.0594 |
| 17 | FrameSNR_P5 | 0.0095 | Yes | Selected | 0.001757 | 0.0547 |
| 18 | FrameSNR_P95 | 0.1575 | Yes | Selected | 0.002529 | 0.0565 |
| 19 | FrameSNR_Range | 0.2009 | Yes | Selected | 0.004312 | 0.0547 |
| 20 | FrameSF_Hist0 | 0.0142 | Yes | Selected | 0.001133 | 0.0589 |
| 21 | FrameSF_Hist1 | 0.0679 | Yes | Selected | 0.001611 | 0.0494 |
| 22 | FrameSF_Hist2 | 0.0123 | Yes | Selected | 0.001348 | 0.0497 |
| 23 | FrameSF_Hist3 | 0.0683 | Yes | Selected | 0.001408 | 0.0502 |
| 24 | FrameSF_Hist4 | 0.0890 | Yes | Selected | 0.001319 | 0.0545 |
| 25 | FrameSF_Hist5 | 0.0495 | Yes | Selected | 0.001194 | 0.0606 |
| 26 | FrameSF_Hist6 | 0.0211 | Yes | Selected | 0.001382 | 0.0598 |
| 27 | FrameSF_Hist7 | 0.0612 | Yes | Selected | 0.000857 | 0.0598 |
| 28 | FrameSF_Hist8 | 0.0217 | No | RF pruned | - | - |
| 29 | FrameSF_Hist9 | 0.0022 | No | RF pruned | - | - |
| 30 | FrameSF_Mean | 0.0363 | Yes | Selected | 0.001143 | 0.0563 |
| 31 | FrameSF_Std | 0.0389 | No | Redundant | - | - |
| 32 | FrameSF_Skew | 0.0818 | Yes | Selected | 0.001207 | 0.0562 |
| 33 | FrameSF_Kurt | 0.0769 | Yes | Selected | 0.001146 | 0.0557 |
| 34 | FrameSF_Median | 0.0501 | Yes | Selected | 0.001239 | 0.0494 |
| 35 | FrameSF_IQR | 0.0534 | Yes | Selected | 0.001108 | 0.0565 |
| 36 | FrameSF_P5 | 0.1178 | Yes | Selected | 0.002747 | 0.0479 |
| 37 | FrameSF_P95 | 0.0424 | Yes | Selected | 0.001387 | 0.0594 |
| 38 | FrameSF_Range | 0.0731 | Yes | Selected | 0.001832 | 0.0592 |
| 39 | FramePC_Hist0 | 0.0245 | No | Redundant | - | - |
| 40 | FramePC_Hist1 | 0.0269 | Yes | Selected | 0.001533 | 0.0603 |
| 41 | FramePC_Hist2 | 0.0169 | Yes | Selected | 0.001415 | 0.0605 |
| 42 | FramePC_Hist3 | 0.0216 | Yes | Selected | 0.001300 | 0.0597 |
| 43 | FramePC_Hist4 | 0.0292 | Yes | Selected | 0.001240 | 0.0605 |
| 44 | FramePC_Hist5 | 0.0529 | Yes | Selected | 0.001334 | 0.0589 |
| 45 | FramePC_Hist6 | 0.0596 | Yes | Selected | 0.001386 | 0.0591 |
| 46 | FramePC_Hist7 | 0.0920 | Yes | Selected | 0.001445 | 0.0586 |
| 47 | FramePC_Hist8 | 0.1104 | Yes | Selected | 0.001821 | 0.0575 |
| 48 | FramePC_Hist9 | 0.1324 | Yes | Selected | 0.001902 | 0.0570 |
| 49 | FramePC_Mean | 0.0330 | Yes | Selected | 0.001480 | 0.0591 |
| 50 | FramePC_Std | 0.0529 | Yes | Selected | 0.001478 | 0.0594 |
| 51 | FramePC_Skew | 0.0170 | Yes | Selected | 0.001527 | 0.0483 |
| 52 | FramePC_Kurt | 0.0166 | Yes | Selected | 0.001773 | 0.0480 |
| 53 | FramePC_Median | 0.0199 | Yes | Selected | 0.001149 | 0.0586 |
| 54 | FramePC_IQR | 0.0251 | No | Redundant | - | - |
| 55 | FramePC_P5 | 0.0882 | Yes | Selected | 0.000506 | 0.0602 |
| 56 | FramePC_P95 | 0.0438 | No | Redundant | - | - |
| 57 | FramePC_Range | 0.1508 | Yes | Selected | 0.002006 | 0.0606 |
| 58 | FrameHNR_Hist0 | 0.1489 | Yes | Selected | 0.003130 | 0.0537 |
| 59 | FrameHNR_Hist1 | 0.0834 | Yes | Selected | 0.001879 | 0.0608 |
| 60 | FrameHNR_Hist2 | 0.1071 | Yes | Selected | 0.002087 | 0.0553 |
| 61 | FrameHNR_Hist3 | 0.0769 | Yes | Selected | 0.001692 | 0.0605 |
| 62 | FrameHNR_Hist4 | 0.0292 | Yes | Selected | 0.001630 | 0.0565 |
| 63 | FrameHNR_Hist5 | 0.0114 | Yes | Selected | 0.001301 | 0.0565 |
| 64 | FrameHNR_Hist6 | 0.0111 | Yes | Selected | 0.001174 | 0.0569 |
| 65 | FrameHNR_Hist7 | 0.0245 | Yes | Selected | 0.001030 | 0.0562 |
| 66 | FrameHNR_Hist8 | 0.0461 | Yes | Selected | 0.000943 | 0.0574 |
| 67 | FrameHNR_Hist9 | 0.0578 | Yes | Selected | 0.000826 | 0.0554 |
| 68 | FrameHNR_Mean | 0.0829 | Yes | Selected | 0.001752 | 0.0581 |
| 69 | FrameHNR_Std | 0.0241 | No | Redundant | - | - |
| 70 | FrameHNR_Skew | 0.1000 | Yes | Selected | 0.001687 | 0.0555 |
| 71 | FrameHNR_Kurt | 0.0059 | Yes | Selected | 0.001542 | 0.0538 |
| 72 | FrameHNR_Median | 0.1070 | Yes | Selected | 0.001592 | 0.0578 |
| 73 | FrameHNR_IQR | 0.0150 | Yes | Selected | 0.001374 | 0.0559 |
| 74 | FrameHNR_P5 | 0.0458 | No | RF pruned | - | - |
| 75 | FrameHNR_P95 | 0.0123 | No | Redundant | - | - |
| 76 | FrameHNR_Range | 0.1060 | Yes | Selected | 0.001632 | 0.0444 |
| 77 | FrameMFCCVar_Hist0 | 0.1254 | Yes | Selected | 0.005278 | 0.0606 |
| 78 | FrameMFCCVar_Hist1 | 0.0160 | Yes | Selected | 0.001265 | 0.0571 |
| 79 | FrameMFCCVar_Hist2 | 0.0196 | Yes | Selected | 0.001488 | 0.0571 |
| 80 | FrameMFCCVar_Hist3 | 0.0368 | Yes | Selected | 0.001339 | 0.0552 |
| 81 | FrameMFCCVar_Hist4 | 0.0564 | Yes | Selected | 0.001423 | 0.0519 |
| 82 | FrameMFCCVar_Hist5 | 0.0476 | Yes | Selected | 0.001423 | 0.0514 |
| 83 | FrameMFCCVar_Hist6 | 0.0091 | Yes | Selected | 0.001437 | 0.0522 |
| 84 | FrameMFCCVar_Hist7 | 0.0491 | Yes | Selected | 0.001574 | 0.0521 |
| 85 | FrameMFCCVar_Hist8 | 0.0483 | Yes | Selected | 0.001586 | 0.0546 |
| 86 | FrameMFCCVar_Hist9 | 0.0543 | Yes | Selected | 0.001253 | 0.0598 |
| 87 | FrameMFCCVar_Mean | 0.0536 | Yes | Selected | 0.001416 | 0.0584 |
| 88 | FrameMFCCVar_Std | 0.1243 | Yes | Selected | 0.001948 | 0.0589 |
| 89 | FrameMFCCVar_Skew | 0.1020 | Yes | Selected | 0.001626 | 0.0517 |
| 90 | FrameMFCCVar_Kurt | 0.0212 | Yes | Selected | 0.001665 | 0.0511 |
| 91 | FrameMFCCVar_Median | 0.0389 | No | Redundant | - | - |
| 92 | FrameMFCCVar_IQR | 0.1232 | No | Redundant | - | - |
| 93 | FrameMFCCVar_P5 | 0.0148 | Yes | Selected | 0.001507 | 0.0567 |
| 94 | FrameMFCCVar_P95 | 0.1039 | Yes | Selected | 0.001861 | 0.0586 |
| 95 | FrameMFCCVar_Range | 0.1643 | Yes | Selected | 0.003858 | 0.0576 |
| 96 | FrameCPP_Hist0 | 0.0048 | No | RF pruned | - | - |
| 97 | FrameCPP_Hist1 | 0.0048 | No | Redundant | - | - |
| 98 | FrameCPP_Hist2 | 0.0000 | No | Constant | - | - |
| 99 | FrameCPP_Hist3 | 0.0000 | No | Constant | - | - |
| 100 | FrameCPP_Hist4 | 0.0000 | No | Constant | - | - |
| 101 | FrameCPP_Hist5 | 0.0000 | No | Constant | - | - |
| 102 | FrameCPP_Hist6 | 0.0000 | No | Constant | - | - |
| 103 | FrameCPP_Hist7 | 0.0000 | No | Constant | - | - |
| 104 | FrameCPP_Hist8 | 0.0000 | No | Constant | - | - |
| 105 | FrameCPP_Hist9 | 0.0000 | No | Constant | - | - |
| 106 | FrameCPP_Mean | 0.0421 | No | Redundant | - | - |
| 107 | FrameCPP_Std | 0.0279 | No | Redundant | - | - |
| 108 | FrameCPP_Skew | 0.1224 | Yes | Selected | 0.001775 | 0.0532 |
| 109 | FrameCPP_Kurt | 0.1209 | Yes | Selected | 0.001988 | 0.0523 |
| 110 | FrameCPP_Median | 0.0525 | Yes | Selected | 0.001664 | 0.0571 |
| 111 | FrameCPP_IQR | 0.0523 | Yes | Selected | 0.001708 | 0.0586 |
| 112 | FrameCPP_P5 | 0.0079 | Yes | Selected | 0.001861 | 0.0503 |
| 113 | FrameCPP_P95 | 0.0281 | Yes | Selected | 0.001723 | 0.0578 |
| 114 | FrameCPP_Range | 0.0611 | Yes | Selected | 0.001959 | 0.0567 |
| 115 | FrameSE_Hist0 | 0.0591 | Yes | Selected | 0.001597 | 0.0569 |
| 116 | FrameSE_Hist1 | 0.0365 | Yes | Selected | 0.001295 | 0.0609 |
| 117 | FrameSE_Hist2 | 0.1051 | Yes | Selected | 0.001564 | 0.0604 |
| 118 | FrameSE_Hist3 | 0.1045 | Yes | Selected | 0.001791 | 0.0530 |
| 119 | FrameSE_Hist4 | 0.0550 | Yes | Selected | 0.001427 | 0.0502 |
| 120 | FrameSE_Hist5 | 0.0215 | Yes | Selected | 0.001255 | 0.0469 |
| 121 | FrameSE_Hist6 | 0.0593 | Yes | Selected | 0.001450 | 0.0478 |
| 122 | FrameSE_Hist7 | 0.1099 | Yes | Selected | 0.001358 | 0.0519 |
| 123 | FrameSE_Hist8 | 0.1139 | Yes | Selected | 0.001261 | 0.0563 |
| 124 | FrameSE_Hist9 | 0.0734 | Yes | Selected | 0.001101 | 0.0574 |
| 125 | FrameSE_Mean | 0.0134 | No | Redundant | - | - |
| 126 | FrameSE_Std | 0.1158 | Yes | Selected | 0.001654 | 0.0585 |
| 127 | FrameSE_Skew | 0.0433 | Yes | Selected | 0.001474 | 0.0622 |
| 128 | FrameSE_Kurt | 0.0927 | Yes | Selected | 0.001319 | 0.0601 |
| 129 | FrameSE_Median | 0.0387 | Yes | Selected | 0.001193 | 0.0480 |
| 130 | FrameSE_IQR | 0.0963 | Yes | Selected | 0.001384 | 0.0564 |
| 131 | FrameSE_P5 | 0.0519 | Yes | Selected | 0.001578 | 0.0502 |
| 132 | FrameSE_P95 | 0.1067 | Yes | Selected | 0.001290 | 0.0579 |
| 133 | FrameSE_Range | 0.0448 | Yes | Selected | 0.001936 | 0.0589 |
| 134 | FrameSR_Hist0 | 0.0570 | Yes | Selected | 0.001742 | 0.0640 |
| 135 | FrameSR_Hist1 | 0.0756 | Yes | Selected | 0.001693 | 0.0555 |
| 136 | FrameSR_Hist2 | 0.0457 | Yes | Selected | 0.001754 | 0.0492 |
| 137 | FrameSR_Hist3 | 0.0294 | Yes | Selected | 0.001609 | 0.0500 |
| 138 | FrameSR_Hist4 | 0.0122 | Yes | Selected | 0.001514 | 0.0475 |
| 139 | FrameSR_Hist5 | 0.0377 | Yes | Selected | 0.001603 | 0.0484 |
| 140 | FrameSR_Hist6 | 0.1020 | Yes | Selected | 0.001631 | 0.0485 |
| 141 | FrameSR_Hist7 | 0.1178 | Yes | Selected | 0.001534 | 0.0522 |
| 142 | FrameSR_Hist8 | 0.0705 | Yes | Selected | 0.001318 | 0.0567 |
| 143 | FrameSR_Hist9 | 0.0791 | Yes | Selected | 0.001196 | 0.0611 |
| 144 | FrameSR_Mean | 0.0658 | No | Redundant | - | - |
| 145 | FrameSR_Std | 0.1159 | No | Redundant | - | - |
| 146 | FrameSR_Skew | 0.0753 | Yes | Selected | 0.001035 | 0.0600 |
| 147 | FrameSR_Kurt | 0.0990 | Yes | Selected | 0.001067 | 0.0588 |
| 148 | FrameSR_Median | 0.0372 | Yes | Selected | 0.000963 | 0.0465 |
| 149 | FrameSR_IQR | 0.0446 | Yes | Selected | 0.001094 | 0.0514 |
| 150 | FrameSR_P5 | 0.0344 | Yes | Selected | 0.000904 | 0.0482 |
| 151 | FrameSR_P95 | 0.1097 | Yes | Selected | 0.001368 | 0.0605 |
| 152 | FrameSR_Range | 0.1156 | Yes | Selected | 0.001731 | 0.0602 |
| 153 | FrameSS_Hist0 | 0.0165 | No | RF pruned | - | - |
| 154 | FrameSS_Hist1 | 0.0317 | Yes | Selected | 0.000546 | 0.0552 |
| 155 | FrameSS_Hist2 | 0.0431 | Yes | Selected | 0.001638 | 0.0505 |
| 156 | FrameSS_Hist3 | 0.0120 | Yes | Selected | 0.001665 | 0.0561 |
| 157 | FrameSS_Hist4 | 0.0232 | Yes | Selected | 0.001602 | 0.0620 |
| 158 | FrameSS_Hist5 | 0.0244 | Yes | Selected | 0.001488 | 0.0574 |
| 159 | FrameSS_Hist6 | 0.0483 | Yes | Selected | 0.001410 | 0.0560 |
| 160 | FrameSS_Hist7 | 0.0431 | Yes | Selected | 0.001503 | 0.0580 |
| 161 | FrameSS_Hist8 | 0.0511 | Yes | Selected | 0.001298 | 0.0635 |
| 162 | FrameSS_Hist9 | 0.0976 | Yes | Selected | 0.001282 | 0.0613 |
| 163 | FrameSS_Mean | 0.0155 | Yes | Selected | 0.001505 | 0.0584 |
| 164 | FrameSS_Std | 0.1025 | Yes | Selected | 0.001641 | 0.0616 |
| 165 | FrameSS_Skew | 0.0146 | Yes | Selected | 0.001926 | 0.0619 |
| 166 | FrameSS_Kurt | 0.0238 | Yes | Selected | 0.001619 | 0.0622 |
| 167 | FrameSS_Median | 0.0092 | No | Redundant | - | - |
| 168 | FrameSS_IQR | 0.0883 | Yes | Selected | 0.001736 | 0.0602 |
| 169 | FrameSS_P5 | 0.0314 | Yes | Selected | 0.001954 | 0.0568 |
| 170 | FrameSS_P95 | 0.0837 | Yes | Selected | 0.001491 | 0.0604 |
| 171 | FrameSS_Range | 0.1555 | Yes | Selected | 0.002404 | 0.0604 |
| 172 | FrameSC_Hist0 | 0.0962 | Yes | Selected | 0.004166 | 0.0617 |
| 173 | FrameSC_Hist1 | 0.1163 | Yes | Selected | 0.009348 | 0.0654 |
| 174 | FrameSC_Hist2 | 0.0971 | No | Redundant | - | - |
| 175 | FrameSC_Hist3 | 0.0304 | Yes | Selected | 0.003880 | 0.0458 |
| 176 | FrameSC_Hist4 | 0.0953 | Yes | Selected | 0.001664 | 0.0510 |
| 177 | FrameSC_Hist5 | 0.1003 | Yes | Selected | 0.001567 | 0.0552 |
| 178 | FrameSC_Hist6 | 0.1135 | Yes | Selected | 0.002279 | 0.0572 |
| 179 | FrameSC_Hist7 | 0.0140 | No | RF pruned | - | - |
| 180 | FrameSC_Hist8 | 0.0009 | No | RF pruned | - | - |
| 181 | FrameSC_Hist9 | 0.0000 | No | Constant | - | - |
| 182 | FrameSC_Mean | 0.1054 | Yes | Selected | 0.002103 | 0.0505 |
| 183 | FrameSC_Std | 0.0232 | No | Redundant | - | - |
| 184 | FrameSC_Skew | 0.1161 | Yes | Selected | 0.002848 | 0.0583 |
| 185 | FrameSC_Kurt | 0.0942 | Yes | Selected | 0.002714 | 0.0586 |
| 186 | FrameSC_Median | 0.0929 | No | Redundant | - | - |
| 187 | FrameSC_IQR | 0.0299 | Yes | Selected | 0.001695 | 0.0597 |
| 188 | FrameSC_P5 | 0.0748 | Yes | Selected | 0.004013 | 0.0464 |
| 189 | FrameSC_P95 | 0.0958 | Yes | Selected | 0.001631 | 0.0546 |
| 190 | FrameSC_Range | 0.0316 | Yes | Selected | 0.001728 | 0.0588 |
| 191 | FrameSBW_Hist0 | 0.0807 | Yes | Selected | 0.001921 | 0.0617 |
| 192 | FrameSBW_Hist1 | 0.0370 | No | RF pruned | - | - |
| 193 | FrameSBW_Hist2 | 0.1145 | Yes | Selected | 0.003208 | 0.0583 |
| 194 | FrameSBW_Hist3 | 0.0654 | Yes | Selected | 0.001401 | 0.0590 |
| 195 | FrameSBW_Hist4 | 0.0235 | Yes | Selected | 0.001636 | 0.0543 |
| 196 | FrameSBW_Hist5 | 0.0368 | Yes | Selected | 0.001657 | 0.0507 |
| 197 | FrameSBW_Hist6 | 0.0404 | Yes | Selected | 0.001583 | 0.0528 |
| 198 | FrameSBW_Hist7 | 0.0113 | Yes | Selected | 0.001374 | 0.0572 |
| 199 | FrameSBW_Hist8 | 0.0139 | Yes | Selected | 0.001547 | 0.0623 |
| 200 | FrameSBW_Hist9 | 0.0221 | No | RF pruned | - | - |
| 201 | FrameSBW_Mean | 0.0251 | Yes | Selected | 0.001263 | 0.0551 |
| 202 | FrameSBW_Std | 0.0766 | Yes | Selected | 0.001763 | 0.0626 |
| 203 | FrameSBW_Skew | 0.0835 | Yes | Selected | 0.001680 | 0.0577 |
| 204 | FrameSBW_Kurt | 0.1192 | Yes | Selected | 0.001666 | 0.0575 |
| 205 | FrameSBW_Median | 0.0142 | No | Redundant | - | - |
| 206 | FrameSBW_IQR | 0.1159 | Yes | Selected | 0.001691 | 0.0621 |
| 207 | FrameSBW_P5 | 0.0948 | Yes | Selected | 0.001568 | 0.0515 |
| 208 | FrameSBW_P95 | 0.0108 | Yes | Selected | 0.001839 | 0.0620 |
| 209 | FrameSBW_Range | 0.0625 | Yes | Selected | 0.002223 | 0.0611 |
| 210 | FrameSFlux_Hist0 | 0.4064 | Yes | Selected | 0.067423 | 0.0674 |
| 211 | FrameSFlux_Hist1 | 0.1216 | Yes | Selected | 0.001370 | 0.0592 |
| 212 | FrameSFlux_Hist2 | 0.0882 | Yes | Selected | 0.001520 | 0.0605 |
| 213 | FrameSFlux_Hist3 | 0.0374 | Yes | Selected | 0.001424 | 0.0622 |
| 214 | FrameSFlux_Hist4 | 0.0179 | Yes | Selected | 0.001530 | 0.0623 |
| 215 | FrameSFlux_Hist5 | 0.0633 | Yes | Selected | 0.001566 | 0.0602 |
| 216 | FrameSFlux_Hist6 | 0.0793 | Yes | Selected | 0.001455 | 0.0557 |
| 217 | FrameSFlux_Hist7 | 0.0300 | Yes | Selected | 0.003341 | 0.0454 |
| 218 | FrameSFlux_Hist8 | 0.0634 | Yes | Selected | 0.001335 | 0.0533 |
| 219 | FrameSFlux_Hist9 | 0.0619 | Yes | Selected | 0.001741 | 0.0624 |
| 220 | FrameSFlux_Mean | 0.0487 | No | Redundant | - | - |
| 221 | FrameSFlux_Std | 0.0518 | No | Redundant | - | - |
| 222 | FrameSFlux_Skew | 0.0617 | Yes | Selected | 0.001681 | 0.0607 |
| 223 | FrameSFlux_Kurt | 0.0357 | Yes | Selected | 0.001880 | 0.0534 |
| 224 | FrameSFlux_Median | 0.0518 | Yes | Selected | 0.001338 | 0.0518 |
| 225 | FrameSFlux_IQR | 0.0438 | No | Redundant | - | - |
| 226 | FrameSFlux_P5 | 0.0312 | Yes | Selected | 0.001710 | 0.0455 |
| 227 | FrameSFlux_P95 | 0.0489 | Yes | Selected | 0.001776 | 0.0620 |
| 228 | FrameSFlux_Range | 0.0527 | Yes | Selected | 0.001661 | 0.0603 |
| 229 | FrameE_Hist0 | 0.0393 | Yes | Selected | 0.001004 | 0.0579 |
| 230 | FrameE_Hist1 | 0.1013 | Yes | Selected | 0.001175 | 0.0552 |
| 231 | FrameE_Hist2 | 0.1810 | Yes | Selected | 0.003693 | 0.0554 |
| 232 | FrameE_Hist3 | 0.1424 | Yes | Selected | 0.002907 | 0.0534 |
| 233 | FrameE_Hist4 | 0.1120 | Yes | Selected | 0.002528 | 0.0509 |
| 234 | FrameE_Hist5 | 0.0936 | Yes | Selected | 0.002399 | 0.0494 |
| 235 | FrameE_Hist6 | 0.0698 | Yes | Selected | 0.001977 | 0.0509 |
| 236 | FrameE_Hist7 | 0.0891 | Yes | Selected | 0.001771 | 0.0597 |
| 237 | FrameE_Hist8 | 0.1482 | Yes | Selected | 0.005180 | 0.0641 |
| 238 | FrameE_Hist9 | 0.0805 | Yes | Selected | 0.002383 | 0.0638 |
| 239 | FrameE_Mean | 0.1830 | Yes | Selected | 0.006015 | 0.0653 |
| 240 | FrameE_Std | 0.1287 | No | Redundant | - | - |
| 241 | FrameE_Skew | 0.0507 | Yes | Selected | 0.001565 | 0.0501 |
| 242 | FrameE_Kurt | 0.0620 | Yes | Selected | 0.001667 | 0.0585 |
| 243 | FrameE_Median | 0.1542 | No | Redundant | - | - |
| 244 | FrameE_IQR | 0.1037 | No | Redundant | - | - |
| 245 | FrameE_P5 | 0.1942 | Yes | Selected | 0.004491 | 0.0635 |
| 246 | FrameE_P95 | 0.1435 | Yes | Selected | 0.004941 | 0.0637 |
| 247 | FrameE_Range | 0.0756 | Yes | Selected | 0.002335 | 0.0573 |
| 248 | FrameAC_Hist0 | 0.0136 | Yes | Selected | 0.001620 | 0.0623 |
| 249 | FrameAC_Hist1 | 0.0782 | Yes | Selected | 0.001394 | 0.0597 |
| 250 | FrameAC_Hist2 | 0.0736 | Yes | Selected | 0.001390 | 0.0516 |
| 251 | FrameAC_Hist3 | 0.0174 | Yes | Selected | 0.001655 | 0.0486 |
| 252 | FrameAC_Hist4 | 0.0168 | Yes | Selected | 0.001670 | 0.0489 |
| 253 | FrameAC_Hist5 | 0.0477 | Yes | Selected | 0.001593 | 0.0527 |
| 254 | FrameAC_Hist6 | 0.1091 | Yes | Selected | 0.002247 | 0.0593 |
| 255 | FrameAC_Hist7 | 0.0308 | Yes | Selected | 0.001672 | 0.0621 |
| 256 | FrameAC_Hist8 | 0.0370 | Yes | Selected | 0.001029 | 0.0558 |
| 257 | FrameAC_Hist9 | 0.0115 | Yes | Selected | 0.000651 | 0.0548 |
| 258 | FrameAC_Mean | 0.0348 | Yes | Selected | 0.001374 | 0.0565 |
| 259 | FrameAC_Std | 0.0457 | Yes | Selected | 0.001741 | 0.0606 |
| 260 | FrameAC_Skew | 0.0583 | Yes | Selected | 0.001598 | 0.0490 |
| 261 | FrameAC_Kurt | 0.0480 | Yes | Selected | 0.001354 | 0.0496 |
| 262 | FrameAC_Median | 0.0281 | No | Redundant | - | - |
| 263 | FrameAC_IQR | 0.0512 | Yes | Selected | 0.001476 | 0.0618 |
| 264 | FrameAC_P5 | 0.0731 | Yes | Selected | 0.001452 | 0.0496 |
| 265 | FrameAC_P95 | 0.0156 | Yes | Selected | 0.001512 | 0.0565 |
| 266 | FrameAC_Range | 0.0571 | Yes | Selected | 0.001938 | 0.0559 |
| 267 | FrameSSkew_Hist0 | 0.1250 | Yes | Selected | 0.001364 | 0.0579 |
| 268 | FrameSSkew_Hist1 | 0.1184 | Yes | Selected | 0.001625 | 0.0559 |
| 269 | FrameSSkew_Hist2 | 0.1380 | Yes | Selected | 0.001603 | 0.0526 |
| 270 | FrameSSkew_Hist3 | 0.1338 | Yes | Selected | 0.001819 | 0.0503 |
| 271 | FrameSSkew_Hist4 | 0.1109 | Yes | Selected | 0.001636 | 0.0491 |
| 272 | FrameSSkew_Hist5 | 0.0773 | Yes | Selected | 0.001522 | 0.0485 |
| 273 | FrameSSkew_Hist6 | 0.0417 | Yes | Selected | 0.001447 | 0.0482 |
| 274 | FrameSSkew_Hist7 | 0.0280 | Yes | Selected | 0.001591 | 0.0481 |
| 275 | FrameSSkew_Hist8 | 0.1057 | Yes | Selected | 0.002019 | 0.0509 |
| 276 | FrameSSkew_Hist9 | 0.0201 | No | Redundant | - | - |
| 277 | FrameSSkew_Mean | 0.0341 | Yes | Selected | 0.001582 | 0.0610 |
| 278 | FrameSSkew_Std | 0.0492 | Yes | Selected | 0.001622 | 0.0597 |
| 279 | FrameSSkew_Skew | 0.0451 | Yes | Selected | 0.001880 | 0.0589 |
| 280 | FrameSSkew_Kurt | 0.0297 | No | Redundant | - | - |
| 281 | FrameSSkew_Median | 0.0224 | Yes | Selected | 0.001404 | 0.0622 |
| 282 | FrameSSkew_IQR | 0.0951 | Yes | Selected | 0.002035 | 0.0610 |
| 283 | FrameSSkew_P5 | 0.1388 | Yes | Selected | 0.001497 | 0.0570 |
| 284 | FrameSSkew_P95 | 0.0737 | Yes | Selected | 0.001752 | 0.0597 |
| 285 | FrameSSkew_Range | 0.0387 | Yes | Selected | 0.002017 | 0.0609 |
| 286 | FrameSKurt_Hist0 | 0.0705 | Yes | Selected | 0.001456 | 0.0504 |
| 287 | FrameSKurt_Hist1 | 0.1053 | Yes | Selected | 0.001479 | 0.0488 |
| 288 | FrameSKurt_Hist2 | 0.0859 | Yes | Selected | 0.001456 | 0.0487 |
| 289 | FrameSKurt_Hist3 | 0.0651 | Yes | Selected | 0.001376 | 0.0483 |
| 290 | FrameSKurt_Hist4 | 0.0297 | Yes | Selected | 0.001363 | 0.0484 |
| 291 | FrameSKurt_Hist5 | 0.0188 | Yes | Selected | 0.001424 | 0.0480 |
| 292 | FrameSKurt_Hist6 | 0.0559 | Yes | Selected | 0.001825 | 0.0512 |
| 293 | FrameSKurt_Hist7 | 0.1019 | Yes | Selected | 0.001893 | 0.0510 |
| 294 | FrameSKurt_Hist8 | 0.1146 | Yes | Selected | 0.001790 | 0.0551 |
| 295 | FrameSKurt_Hist9 | 0.0352 | Yes | Selected | 0.001380 | 0.0618 |
| 296 | FrameSKurt_Mean | 0.0525 | Yes | Selected | 0.001774 | 0.0601 |
| 297 | FrameSKurt_Std | 0.0225 | Yes | Selected | 0.001588 | 0.0603 |
| 298 | FrameSKurt_Skew | 0.0113 | Yes | Selected | 0.002079 | 0.0566 |
| 299 | FrameSKurt_Kurt | 0.0225 | Yes | Selected | 0.002162 | 0.0568 |
| 300 | FrameSKurt_Median | 0.0314 | Yes | Selected | 0.001436 | 0.0616 |
| 301 | FrameSKurt_IQR | 0.0755 | Yes | Selected | 0.001702 | 0.0611 |
| 302 | FrameSKurt_P5 | 0.0883 | Yes | Selected | 0.001378 | 0.0600 |
| 303 | FrameSKurt_P95 | 0.0862 | Yes | Selected | 0.002039 | 0.0595 |
| 304 | FrameSKurt_Range | 0.0365 | Yes | Selected | 0.001960 | 0.0601 |
| 305 | FrameSCF_Hist0 | 0.0034 | No | RF pruned | - | - |
| 306 | FrameSCF_Hist1 | 0.0177 | No | RF pruned | - | - |
| 307 | FrameSCF_Hist2 | 0.0516 | No | RF pruned | - | - |
| 308 | FrameSCF_Hist3 | 0.0972 | Yes | Selected | 0.001245 | 0.0579 |
| 309 | FrameSCF_Hist4 | 0.1116 | Yes | Selected | 0.001156 | 0.0569 |
| 310 | FrameSCF_Hist5 | 0.1219 | Yes | Selected | 0.001235 | 0.0534 |
| 311 | FrameSCF_Hist6 | 0.0419 | Yes | Selected | 0.001286 | 0.0492 |
| 312 | FrameSCF_Hist7 | 0.0284 | Yes | Selected | 0.001511 | 0.0511 |
| 313 | FrameSCF_Hist8 | 0.0867 | Yes | Selected | 0.001384 | 0.0587 |
| 314 | FrameSCF_Hist9 | 0.0125 | No | Redundant | - | - |
| 315 | FrameSCF_Mean | 0.0184 | Yes | Selected | 0.001176 | 0.0573 |
| 316 | FrameSCF_Std | 0.0471 | Yes | Selected | 0.001609 | 0.0561 |
| 317 | FrameSCF_Skew | 0.0603 | Yes | Selected | 0.001455 | 0.0506 |
| 318 | FrameSCF_Kurt | 0.0851 | Yes | Selected | 0.001591 | 0.0549 |
| 319 | FrameSCF_Median | 0.0119 | No | Redundant | - | - |
| 320 | FrameSCF_IQR | 0.0503 | Yes | Selected | 0.001544 | 0.0561 |
| 321 | FrameSCF_P5 | 0.1175 | Yes | Selected | 0.001345 | 0.0552 |
| 322 | FrameSCF_P95 | 0.0193 | Yes | Selected | 0.001626 | 0.0559 |
| 323 | FrameSCF_Range | 0.0177 | Yes | Selected | 0.001904 | 0.0594 |
| 324 | FrameSHR_Hist0 | 0.0112 | Yes | Selected | 0.001701 | 0.0578 |
| 325 | FrameSHR_Hist1 | 0.0272 | Yes | Selected | 0.001613 | 0.0599 |
| 326 | FrameSHR_Hist2 | 0.0300 | Yes | Selected | 0.001502 | 0.0595 |
| 327 | FrameSHR_Hist3 | 0.0257 | Yes | Selected | 0.001462 | 0.0600 |
| 328 | FrameSHR_Hist4 | 0.0177 | Yes | Selected | 0.001607 | 0.0572 |
| 329 | FrameSHR_Hist5 | 0.0074 | Yes | Selected | 0.001792 | 0.0549 |
| 330 | FrameSHR_Hist6 | 0.0296 | Yes | Selected | 0.001981 | 0.0584 |
| 331 | FrameSHR_Hist7 | 0.0335 | Yes | Selected | 0.002118 | 0.0581 |
| 332 | FrameSHR_Hist8 | 0.0106 | Yes | Selected | 0.001497 | 0.0536 |
| 333 | FrameSHR_Hist9 | 0.0081 | Yes | Selected | 0.001469 | 0.0586 |
| 334 | FrameSHR_Mean | 0.0016 | No | Redundant | - | - |
| 335 | FrameSHR_Std | 0.0295 | Yes | Selected | 0.001612 | 0.0567 |
| 336 | FrameSHR_Skew | 0.0144 | Yes | Selected | 0.001680 | 0.0560 |
| 337 | FrameSHR_Kurt | 0.0127 | Yes | Selected | 0.001629 | 0.0607 |
| 338 | FrameSHR_Median | 0.0066 | Yes | Selected | 0.001597 | 0.0579 |
| 339 | FrameSHR_IQR | 0.0498 | Yes | Selected | 0.001604 | 0.0532 |
| 340 | FrameSHR_P5 | 0.0217 | Yes | Selected | 0.001475 | 0.0508 |
| 341 | FrameSHR_P95 | 0.0160 | Yes | Selected | 0.001667 | 0.0499 |
| 342 | FrameSHR_Range | 0.1214 | Yes | Selected | 0.002567 | 0.0503 |
| 343 | FrameF0_Hist0 | 0.0228 | No | Redundant | - | - |
| 344 | FrameF0_Hist1 | 0.0612 | Yes | Selected | 0.000933 | 0.0547 |
| 345 | FrameF0_Hist2 | 0.0041 | Yes | Selected | 0.001168 | 0.0575 |
| 346 | FrameF0_Hist3 | 0.0277 | Yes | Selected | 0.001453 | 0.0623 |
| 347 | FrameF0_Hist4 | 0.0104 | Yes | Selected | 0.001356 | 0.0605 |
| 348 | FrameF0_Hist5 | 0.0631 | Yes | Selected | 0.001444 | 0.0597 |
| 349 | FrameF0_Hist6 | 0.0637 | Yes | Selected | 0.001403 | 0.0552 |
| 350 | FrameF0_Hist7 | 0.0417 | Yes | Selected | 0.001026 | 0.0535 |
| 351 | FrameF0_Hist8 | 0.0346 | Yes | Selected | 0.000605 | 0.0536 |
| 352 | FrameF0_Hist9 | 0.0419 | Yes | Selected | 0.000783 | 0.0511 |
| 353 | FrameF0_Mean | 0.0114 | Yes | Selected | 0.001518 | 0.0579 |
| 354 | FrameF0_Std | 0.0251 | Yes | Selected | 0.001570 | 0.0502 |
| 355 | FrameF0_Skew | 0.0372 | Yes | Selected | 0.001766 | 0.0482 |
| 356 | FrameF0_Kurt | 0.0242 | Yes | Selected | 0.001691 | 0.0539 |
| 357 | FrameF0_Median | 0.0134 | Yes | Selected | 0.001191 | 0.0580 |
| 358 | FrameF0_IQR | 0.0089 | Yes | Selected | 0.001575 | 0.0501 |
| 359 | FrameF0_P5 | 0.0443 | No | RF pruned | - | - |
| 360 | FrameF0_P95 | 0.0126 | Yes | Selected | 0.001641 | 0.0502 |
| 361 | FrameF0_Range | 0.0797 | Yes | Selected | 0.001403 | 0.0448 |
| 362 | FrameZCR_Hist0 | 0.0161 | Yes | Selected | 0.001374 | 0.0604 |
| 363 | FrameZCR_Hist1 | 0.0723 | Yes | Selected | 0.002033 | 0.0636 |
| 364 | FrameZCR_Hist2 | 0.0529 | Yes | Selected | 0.001916 | 0.0628 |
| 365 | FrameZCR_Hist3 | 0.0516 | Yes | Selected | 0.001528 | 0.0586 |
| 366 | FrameZCR_Hist4 | 0.0962 | Yes | Selected | 0.001752 | 0.0510 |
| 367 | FrameZCR_Hist5 | 0.0742 | Yes | Selected | 0.001936 | 0.0473 |
| 368 | FrameZCR_Hist6 | 0.0173 | Yes | Selected | 0.001593 | 0.0478 |
| 369 | FrameZCR_Hist7 | 0.0769 | Yes | Selected | 0.001368 | 0.0497 |
| 370 | FrameZCR_Hist8 | 0.1315 | Yes | Selected | 0.001569 | 0.0518 |
| 371 | FrameZCR_Hist9 | 0.1385 | Yes | Selected | 0.001354 | 0.0553 |
| 372 | FrameZCR_Mean | 0.0664 | Yes | Selected | 0.001025 | 0.0502 |
| 373 | FrameZCR_Std | 0.1446 | Yes | Selected | 0.001438 | 0.0589 |
| 374 | FrameZCR_Skew | 0.0276 | Yes | Selected | 0.001339 | 0.0608 |
| 375 | FrameZCR_Kurt | 0.0563 | Yes | Selected | 0.001245 | 0.0596 |
| 376 | FrameZCR_Median | 0.0398 | Yes | Selected | 0.001115 | 0.0463 |
| 377 | FrameZCR_IQR | 0.0943 | Yes | Selected | 0.000926 | 0.0536 |
| 378 | FrameZCR_P5 | 0.0714 | Yes | Selected | 0.001646 | 0.0465 |
| 379 | FrameZCR_P95 | 0.1386 | No | Redundant | - | - |
| 380 | FrameZCR_Range | 0.1597 | Yes | Selected | 0.002216 | 0.0595 |
| 381 | FrameGNE_Hist0 | 0.0632 | Yes | Selected | 0.002325 | 0.0593 |
| 382 | FrameGNE_Hist1 | 0.0758 | Yes | Selected | 0.000790 | 0.0591 |
| 383 | FrameGNE_Hist2 | 0.0659 | Yes | Selected | 0.000919 | 0.0592 |
| 384 | FrameGNE_Hist3 | 0.0505 | Yes | Selected | 0.001190 | 0.0563 |
| 385 | FrameGNE_Hist4 | 0.0531 | Yes | Selected | 0.001132 | 0.0570 |
| 386 | FrameGNE_Hist5 | 0.0675 | Yes | Selected | 0.001182 | 0.0570 |
| 387 | FrameGNE_Hist6 | 0.0935 | Yes | Selected | 0.001311 | 0.0550 |
| 388 | FrameGNE_Hist7 | 0.0857 | Yes | Selected | 0.001509 | 0.0524 |
| 389 | FrameGNE_Hist8 | 0.0236 | Yes | Selected | 0.001301 | 0.0492 |
| 390 | FrameGNE_Hist9 | 0.0508 | Yes | Selected | 0.001024 | 0.0568 |
| 391 | FrameGNE_Mean | 0.0332 | Yes | Selected | 0.001112 | 0.0571 |
| 392 | FrameGNE_Std | 0.0477 | No | Redundant | - | - |
| 393 | FrameGNE_Skew | 0.0864 | Yes | Selected | 0.001197 | 0.0549 |
| 394 | FrameGNE_Kurt | 0.0846 | Yes | Selected | 0.001107 | 0.0578 |
| 395 | FrameGNE_Median | 0.0391 | Yes | Selected | 0.001190 | 0.1246 |
| 396 | FrameGNE_IQR | 0.0607 | Yes | Selected | 0.001107 | 0.0548 |
| 397 | FrameGNE_P5 | 0.0512 | Yes | Selected | 0.001162 | 0.0533 |
| 398 | FrameGNE_P95 | 0.1073 | Yes | Selected | 0.002755 | 0.1511 |
| 399 | FrameGNE_Range | 0.0453 | Yes | Selected | 0.002076 | 0.2552 |
| 400 | FrameDeltaMFCC_Hist0 | 0.1183 | Yes | Selected | 0.002436 | 0.0432 |
| 401 | FrameDeltaMFCC_Hist1 | 0.1172 | No | Redundant | - | - |
| 402 | FrameDeltaMFCC_Hist2 | 0.1467 | Yes | Selected | 0.002146 | 0.0592 |
| 403 | FrameDeltaMFCC_Hist3 | 0.0034 | No | RF pruned | - | - |
| 404 | FrameDeltaMFCC_Hist4 | 0.0000 | No | Constant | - | - |
| 405 | FrameDeltaMFCC_Hist5 | 0.0000 | No | Constant | - | - |
| 406 | FrameDeltaMFCC_Hist6 | 0.0000 | No | Constant | - | - |
| 407 | FrameDeltaMFCC_Hist7 | 0.0000 | No | Constant | - | - |
| 408 | FrameDeltaMFCC_Hist8 | 0.0000 | No | Constant | - | - |
| 409 | FrameDeltaMFCC_Hist9 | 0.0000 | No | Constant | - | - |
| 410 | FrameDeltaMFCC_Mean | 0.0973 | Yes | Selected | 0.002721 | 0.0614 |
| 411 | FrameDeltaMFCC_Std | 0.1272 | Yes | Selected | 0.002951 | 0.0618 |
| 412 | FrameDeltaMFCC_Skew | 0.1044 | Yes | Selected | 0.003017 | 0.0536 |
| 413 | FrameDeltaMFCC_Kurt | 0.0959 | Yes | Selected | 0.002857 | 0.0526 |
| 414 | FrameDeltaMFCC_Median | 0.0557 | Yes | Selected | 0.003103 | 0.0604 |
| 415 | FrameDeltaMFCC_IQR | 0.0927 | No | Redundant | - | - |
| 416 | FrameDeltaMFCC_P5 | 0.0122 | Yes | Selected | 0.002243 | 0.0601 |
| 417 | FrameDeltaMFCC_P95 | 0.1206 | No | Redundant | - | - |
| 418 | FrameDeltaMFCC_Range | 0.1910 | Yes | Selected | 0.005911 | 0.0596 |
| 419 | FrameDeltaDeltaMFCC_Hist0 | 0.1122 | Yes | Selected | 0.002236 | 0.0345 |
| 420 | FrameDeltaDeltaMFCC_Hist1 | 0.1122 | No | Redundant | - | - |
| 421 | FrameDeltaDeltaMFCC_Hist2 | 0.0436 | No | RF pruned | - | - |
| 422 | FrameDeltaDeltaMFCC_Hist3 | 0.0000 | No | Constant | - | - |
| 423 | FrameDeltaDeltaMFCC_Hist4 | 0.0000 | No | Constant | - | - |
| 424 | FrameDeltaDeltaMFCC_Hist5 | 0.0000 | No | Constant | - | - |
| 425 | FrameDeltaDeltaMFCC_Hist6 | 0.0000 | No | Constant | - | - |
| 426 | FrameDeltaDeltaMFCC_Hist7 | 0.0000 | No | Constant | - | - |
| 427 | FrameDeltaDeltaMFCC_Hist8 | 0.0000 | No | Constant | - | - |
| 428 | FrameDeltaDeltaMFCC_Hist9 | 0.0000 | No | Constant | - | - |
| 429 | FrameDeltaDeltaMFCC_Mean | 0.0750 | No | Redundant | - | - |
| 430 | FrameDeltaDeltaMFCC_Std | 0.0927 | Yes | Selected | 0.002964 | 0.0622 |
| 431 | FrameDeltaDeltaMFCC_Skew | 0.0929 | No | Redundant | - | - |
| 432 | FrameDeltaDeltaMFCC_Kurt | 0.1055 | Yes | Selected | 0.002667 | 0.0556 |
| 433 | FrameDeltaDeltaMFCC_Median | 0.0496 | No | Redundant | - | - |
| 434 | FrameDeltaDeltaMFCC_IQR | 0.0859 | Yes | Selected | 0.003578 | 0.0626 |
| 435 | FrameDeltaDeltaMFCC_P5 | 0.0142 | Yes | Selected | 0.002151 | 0.0608 |
| 436 | FrameDeltaDeltaMFCC_P95 | 0.0905 | No | Redundant | - | - |
| 437 | FrameDeltaDeltaMFCC_Range | 0.1556 | Yes | Selected | 0.004231 | 0.0602 |
| 438 | GlobalDuration | 0.4328 | No | Redundant | - | - |
| 439 | GlobalVADRatio | 0.1017 | Yes | Selected | 0.001186 | 0.0215 |
| 440 | GlobalEnergy | 0.1688 | No | Redundant | - | - |
| 441 | GlobalClipping | 0.0065 | No | RF pruned | - | - |
| 442 | GlobalBandwidth | 0.0133 | Yes | Selected | 0.001307 | 0.0498 |
| 443 | GlobalReverb | 0.0432 | Yes | Selected | 0.001860 | 0.0580 |
| 444 | RT60_Est | 0.2380 | Yes | Selected | 0.019049 | 0.0315 |
| 445 | C50_Est | 0.1900 | Yes | Selected | 0.010135 | 0.0646 |
| 446 | ModulationDepth | 0.1632 | Yes | Selected | 0.004591 | 0.0336 |
| 447 | SpectralCentroid | 0.0252 | Yes | Selected | 0.001057 | 0.0512 |
| 448 | GlobalZCR | 0.0664 | No | Redundant | - | - |
| 449 | Jitter | 0.0845 | No | Redundant | - | - |
| 450 | Shimmer | 0.0260 | Yes | Selected | 0.003409 | 0.0464 |
| 451 | JitterPPQ5 | 0.0845 | Yes | Selected | 0.002025 | 0.0500 |
| 452 | JitterRAP | 0.0748 | No | Redundant | - | - |
| 453 | ShimmerAPQ3 | 0.0455 | Yes | Selected | 0.002588 | 0.0473 |
| 454 | ShimmerAPQ5 | 0.0293 | No | Redundant | - | - |
| 455 | ShimmerAPQ11 | 0.0396 | Yes | Selected | 0.003260 | 0.0453 |
| 456 | SpeechRate | 0.0267 | Yes | Selected | 0.001786 | 0.0609 |
| 457 | SpeakerTurns | 0.4373 | Yes | Selected | 0.068313 | 0.0509 |
| 458 | SegmentalSNR | 0.1767 | Yes | Selected | 0.005314 | 0.0610 |
| 459 | WADA_SNR | 0.1438 | Yes | Selected | 0.004010 | 0.0598 |
| 460 | NoiseFloorLevel | 0.1187 | No | Redundant | - | - |
| 461 | NoiseBandwidth | 0.0677 | Yes | Selected | 0.000911 | 0.0208 |
| 462 | NoiseStationarity | 0.0809 | Yes | Selected | 0.001325 | 0.0588 |
| 463 | LTAS_Slope | 0.0545 | Yes | Selected | 0.001707 | 0.0600 |
| 464 | LTAS_Tilt | 0.0275 | No | Redundant | - | - |
| 465 | SpectralFluxMean | 0.0457 | No | Redundant | - | - |
| 466 | SpectralFluxStd | 0.0561 | Yes | Selected | 0.003433 | 0.0630 |
| 467 | SpectralRolloff | 0.0658 | No | Redundant | - | - |
| 468 | SpectralEntropy | 0.0134 | No | Redundant | - | - |
| 469 | SpectralSkewness | 0.0427 | Yes | Selected | 0.001608 | 0.0616 |
| 470 | SpectralKurtosis | 0.0193 | Yes | Selected | 0.001498 | 0.0612 |
| 471 | SpectralCrest | 0.0107 | Yes | Selected | 0.001683 | 0.0581 |
| 472 | CPP_Mean | 0.0265 | No | Redundant | - | - |
| 473 | CPP_Std | 0.0155 | Yes | Selected | 0.001648 | 0.0620 |
| 474 | NHR | 0.0265 | Yes | Selected | 0.001418 | 0.0585 |
| 475 | H1H2 | 0.0598 | Yes | Selected | 0.002088 | 0.0580 |
| 476 | H1A3 | 0.0568 | Yes | Selected | 0.002321 | 0.0613 |
| 477 | UnvoicedFrameRatio | 0.0346 | Yes | Selected | 0.001513 | 0.0482 |
| 478 | EnergyRange | 0.0756 | No | Redundant | - | - |
| 479 | EnergyContourVariance | 0.1287 | No | Redundant | - | - |
| 480 | PauseDurationMean | 0.1184 | Yes | Selected | 0.001854 | 0.0547 |
| 481 | PauseRate | 0.0848 | Yes | Selected | 0.001417 | 0.0568 |
| 482 | LongestPause | 0.1220 | Yes | Selected | 0.002233 | 0.0533 |
| 483 | SpeechContinuity | 0.1984 | Yes | Selected | 0.032193 | 0.0520 |
| 484 | OnsetStrengthMean | 0.0759 | Yes | Selected | 0.003554 | 0.0596 |
| 485 | OnsetStrengthStd | 0.1025 | Yes | Selected | 0.002120 | 0.0591 |
| 486 | ClickRate | 0.0000 | No | Constant | - | - |
| 487 | DropoutRate | 0.0522 | Yes | Selected | 0.001392 | 0.0579 |
| 488 | SaturationRatio | 0.0306 | No | RF pruned | - | - |
| 489 | MusicalNoiseLevel | 0.0477 | Yes | Selected | 0.001470 | 0.0489 |
| 490 | QuantizationNoise | 0.0658 | Yes | Selected | 0.002572 | 0.0661 |
| 491 | DCOffset | 0.1511 | Yes | Selected | 0.008429 | 0.0657 |
| 492 | PowerLineHum | 0.0840 | Yes | Selected | 0.002170 | 0.0542 |
| 493 | AGC_Activity | 0.1520 | Yes | Selected | 0.003664 | 0.0528 |
| 494 | SubbandSNR_Low | 0.1232 | Yes | Selected | 0.002219 | 0.0185 |
| 495 | SubbandSNR_Mid | 0.1554 | Yes | Selected | 0.003735 | 0.0149 |
| 496 | SubbandSNR_High | 0.1406 | No | Redundant | - | - |
| 497 | LowToHighEnergyRatio | 0.0182 | Yes | Selected | 0.001300 | 0.0618 |
| 498 | LPCResidualEnergy | 0.0314 | Yes | Selected | 0.001010 | 0.0555 |
| 499 | VocalTractRegularity | 0.0495 | Yes | Selected | 0.001932 | 0.0567 |
| 500 | InterruptionCount | 0.1599 | Yes | Selected | 0.003611 | 0.0538 |
| 501 | SII_Estimate | 0.0383 | No | RF pruned | - | - |
| 502 | ModulationSpectrumArea | 0.0261 | Yes | Selected | 0.001836 | 0.0578 |
| 503 | NAQ | 0.0107 | Yes | Selected | 0.001543 | 0.0515 |
| 504 | QOQ | 0.0067 | Yes | Selected | 0.000699 | 0.0178 |
| 505 | HRF | 0.0128 | Yes | Selected | 0.001470 | 0.0395 |
| 506 | PSP | 0.0134 | Yes | Selected | 0.001715 | 0.0476 |
| 507 | GCI_Rate | 0.0114 | Yes | Selected | 0.001494 | 0.0513 |
| 508 | GOI_Regularity | 0.0760 | Yes | Selected | 0.001737 | 0.0500 |
| 509 | MDVP_Fo | 0.0112 | No | Redundant | - | - |
| 510 | MDVP_Jitter | 0.0845 | Yes | Selected | 0.001922 | 0.0534 |
| 511 | MDVP_Shimmer | 0.0250 | No | Redundant | - | - |
| 512 | MDVP_NHR | 0.0265 | No | Redundant | - | - |
| 513 | MDVP_VTI | 0.0694 | Yes | Selected | 0.001527 | 0.0526 |
| 514 | MDVP_SPI | 0.0311 | Yes | Selected | 0.001437 | 0.0611 |
| 515 | MDVP_DVB | 0.0352 | Yes | Selected | 0.001544 | 0.0524 |
| 516 | Tremor_Freq | 0.0705 | Yes | Selected | 0.001875 | 0.0541 |
| 517 | Tremor_Intensity | 0.1838 | Yes | Selected | 0.007064 | 0.0588 |
| 518 | Tremor_CycleVariation | 0.1173 | Yes | Selected | 0.001802 | 0.0548 |
| 519 | Tremor_Regularity | 0.0336 | Yes | Selected | 0.001924 | 0.0558 |
| 520 | F1_Mean | 0.0139 | Yes | Selected | 0.001500 | 0.0493 |
| 521 | F2_Mean | 0.0511 | Yes | Selected | 0.001623 | 0.0527 |
| 522 | F3_Mean | 0.0654 | Yes | Selected | 0.002080 | 0.0537 |
| 523 | FormantDispersion | 0.0415 | Yes | Selected | 0.002302 | 0.0579 |
| 524 | F1_BW | 0.0423 | Yes | Selected | 0.001752 | 0.0510 |
| 525 | F2_BW | 0.0083 | Yes | Selected | 0.001762 | 0.0564 |
| 526 | F3_BW | 0.0079 | Yes | Selected | 0.002241 | 0.0544 |
| 527 | AlphaRatio | 0.0377 | Yes | Selected | 0.001691 | 0.0628 |
| 528 | HammarbergIndex | 0.0359 | Yes | Selected | 0.001733 | 0.0598 |
| 529 | SpectralSlope0500_1500 | 0.0188 | Yes | Selected | 0.001820 | 0.0518 |
| 530 | MeanF0 | 0.0112 | No | Redundant | - | - |
| 531 | F0_StdDev | 0.0555 | Yes | Selected | 0.001707 | 0.0489 |
| 532 | FrequencyWeightedSNR | 0.0608 | Yes | Selected | 0.001146 | 0.0523 |
| 533 | SRMR | 0.0922 | Yes | Selected | 0.002832 | 0.0617 |
| 534 | DNSMOS_SIG | 0.0000 | No | Constant | - | - |
| 535 | DNSMOS_BAK | 0.0000 | No | Constant | - | - |
| 536 | DNSMOS_OVRL | 0.0000 | No | Constant | - | - |
| 537 | NISQA_MOS | 0.0000 | No | Constant | - | - |
| 538 | NISQA_NOI | 0.0000 | No | Constant | - | - |
| 539 | NISQA_DIS | 0.0000 | No | Constant | - | - |
| 540 | NISQA_COL | 0.0000 | No | Constant | - | - |
| 541 | NISQA_LOUD | 0.0000 | No | Constant | - | - |
| 542 | AVQI | 0.0269 | Yes | Selected | 0.001399 | 0.0586 |
| 543 | DSI | 0.2879 | Yes | Selected | 0.020647 | 0.0487 |
| 544 | CSID | 0.0200 | No | Redundant | - | - |

## Redundancy Pairs Removed

| Removed | Kept | Pearson r |
|---------|------|-----------|
| SpectralRolloff | FrameSR_Mean | 1.0000 |
| EnergyRange | FrameE_Range | 1.0000 |
| GlobalZCR | FrameZCR_Mean | 1.0000 |
| MDVP_NHR | NHR | 1.0000 |
| SpectralEntropy | FrameSE_Mean | 1.0000 |
| MeanF0 | MDVP_Fo | 1.0000 |
| Jitter | MDVP_Jitter | 1.0000 |
| FrameCPP_Hist1 | FrameCPP_Hist0 | 1.0000 |
| SpectralFluxMean | FrameSFlux_Mean | 1.0000 |
| FrameSFlux_Std | SpectralFluxStd | 0.9999 |
| CPP_Mean | AVQI | 0.9998 |
| FrameDeltaDeltaMFCC_Hist1 | FrameDeltaDeltaMFCC_Hist0 | 0.9998 |
| FrameE_IQR | FrameSNR_IQR | 0.9985 |
| FrameDeltaMFCC_Hist1 | FrameDeltaMFCC_Hist0 | 0.9973 |
| GlobalDuration | SpeakerTurns | 0.9973 |
| MDVP_Fo | GCI_Rate | 0.9965 |
| CSID | AVQI | 0.9964 |
| FrameCPP_Std | FrameCPP_P95 | 0.9948 |
| FrameF0_Hist0 | UnvoicedFrameRatio | 0.9934 |
| FrameE_Std | FrameSNR_Std | 0.9922 |
| LTAS_Tilt | AlphaRatio | 0.9907 |
| FrameSC_Median | FrameSC_Mean | 0.9883 |
| FrameSS_Median | FrameSS_Mean | 0.9861 |
| FramePC_P95 | FramePC_Std | 0.9861 |
| FrameSR_Mean | FrameZCR_Mean | 0.9858 |
| NoiseFloorLevel | SegmentalSNR | 0.9846 |
| FrameSNR_Median | FrameSNR_Mean | 0.9844 |
| MDVP_Shimmer | Shimmer | 0.9841 |
| FrameMFCCVar_Median | FrameMFCCVar_Mean | 0.9838 |
| FrameDeltaMFCC_P95 | FrameDeltaMFCC_Std | 0.9838 |
| EnergyContourVariance | FrameSNR_Std | 0.9806 |
| FrameSCF_Median | FrameSCF_Mean | 0.9804 |
| FrameSF_Std | FrameSF_P95 | 0.9795 |
| FramePC_IQR | FramePC_Mean | 0.9773 |
| FrameSFlux_Mean | FrameSFlux_Median | 0.9757 |
| FrameSCF_Hist9 | FrameSCF_Mean | 0.9737 |
| FrameSBW_Median | FrameSBW_Mean | 0.9732 |
| FrameDeltaDeltaMFCC_P95 | FrameDeltaDeltaMFCC_Std | 0.9728 |
| FrameSC_Std | FrameSC_IQR | 0.9727 |
| SubbandSNR_High | SubbandSNR_Mid | 0.9725 |
| FrameAC_Median | FrameAC_Mean | 0.9696 |
| FramePC_Hist0 | FramePC_Mean | 0.9684 |
| FrameGNE_Std | FrameGNE_P5 | 0.9674 |
| FrameCPP_Mean | FrameCPP_IQR | 0.9674 |
| FrameDeltaDeltaMFCC_Mean | FrameDeltaDeltaMFCC_IQR | 0.9673 |
| FrameE_Median | GlobalEnergy | 0.9669 |
| FrameSHR_Mean | FrameSHR_Median | 0.9647 |
| FrameHNR_P95 | FrameHNR_Std | 0.9640 |
| FrameZCR_P95 | FrameZCR_Std | 0.9628 |
| FrameSSkew_Hist9 | FrameSKurt_Hist9 | 0.9610 |
| FrameDeltaDeltaMFCC_Skew | FrameDeltaDeltaMFCC_Kurt | 0.9604 |
| ShimmerAPQ5 | ShimmerAPQ3 | 0.9571 |
| FrameSE_Mean | FrameSE_Median | 0.9567 |
| FrameSSkew_Kurt | FrameSSkew_Skew | 0.9558 |
| GlobalEnergy | FrameE_Mean | 0.9550 |
| FrameDeltaMFCC_IQR | FrameDeltaMFCC_Mean | 0.9547 |
| JitterRAP | JitterPPQ5 | 0.9544 |
| FrameSNR_Mean | FrameSNR_P95 | 0.9544 |
| FrameSR_Std | FrameZCR_Std | 0.9540 |
| FrameSC_Hist2 | FrameSC_Mean | 0.9532 |
| FrameHNR_Std | AVQI | 0.9529 |
| FrameMFCCVar_IQR | FrameMFCCVar_Std | 0.9514 |
| FrameDeltaDeltaMFCC_Median | FrameDeltaMFCC_Median | 0.9510 |
| FrameSFlux_IQR | SpectralFluxStd | 0.9500 |

## Top 10 by Spearman |rho|

| Feature | |rho| |
|---------|-------|
| SpeakerTurns | 0.4373 |
| GlobalDuration | 0.4328 |
| FrameSFlux_Hist0 | 0.4064 |
| DSI | 0.2879 |
| RT60_Est | 0.2380 |
| FrameSNR_Range | 0.2009 |
| SpeechContinuity | 0.1984 |
| FrameE_P5 | 0.1942 |
| FrameDeltaMFCC_Range | 0.1910 |
| C50_Est | 0.1900 |

## Top 10 by RF Importance

| Feature | Importance |
|---------|------------|
| SpeakerTurns | 0.068313 |
| FrameSFlux_Hist0 | 0.067423 |
| SpeechContinuity | 0.032193 |
| DSI | 0.020647 |
| RT60_Est | 0.019049 |
| FrameSNR_Hist0 | 0.010194 |
| C50_Est | 0.010135 |
| FrameSC_Hist1 | 0.009348 |
| DCOffset | 0.008429 |
| Tremor_Intensity | 0.007064 |

## Top 10 by ERC AUC (best quality predictors)

| Feature | ERC AUC |
|---------|---------|
| SubbandSNR_Mid | 0.0149 |
| QOQ | 0.0178 |
| SubbandSNR_Low | 0.0185 |
| NoiseBandwidth | 0.0208 |
| GlobalVADRatio | 0.0215 |
| RT60_Est | 0.0315 |
| ModulationDepth | 0.0336 |
| FrameDeltaDeltaMFCC_Hist0 | 0.0345 |
| HRF | 0.0395 |
| FrameDeltaMFCC_Hist0 | 0.0432 |