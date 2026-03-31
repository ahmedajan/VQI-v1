# X1.12 Comprehensive Comparison — VQI-S

**Total models evaluated:** 28
**RF baseline AUC:** 0.8719
**RF baseline ERC@20%:** 8.9%

## Top 10 by AUC-ROC

| Rank | Model | AUC-ROC | F1 | Brier | ms/sample | Mean ERC@20% |
|------|-------|---------|----|----|-----------|-------------|
| 1 | SVM (Clf, 20K) | 0.9371 | 0.8979 | 0.1015 | 2.61 | 2.0% |
| 2 | SVM (Reg, 20K) | 0.9350 | 0.9160 | 0.1201 | 2.75 | 3.1% |
| 3 | SVM (Clf, 58K) | 0.9298 | 0.8878 | 0.1092 | 4.09 | 4.4% |
| 4 | MLP (Clf, 20K) | 0.9281 | 0.8718 | 0.1326 | 0.03 | 9.6% |
| 5 | SVM (Reg, 58K) | 0.9257 | 0.9088 | 0.1222 | 5.01 | 5.8% |
| 6 | LightGBM (Clf, 20K) | 0.9215 | 0.8769 | 0.1265 | 0.11 | 8.9% |
| 7 | MLP (Reg, 20K) | 0.9204 | 0.9040 | 0.1109 | 0.03 | 18.9% |
| 8 | MLP (Clf, 58K) | 0.9189 | 0.8924 | 0.1133 | 0.03 | 13.0% |
| 9 | XGBoost (Clf, 58K) | 0.9170 | 0.8839 | 0.1243 | 0.04 | 7.8% |
| 10 | XGBoost (Clf, 20K) | 0.9141 | 0.8724 | 0.1313 | 0.05 | 2.3% |

## Candidates Passing All Criteria

**Candidates:** 6

| Model | AUC-ROC | Mean ERC@20% | Speed (ms) |
|-------|---------|-------------|-----------|
| Ridge (Reg, 20K) | 0.8803 | 20.2% | 0.01 |
| MLP (Reg, 20K) | 0.9204 | 18.9% | 0.03 |
| LogReg (Clf, 20K) | 0.8816 | 12.9% | 0.01 |
| MLP (Clf, 20K) | 0.9281 | 9.6% | 0.03 |
| XGBoost (Reg, 20K) | 0.9101 | 8.6% | 0.04 |
| TabNet (Clf, 20K) | 0.8950 | 7.5% | 0.93 |

## RECOMMENDATION: **Ridge (Reg, 20K)**

- AUC-ROC: 0.8803 (RF baseline: 0.8719)
- Mean ERC@20%: 20.2% (RF baseline: 8.9%)
- Inference: 0.01 ms/sample
- Score range: [0-100]