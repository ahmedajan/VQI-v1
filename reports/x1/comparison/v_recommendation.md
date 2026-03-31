# X1.12 Comprehensive Comparison — VQI-V

**Total models evaluated:** 28
**RF baseline AUC:** 0.8812
**RF baseline ERC@20%:** 7.4%

## Top 10 by AUC-ROC

| Rank | Model | AUC-ROC | F1 | Brier | ms/sample | Mean ERC@20% |
|------|-------|---------|----|----|-----------|-------------|
| 1 | SVM (Reg, 20K) | 0.9469 | 0.9237 | 0.1169 | 1.43 | -1.3% |
| 2 | SVM (Clf, 20K) | 0.9436 | 0.9092 | 0.0939 | 1.27 | -3.1% |
| 3 | SVM (Clf, 58K) | 0.9412 | 0.9041 | 0.1002 | 1.68 | 5.1% |
| 4 | SVM (Reg, 58K) | 0.9385 | 0.9119 | 0.1174 | 2.80 | 10.2% |
| 5 | MLP (Reg, 20K) | 0.9373 | 0.9188 | 0.1055 | 0.02 | 11.6% |
| 6 | XGBoost (Clf, 20K) | 0.9344 | 0.8960 | 0.1085 | 0.04 | 4.1% |
| 7 | MLP (Reg, 58K) | 0.9340 | 0.9135 | 0.1055 | 0.02 | 16.8% |
| 8 | MLP (Clf, 20K) | 0.9337 | 0.9030 | 0.1063 | 0.03 | 5.4% |
| 9 | MLP (Clf, 58K) | 0.9289 | 0.9021 | 0.0999 | 0.03 | 11.7% |
| 10 | XGBoost (Clf, 58K) | 0.9279 | 0.8951 | 0.1117 | 0.04 | 5.8% |

## Candidates Passing All Criteria

**Candidates:** 7

| Model | AUC-ROC | Mean ERC@20% | Speed (ms) |
|-------|---------|-------------|-----------|
| XGBoost (Reg, 58K) | 0.9130 | 14.3% | 0.04 |
| SVM (Reg, 58K) | 0.9385 | 10.2% | 2.80 |
| RF (Clf, 20K) | 0.8812 | 7.4% | 7.01 |
| XGBoost (Clf, 58K) | 0.9279 | 5.8% | 0.04 |
| LightGBM (Clf, 20K) | 0.9277 | 5.1% | 0.07 |
| LightGBM (Reg, 58K) | 0.9183 | 4.7% | 0.07 |
| LightGBM (Reg, 20K) | 0.9251 | 4.0% | 0.07 |

## RECOMMENDATION: **XGBoost (Reg, 58K)**

- AUC-ROC: 0.9130 (RF baseline: 0.8812)
- Mean ERC@20%: 14.3% (RF baseline: 7.4%)
- Inference: 0.04 ms/sample
- Score range: [20-99]