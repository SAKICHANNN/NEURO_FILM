# U5.R2AO1 Velvia chart operator explainability

Two formal reports are byte-identical at SHA-256 `38b355f9...930b`.
All frozen gates pass across six leave-one-chart-row-out folds.

| Model | Mean held-row RGB RMSE | Mean held-row Delta E76 |
|---|---:|---:|
| Identity | 0.16022 | 21.40 |
| Per-channel affine | 0.09791 | 17.45 |
| Full affine | 0.08249 | 14.41 |
| One matrix + response curves | **0.04404** | **8.18** |
| Two matrices + response curves | 0.04497 | 9.78 |

The one-matrix model reduces RMSE by 72.51% against identity and 46.62%
against full affine. The extra scan matrix lowers training error but does not
improve held-row error, so the simpler model is retained for AO2.

This is development explainability for 24 patches from one author-rendered
sRGB figure, not independent confirmation, RAW/stock response identification,
calibration or product promotion.
