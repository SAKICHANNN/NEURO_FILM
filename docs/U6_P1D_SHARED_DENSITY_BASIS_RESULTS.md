# U6.P1D shared density-basis fidelity

Date: 2026-07-28

One unrouted rank-6 NMF basis was fit on all pre-2020 target charges and tested
on a new forward holdout: six Ektachrome-compatible and six
Velvia-50-compatible charges from 2020--2024.

| Metric | Median / p95 |
|---|---:|
| optical-density RMSE | .01582 / .01663 D |
| transmittance RMSE | .924 / 1.414 percentage points |
| synthetic scanner A RGB RMSE | -- / .00786 |
| synthetic scanner B RGB RMSE | -- / .00575 |

Rank 6 improves median density error 19.70% over rank 5. The frozen rank-3
negative control fails clearly at .14776 D p95 and .0671/.0774 scanner p95.
All fits converge; two reports are byte-identical at
`59454440...0f0e52`.

Retain rank 6 only as an internal generic offline reference primitive. No
basis components are committed or redistributed, and no family routing,
identified dye-layer, scene-to-density, stock-response or product claim opens.
