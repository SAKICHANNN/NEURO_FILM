# U5.R2AQ1 ColorReference Velvia 100F pair-table result

## Decision

AQ1 passes every preregistered geometry, measurement-integrity and repeated-set
identifiability gate. It opens only a separately frozen held-target-set
explicit-proxy baseline experiment (`U5.R2AQ2`).

It does not establish independent rolls/processes, a camera-to-film operator,
a calibrated Velvia response or a product profile.

## Exact evidence

- Config SHA-256:
  `3843dc8b81980efd7b5c4ddf179e383c0f4fdf59539365f04ff511696f02aea1`
- Software commit:
  `c16c459083d6da580ff671c430a490257a76c725`
- Two exact formal reports:
  `21177de2944d866079e8ac3c45862e6a83b397928a01366d3ded53bc52555e6c`
- Canonical 8,640-row TSV:
  `0bcb92eb5131b052705922caf747a467ed8a458c90bca2c14dbc66b800cdf595`

The frozen 12x22 colour grid plus 24-step grayscale row maps exactly to the
288 measured sample IDs. Every 3x3 source neighbourhood is constant. The five
source slides contain 285--288 unique RGB triplets and span code 0--255. IT8
and CGATS base fields match exactly; 41 spectral values cover 380--780 nm.

## Repeated-set result

Across six target sets, five slides and 288 patches:

| Metric | Result | Frozen gate |
|---|---:|---:|
| median patch/set radius | 0.5527 Delta E76 | <=4 |
| p95 patch/set radius | 1.2799 Delta E76 | <=10 |
| maximum radius | 2.8553 Delta E76 | report |
| fraction above 10 | 0% | <=5% |
| median within-slide distance-structure Spearman | 0.999898 | >=0.90 |
| minimum distance-structure Spearman | 0.999477 | report |
| between-set centroid variance / total variance | 0.00155% | <=2% |
| leave-one-set consensus RMSE | 0.8905 Delta E76 | <=6 |

Thus target-set nuisance is small relative to the manufactured chart
structure, and a complete-set holdout can test whether an explicit mapping
generalizes across these target sets.

## Epistemic boundary

The five inputs are unknown-profile film-recorder RGB grids, not camera scenes
or sRGB. The six target sets share declared production date 2005:05 and remain
unidentified physical sheets/rolls/process sessions. AQ2 may measure only a
recorder-device-to-measured-developed-slide proxy. It must fit on complete
development sets and score untouched target sets; no photo rendering or stock
claim is allowed.
