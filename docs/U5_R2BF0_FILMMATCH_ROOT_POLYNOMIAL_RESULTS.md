# U5.R2BF0 — FilmMatch root-polynomial directionality

## Decision

Close the signed root-polynomial route on the frozen FilmMatch source.
Exposure homogeneity is numerically satisfied, but the selected degree-2
operator fails every other frozen forward-capacity gate and is not render
safe. No validation image, visual review, bounded-execution rescue or product
integration opens.

Two executions are byte-identical:

- report SHA-256: `c5d2f783118e7ceebbb2db1ca0ec3080b7e032506f56ad5a26911d89be22dd15`
- stable evidence ID: `8f1330432ee5d75a20bd93a1d499efcaba17d20ee9e00cef734217597aa5468b`

## Forward grouped evidence

Sony S-Log3 samples are decoded with the already frozen Sony formula before
fitting. Signed roots retain the small negative scene-linear black values.

| Model | Held illuminant median RMSE | vs full affine | Held EV median RMSE | vs full affine |
|---|---:|---:|---:|---:|
| full affine | 0.265724 | — | 0.273414 | — |
| ordinary polynomial degree 3 | 0.188141 | +29.20% | 0.219653 | +19.66% |
| signed root polynomial degree 2 | 0.374155 | -40.81% | 0.388203 | -41.98% |
| signed root polynomial degree 3 | 0.380495 | -43.19% | 0.387821 | -41.84% |

The selected signed-root degree-2 model has a worst held-EV regression of
9.04% against homogeneous affine and a 58.13% worst held-sample output
out-of-cube fraction. Its dense structural audit has a negative minimum
Jacobian (`-0.6338`) and 38.06% out-of-target-cube probes. Scale
equivariance is exact at the reported precision, so the failure is not an
implementation failure of the defining property.

The inverse film-scan-to-scene-linear diagnostic also selects degree 2 but
does not clear the affine comparisons. RPCC is therefore not supported even
as an inverse correction winner on this source.

## Post-result observation

The unconstrained ordinary degree-3 control has materially better grouped
capacity, but it is not a candidate: its minimum sampled Jacobian is
`-0.04797`, 26.30% of the dense domain leaves the target cube, and positive
scale equivariance error exceeds `1000`. This observation cannot reopen BF0
or authorize clipping, blending or post-result threshold changes. Earlier
FilmMatch AX3–AX17 work already tested bounded nonlinear/residual families and
closed the final safe degree-5 global challenger on blind preference.

## Claim ceiling

Development-only directionality and grouped-capacity evidence for explicit
polynomial operators on one author-prepared Sony FX3/Ektachrome E100 paired
chart source. This is not calibrated stock response, identified
digital-to-film recovery, scanner independence or product evidence.
