# U5.R2AO4M FILM2PAINT-style curve/matrix capacity result

Date: 2026-07-29

Decision: **automatic failure; close this family on the 71-pair pool**

Claim ceiling: equal-parameter clean-room display-proxy capacity evidence only.

## Experiment

AO4M clean-roomed only the simplest operator order described by FILM2PAINT:
three endpoint-fixed monotone four-knot channel curves followed by one bounded
positive matrix. It did not use the unavailable FILM2PAINT data, source code,
knots, parameters or Adobe-RGB pipeline.

The candidate and retained bounded one-matrix control each have 12 free
parameters. Five folds hold out rows inside both the 24-pair chart and 47-pair
palette domains; two additional fits leave out each complete domain. All
parameters, mixtures, bounds, folds, gates and restarts were frozen before the
implementation result was observed.

Two 63,979-byte formal reports are byte-identical at
`62e24b682c0263ec255ff560b422f602df685b5a3bb676e7e8493ab92b48bc3b`.

## Result

| Measure | Bounded one-matrix | Curve then matrix | Candidate gain |
|---|---:|---:|---:|
| Combined RGB RMSE | 0.033515 | 0.033128 | +1.15% |
| Combined mean Delta E76 | 6.6013 | 9.2814 | -40.60% |
| Chart RGB RMSE | 0.049404 | 0.049820 | -0.84% |
| Palette RGB RMSE | 0.021225 | 0.019761 | +6.90% |

The candidate has an interesting cross-domain signal:

- chart fit to palette confirmation improves RGB RMSE by 25.01%;
- palette fit to chart confirmation improves RGB RMSE by 9.40%.

That signal is not sufficient for promotion. The frozen combined RGB gain is
only 1.15% versus a 5% minimum, the chart held-row domain regresses, and mean
Delta E76 becomes 40.60% worse. The three corresponding automatic gates fail.

## Failure attribution

Every structural gate passes:

- zero raw out-of-cube fraction;
- minimum sampled Jacobian determinant 0.01972;
- endpoint error at most `2.22e-16`;
- analytic inverse roundtrip error at most `3.61e-16`;
- train/confirmation RGB gap 0.00402;
- exact source nonmutation and two-run replay.

The failure is therefore not clipping, gamut escape, folding, inversion or
optimizer instability. This safe operator order places its limited capacity
in a way that helps whole-domain transfer but hurts held-row perceptual
accuracy, especially on the chart domain.

## Branch decision

- Retain the existing bounded one-matrix global proxy operator.
- Do not render photographs, change metric weights, move knots, relax the
  five-percent gate or add capacity to rescue AO4M.
- Do not claim a FILM2PAINT reproduction, Velvia stock response, calibration
  or product evidence.
- Preserve the curve/matrix implementation as a tested negative baseline.

The next distinct algorithm leaf is AN2: preregister a deterministic two-stage
sparse-outlier rejection challenger on the existing synthetic paired-recovery
witnesses. It must pass there before any display-proxy or photographic use.
