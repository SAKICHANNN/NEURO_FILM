# U5.R2S1C Canonical-Histogram Score Confirmation Results

Date: 2026-07-26

Decision: **all frozen gates pass; retain the synthetic KDE-score mechanism**

## Reproducibility

- software commit:
  `d57f6b2186e41bd3437f62411e777d4564c03e65`;
- config SHA-256:
  `cab8930c9676b597ce2e18588d9ebe64a64cf97e0c115049a4f8f5aaef2a0c91`;
- both reports are byte-identical at
  `2886c049caafb270151d1bd8de2109dafc82919a18d18a1620e8accfa8a13739`;
- the untouched confirmation uses all 128 palettes from seed `27012`;
- seven combined S1 focused tests pass.

## Confirmation result

| Measure | Frozen gate | Result |
|---|---:|---:|
| Median operator RMSE | <= .040 | .03046 |
| p90 operator RMSE | <= .065 | .05029 |
| Median direction cosine | >= .90 | .9593 |
| Median style retention | .80-1.20 | .9845 |
| Median non-affine retention | >= .75 | .9092 |
| Median palette log-density gain | >= 1.5 | 2.7308 |
| Median reference separation | >= .80 | .9339 |
| Median error reduction vs hard Top-1 | >= 45% | 63.46% |
| Median error reduction vs global mean | >= 50% | 71.88% |
| Minimum sampled determinant | > .005 | .01500 |
| Maximum sampled Jacobian norm | <= 8 | 3.993 |
| Worst inverse error | <= 1e-5 | 7.35e-7 |
| Histogram/operator permutation error | 0 | 0 / 0 |
| Repeat report | exact | pass |

Every gate passes.

## Meaning

If a target palette density is available as canonical colour samples, a fixed
KDE can estimate its score well enough to drive a strong, non-affine,
reference-sensitive and structurally safe explicit RGB flow. A neural network
is unnecessary for this subproblem.

Hard case reuse is not competitive in this formulation: its confirmation
median operator error is `.08335`, versus `.03046` for KDE. The global mean
reaches `.10829` and has zero reference separation. This is direct evidence
against solving style diversity by averaging all references into one model.

## Why images still do not open

The synthetic query histogram directly samples the latent target palette
density. A real film photograph instead combines scene content and object
colours, illumination and exposure, film/process/scan interpretation and
source-specific nuisance.

Applying this mechanism directly to arbitrary film scans would therefore
mostly learn scene palette, exactly the shortcut observed in the project's
stock-identifiability audits. The passed score estimator does not solve
unpaired digital-to-film operator identification.

The next legal leaf is a synthetic nuisance-identifiability experiment. It
must independently vary content and style, compare raw KDE against
distribution-matched density-ratio controls, and allow a learned model to
predict only the bounded explicit velocity grid. No project photograph or
film pixel may open yet.
