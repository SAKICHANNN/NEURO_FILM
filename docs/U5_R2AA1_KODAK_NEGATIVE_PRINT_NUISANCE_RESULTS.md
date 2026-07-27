# U5.R2AA1 Kodak Negative-to-Print Nuisance Results

Date: 2026-07-28

Node: `ULT > U5 > U5.R2 > U5.R2AA1`

Decision: **`nuisance_unidentified`; close this datasheet chain without
refitting, visual rendering or rescue**

## Result

The preregistered Kodak VISION3 250D to VISION 2383 synthetic chain produces a
very strong, non-basic colour direction, but that direction is not stable to
the physical variables absent from the public sheets.

| Gate family | Result | Frozen requirement | Outcome |
|---|---:|---:|---|
| nominal identity Delta E76, median | 54.4301 | >= 7.0 | pass |
| joint best-basic residual, median | 31.5805 | >= 4.9 | pass |
| per-channel affine residual, median | 20.6033 | >= 3.0 | pass |
| best-basic residual >=2 fraction | 100% | >=75% | pass |
| spectrum nuisance ratio, median / p95 | .7838 / 2.4622 | <=.25 / .50 | fail |
| placement nuisance ratio, median / p95 | .4078 / 1.9182 | <=.35 / .75 | fail |
| dye-mapping nuisance ratio, median / p95 | .2004 / .8720 | <=.25 / .50 | fail |
| printer nuisance ratio, median / p95 | .0715 / .2981 | <=.35 / .75 | pass |
| viewer nuisance ratio, median / p95 | .0640 / .3115 | <=.25 / .50 | pass |
| complete-ensemble ratio, median / p95 | 1.4324 / 4.4665 | <=.50 / 1.00 | fail |
| nominal / ensemble neutral chroma max | 65.7002 / 129.3609 | <=4 / 8 | fail |
| Jacobian determinant min / norm max | 6.69e-5 / 1.1277 | >0 / <=8 | pass |
| raw absolute max / outside-cube fraction | 1.1179 / 12.7739% | <=4 / <=25% | pass |

The full nuisance ensemble changes the output more than the nominal effect for
a typical synthetic colour. A large nominal style score cannot identify which
of those different directions corresponds to a real 250D-to-2383 process.

## Source and repeat evidence

- curve data SHA-256:
  `62e463c8e0f0784da1980b66b0b338a1a2a3f6ece5ffe34268d2bc1987c2187d`;
- config SHA-256:
  `016e237e62a1f8961c1c3e682614d58a497b87fd409a72ff64f46268bc00fe71`;
- software commit: `cfd1271598534120c14ee215f7560f8dfb382c8b`;
- report payload SHA-256:
  `8c2fbbc30e422c7bf5a4370c0c648010e88ec7b96948c397a1ea7f92a5903e6b`;
- stored report SHA-256:
  `c4161c01097dc6028e51352eb204319e55371e7f823bab884de7de249c906e75`;
- stored arrays SHA-256:
  `7569d7c0a531d7da3c0667309750f6077b771f9959f2d1e00bc2ab9021882c4b`;
- 729 colours, 33 neutral samples, 69 wavelengths and all 162 preregistered
  members;
- both complete evaluations and every retained array fingerprint are exact.

All six graph overlays were reviewed at source resolution. The automatic trace
gate masks known axes/grid lines and confirms every retained point lies on
remaining source ink. Base/metamer reconstruction, finite arithmetic,
metamer separation, neutral monotonicity, raw range and sampled Jacobian gates
pass.

## Pre-result factual repair

The first diagnostic execution was rejected before scientific propagation. It
revealed that H-61B and AA0 record `1.09/1.06/1.03` in R/G/B order while the
characteristic implementation consumes B/G/R rows. The implementation now
reverses that exact observed vector only at the interface and pins the
`1.03/1.06/1.09` B/G/R roundtrip in a regression test. Digitized
characteristics also use their physical monotone envelope and invert the same
PCHIP curve used by the forward path.

No source, trace point, nuisance member, nominal choice, control, threshold or
branch changed. The rejected diagnostic is not AA1 evidence.

## Interpretation and stop

AA1 is useful negative evidence:

> Public characteristic, sensitivity and dye-density graphs can define a
> strong explicit film-inspired transform, but they do not identify a stable
> negative-to-print colour operator when original spectra, exposure placement,
> analytical dye mapping, printer primaries and viewing conditions are
> missing.

The branch stops at `nuisance_unidentified`. Do not choose one nuisance member
on project photographs; fit, clamp, gamut-map or bake it into a LUT; render a
visual shortlist; train a selector; or claim a real Kodak response.

AA2 does not open. Ultimate remains active and continues through a distinct
evidence-authorized data, explicit-algorithm or product leaf.
