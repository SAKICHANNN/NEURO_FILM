# Roll2Film CT5 FilmSet internal pilot

> Date: 2026-07-15
> Node: `ULT > U5.CT5`
> State: pilot-only; no candidate promoted

## Execution boundary

The baseline/evaluator policy and implementation were committed before the
internal pilot target fold was loaded. Operators were fitted from 2,096
source-only identities and 2,096 target-only identities per recipe, with 512
ICC-normalized linear-sRGB pixels per image and equal image weighting. Pilot
evaluation used 227 separate identities with 2,048 aligned pixels per image.

The 238-identity confirmatory arrays were not loaded by the pilot runner. The
official 628 manifest was hash-checked as a sealed file but never parsed or
decoded. FilmSet remains Capture One recipe evidence, not physical film truth.

## Pilot recipe fidelity and strength

Values below are sampled mean Delta-E 2000 to the hidden recipe target and
median Delta-E 2000 from input, averaged over images. Lower target Delta-E is
better; higher input Delta-E means a stronger visible colour displacement.

### Cinema

| Candidate | Target Delta-E | Style strength | Raw out-of-range |
|---|---:|---:|---:|
| best joint basic | 1.908 | 3.316 | 0.000% |
| Lab mean/std | **1.730** | 3.394 | 0.000% |
| Gaussian/Bures | 2.052 | 3.697 | 0.000% |
| pooled L2 | 2.168 | 3.686 | 0.515% |
| sliced OT | 2.216 | 3.904 | 0.725% |
| paired per-image affine oracle | 1.444 | 3.787 | 0.104% |

### ClassNeg

| Candidate | Target Delta-E | Style strength | Raw out-of-range |
|---|---:|---:|---:|
| best joint basic in strong stratum | 4.993 | 3.595 | 10.237% |
| Lab mean/std | 4.190 | 2.931 | 0.000% |
| Gaussian/Bures | 4.272 | 4.298 | 10.065% |
| sliced OT | 3.788 | 4.855 | 1.474% |
| pooled L2 | **2.106** | 4.912 | 1.808% |
| paired per-image affine oracle | 3.604 | 5.528 | 11.708% |

The pooled L2 result beating the affine oracle is not paradoxical: the oracle
is deliberately per-image affine, while pooled L2 contains monotone nonlinear
curves. It is an affine-family paired reference, not an unrestricted ceiling.

### Velvia

| Candidate | Target Delta-E | Style strength | Raw out-of-range |
|---|---:|---:|---:|
| best joint basic in strong stratum | 3.770 | 2.907 | 11.037% |
| Lab mean/std | 3.542 | 2.298 | 0.000% |
| per-channel quantile | **3.121** | 2.957 | 0.553% |
| sliced OT | 3.387 | 3.707 | 7.777% |
| pooled L2 | 3.425 | 4.603 | 1.003% |
| paired per-image affine oracle | 3.087 | 3.493 | 6.412% |

## Pilot-derived freeze

The primary style metric is divided into three strata before confirmatory use:

- low: `[0, 1.5)`;
- moderate: `[1.5, 2.75)`;
- strong: `[2.75, infinity)`.

Within each domain/stratum, the eligible basic candidate with the lowest pilot
target Delta-E is frozen in
`configs/roll2film_ct5_pilot_decision.json`. Confirmatory comparisons use the
paired per-image improvement over that fixed adversary and bootstrap duplicate
clusters. No candidate strength, parameter, stratum or best-basic identity may
change after confirmatory results are observed.

## Interpretation

The pilot rejects a simple story in which one theoretically superior method
wins everywhere:

- pooled L2 shows a large nonlinear advantage on ClassNeg;
- Lab mean/std is strongest on Cinema among deployable pilot baselines;
- per-channel quantile is strongest on Velvia;
- pooled L2 and sliced OT have nonzero off-range risk, while several basic or
  affine methods have much larger ClassNeg/Velvia excursions;
- recipe fidelity does not establish film style, appeal or severe-artifact
  safety.

This pattern makes a fixed per-recipe champion bank plausible, but does not yet
justify per-photo ML routing. Confirmatory uncertainty, full-resolution output,
visual severe adjudication and best-basic residuals remain mandatory.

## Reproducibility

- policy config SHA-256:
  `b85e1a58cb07ce4138fb45a514051630700cfbc949c1798ca2024a97c50090c1`;
- implementation config SHA-256:
  `d06f77776db2d56870f56bc27f2f127fef122a6233f84e7e9d9dcb1bac4c3f90`;
- data-cache report SHA-256:
  `fc3068e02e4a90a262c33901bda49e7eab726c5a099bd9906ff64fda2f8efad9`;
- pilot software commit: `9843bb2e892f0a7badf9fbc4088b6e15b756bb78`;
- pilot report SHA-256:
  `145a3bca313212d625a0245b6366d77ceb5732e883d8f87255bab53aeaa434fa`;
- frozen pilot-decision SHA-256:
  `0d80f9ac3e372b5c9820a8c47cec7436396adb90e0195454745f3a3daf53202b`;
- a complete rerun produced the same pilot report byte for byte;
- 74 repository tests pass after the quantized-knot failure shield.

Generated caches, fitted bundles and per-image results remain ignored and
internal-only.
