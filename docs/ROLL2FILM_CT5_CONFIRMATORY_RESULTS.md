# Roll2Film CT5 FilmSet confirmatory results

> Date: 2026-07-15
> Node: `ULT > U5.CT5`
> State: sampled statistical gate complete; full-resolution severe gate pending

## Frozen execution

The confirmatory evaluator replayed the exact operator bundles written by the
pilot. It did not refit parameters. Candidate strength strata and best-basic
adversaries came from the committed pilot decision. The untouched fold contains
238 identities; duplicate clusters, not pixels, are the bootstrap unit.

The official 628 remained unopened. Sampled SSIM and visual severe status are
explicitly `not-run` because the fixed pixel cache has no spatial adjacency.

## Confirmatory results

Positive confidence intervals mean lower target Delta-E than the frozen
strength-stratum best-basic. Style pass requires at least the best-basic's
measured input displacement.

### Cinema - strong best-basic: joint WB/contrast/saturation

| Candidate | Target Delta-E | Style | Improvement 95% CI | Statistical status | Raw OOR |
|---|---:|---:|---:|---|---:|
| Lab mean/std | **1.875** | 3.459 | [0.140, 0.208] | pass | 0.000% |
| Gaussian/Bures | 2.248 | 3.806 | [-0.249, -0.151] | fail fidelity | 0.000% |
| pooled L2 | 2.407 | 3.809 | [-0.498, -0.239] | fail fidelity | 0.743% |
| sliced OT | 2.411 | 3.959 | [-0.440, -0.279] | fail fidelity | 0.833% |
| per-channel quantile | 3.034 | 3.133 | [-1.057, -0.918] | fail fidelity/style | 0.000% |

The basic target Delta-E is `2.049`; the paired per-image affine reference is
`1.521`. Cinema therefore rejects pooled L2 despite its stronger displacement.

### ClassNeg - strong best-basic: joint WB/contrast/saturation

| Candidate | Target Delta-E | Style | Improvement 95% CI | Statistical status | Raw OOR |
|---|---:|---:|---:|---|---:|
| pooled L2 | **2.321** | 4.833 | [2.709, 2.928] | pass | 2.286% |
| sliced OT | 4.039 | 4.718 | [0.981, 1.219] | pass | 1.865% |
| Gaussian/Bures | 4.489 | 4.444 | [0.535, 0.756] | pass | 11.364% |
| Lab mean/std | 4.313 | 2.931 | [0.730, 0.923] | fail style | 0.000% |
| per-channel quantile | 4.577 | 3.299 | [0.429, 0.695] | fail style | 0.000% |

The basic target Delta-E is `5.136`; the paired affine reference is `3.552`.
Pooled L2 closes 178% of that affine-family gap because its monotone nonlinear
capacity can outperform a per-image affine fit. This is not an unrestricted
paired upper bound.

### Velvia - strong best-basic except moderate Lab comparison

| Candidate | Target Delta-E | Style | Improvement 95% CI | Statistical status | Raw OOR |
|---|---:|---:|---:|---|---:|
| pooled L2 | **3.456** | 4.407 | [0.266, 0.574] | pass | 1.244% |
| sliced OT | 3.500 | 3.638 | [0.238, 0.516] | pass | 8.611% |
| Lab mean/std | 3.624 | 2.274 | [0.118, 0.309] | pass in moderate stratum | 0.000% |
| per-channel quantile | 3.299 | 2.840 | [0.437, 0.727] | fail frozen style floor | 0.612% |
| Gaussian/Bures | 3.892 | 3.140 | [-0.104, 0.063] | fail fidelity | 2.169% |

The strong basic target Delta-E/style are `3.872/2.919`; the paired affine
reference target Delta-E is `3.120`. Pooled L2 is the strongest statistically
eligible candidate, while Lab is the conservative zero-OOR comparator.

## Decision

No single operator passes all three recipe domains. The universal pooled-L2
claim fails. The supported deterministic architecture is a **fixed per-recipe
champion bank**, not per-photo ML routing:

- Cinema: Lab mean/std primary;
- ClassNeg: pooled L2 primary, with sliced/Bures/basic risk comparators;
- Velvia: pooled L2 primary, with Lab as the conservative comparator.

This is not yet a product promotion. Pooled L2 and sliced/Bures outputs have
raw out-of-range risk, and sampled pixels cannot reveal banding, seams, edge
contamination, faces, text or red-speckle failures. All confirmatory survivors
must pass full-resolution metrics and visual severe vetoes. Statistical
failures are not reopened by visual preference.

The evidence does show that an explicit nonlinear operator can learn a strong,
recipe-specific transformation rather than merely raise saturation: pooled L2
decisively beats a strong WB/contrast/saturation adversary on ClassNeg and
Velvia, while correctly losing on Cinema. That heterogeneity is exactly why a
single averaged neural LUT can look bland.

## Reproducibility

- frozen pilot decision SHA-256:
  `68bd158a9f3d0d9dfd3229ef84a5e950fef3d474f772185405a3b6e462744f23`;
- confirmatory software commit:
  `58c1799a3ca923303915cf60b514761f5829cfef`;
- confirmatory report SHA-256:
  `51e16eb047296e93b3bc7c739adf5a499a7f80f3d48bf95a228db059614d5fe3`;
- frozen confirmatory-decision SHA-256:
  `7b38f46ae0414d2bcf33164f651e757122ffceb20da991eaf21a19fa66600879`;
- complete rerun: byte-identical;
- final 628 parsed/decoded: `false`;
- repository tests before execution: 75 passed.

Generated per-image metrics and operator bundles remain ignored and
internal-only.
