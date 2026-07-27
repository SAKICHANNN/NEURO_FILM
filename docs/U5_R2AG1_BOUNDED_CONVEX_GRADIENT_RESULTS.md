# U5.R2AG1 — bounded convex-gradient representation results

Date: 2026-07-28  
Decision: `close_frozen_gate_failure`

## Outcome

The 65-scalar bounded convex-gradient map is structurally safe but does not
have enough capacity for the frozen density-domain control.

Two formal one-thread CPU runs are byte-identical. Every construction-level
gate passes:

- exact identity;
- 65 raw fitted scalars;
- output remains in `[0,1]` without clamping;
- strength and centred-bias bounds;
- analytic positive-definite Jacobian and determinant floors;
- Jacobian spectral-norm bound;
- damped-Newton inverse roundtrip;
- exact serialization replay.

The positive-film control passes fidelity at `0.009446` RGB RMSE and improves
over the global affine baseline by `65.71%`. The density control remains at
`0.040467` RGB RMSE against the frozen `0.015` ceiling, despite improving on
the affine baseline by `42.32%`. This fidelity failure is independently
decisive.

The non-identity maps also differ by one or two float64 ulps when evaluated
whole versus partitioned (`1.11e-16` and `2.22e-16`). The frozen contract
requires bit-exact partition parity, so that gate also fails. It is not a
visible artifact and does not alter the capacity conclusion.

## Exact evidence

- Implementation commit:
  `608bcd0db4074d3e71f12c35cdfc3cc91febe11f`
- Config SHA-256:
  `224ce2256c88188a9145d6d81ac50d1eb26820117608bc051e42b2160c191a9e`
- Run A report SHA-256:
  `33cdc6b6931c94a134c957dbf2f9d186e27878803f42a0ba65f3a981b910c2df`
- Run B report SHA-256:
  `33cdc6b6931c94a134c957dbf2f9d186e27878803f42a0ba65f3a981b910c2df`
- Repeated decision SHA-256:
  `7a265ba00a3e39c5a72a9ab211b0e67258c96ab53747d57ebfcfa00651adbe0e`
- Focused tests: `5 passed`
- Complete CPU suite before formal execution: `1030 passed`

Formal reports and fitted operators remain ignored under
`outputs/u5_r2ag1_bounded_convex_gradient_representation_v1/`.

## Structural metrics

| Target | Confirm RMSE | Affine gain | Min eigenvalue | Min determinant | Max norm | Inverse max error |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0 | n/a | 1.0000 | 1.0000 | 1.0000 | 0 |
| density cyan 50% | 0.040467 | 42.32% | 0.4450 | 0.09067 | 1.8554 | 1.97e-12 |
| positive warm 35% | 0.009446 | 65.71% | 0.6513 | 0.27825 | 1.3869 | 1.38e-12 |

## Decision and boundary

AG1 closes under its preregistered branch. Do not add anchors, attach an affine
wrapper, lower the temperature, search optimizers, relax fidelity, or repair
partition parity. The retained O0 cube-diffeomorphic flow remains the smallest
tested safe representation that passes both frozen nonlinear controls.

This is paired-synthetic representation evidence only. It does not identify
an unpaired digital-to-film operator, validate optimal transport as film
learning, authorize current pixels, establish a stock or latent mode, or open
production integration.
