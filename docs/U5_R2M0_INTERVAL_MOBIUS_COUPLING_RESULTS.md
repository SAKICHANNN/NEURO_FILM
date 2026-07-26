# U5.R2M0 Interval-Möbius Coupling Results

## Decision

**Closed: structural pass, positive-control pass, density fidelity and exact
identity fail.**

The fixed six-stage, 108-coefficient explicit operator is bounded,
orientation-preserving, analytically invertible and comfortably below the
local Jacobian-norm budget. It does not represent the frozen density-cyan
control accurately enough, and its common fitted identity is not exact. The
automatic conjunction fails, so no real-image render or visual review is
allowed.

## Reproducibility

- implementation commit:
  `87a3c01a422250be3baba783322d8d3ee3458405`;
- config SHA-256:
  `fd535e9d582d13c6565b47291905d09c636daf88c7c14dfd47ec19a9c476d534`;
- two formal report SHA-256 values:
  `c4c469a14f16c1024b4148758900b8bd760ea354a7c207345ec01a2c409ef898`;
- both formal exits: failure, byte-identical reports;
- five focused M0 tests and 17 combined K1/K2/K3/M0 tests pass;
- complete CPU suite: `873 passed`.

The reports and fitted operator payloads remain in the ignored
`outputs/u5_r2m0_interval_mobius_coupling_v1/` evidence directory.

## Frozen results

| Target | Confirm RMSE | Affine RMSE | Gain over affine | min det(J) | max ||J||2 | inverse max error |
|---|---:|---:|---:|---:|---:|---:|
| density-cyan s0.50 | .030890 | .070155 | 55.97% | .07639 | 2.8415 | 8.88e-16 |
| positive-warm s0.35 | .011700 | .027546 | 57.52% | .25591 | 1.7775 | 6.66e-16 |
| identity | 8.71e-6 | 3.14e-16 | n/a | .99980 | 1.0002 | 2.78e-16 |

The density target misses the frozen `.015` confirmation ceiling by more than
2x. The positive target passes. Identity maximum absolute error is
`5.97e-5`, above the exact `1e-12` gate. All targets remain in `[0,1]`;
minimum analytic stage derivative passes; every sampled determinant is
positive; maximum local norm is only `2.8415` versus the frozen `8`; inverse,
serialization and partition checks pass, with partition error exactly zero.

## Interpretation

Independent lower lift, upper compression and log-odds bend solve the main
regularity problem seen in K0/K3: the operator does not fold the cube or create
large local gain. It also has enough nonlinearity to beat global affine and
fit the simpler positive-film witness.

The failure is expressive, not numerical instability. The frozen six-stage
polynomial conditioners cannot reproduce the more complex exposure-dependent
density-cyan trajectory to the required fidelity. Exact identity also does not
emerge from the common regularized fitter. Special-casing identity or adding
stages after seeing these results would invalidate the preregistered
experiment.

This is useful negative evidence: safe invertibility and moderate Jacobian
norm are achievable, but they do not by themselves provide the particular
strong film-colour direction the project needs.

## Binding branch

- close M0 without image rendering;
- do not add stages, features, optimizer variants or coefficient headroom;
- do not special-case identity or weaken exactness/fidelity gates;
- retain the module only as isolated research infrastructure;
- select a separately motivated explicit representation, physical film-effect
  algorithm or stronger-data leaf.

No real-film operator, stock identity, calibration, preference, training,
production or general artifact-safety claim opens.
