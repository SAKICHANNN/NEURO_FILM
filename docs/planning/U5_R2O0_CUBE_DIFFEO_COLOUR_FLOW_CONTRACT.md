# U5.R2O0 — Cube-Preserving Diffeomorphic Colour Flow Contract

**Status:** frozen before implementation or result inspection  
**DRPT level:** L2  
**Parent:** U5.R2K0–K3 and U5.R2N1 negative representation evidence  
**Primary writer:** current Codex Goal session

## Question

Can one compact, explicit stationary colour-velocity field reproduce the two
frozen nonlinear project controls while remaining bounded, invertible,
orientation-preserving and below the existing Jacobian-norm ceiling?

This is a synthetic representation audit. It does not fit photographs, infer a
film stock, solve unpaired digital-to-film identification or reopen the
completed CT5 sliced-transport experiment.

## Research basis and exclusion boundary

Continuous normalizing flows and neural ODEs establish the general principle
that integrating a sufficiently regular ODE velocity field defines a
continuous invertible transformation. Stationary-velocity diffeomorphic
registration uses the same exponential-map principle for spatial
transformations. R2O0 applies that mathematical idea to the three-dimensional
colour cube, not to image geometry.

Primary method precedents:

- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018:
  <https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html>
- Grathwohl et al., *FFJORD: Free-form Continuous Dynamics for Scalable
  Reversible Generative Models*, ICLR 2019:
  <https://openreview.net/forum?id=rJxgknCcK7>
- Mang and Biros, *Constrained H1-regularization schemes for diffeomorphic
  image registration*, 2015: <https://arxiv.org/abs/1503.00757>

No source code, checkpoint, learned density, image encoder or external
parameters are copied. The project implementation is clean-room and the
optimizer estimates only a finite explicit velocity grid.

## Candidate

The stationary field has one trilinearly interpolated `4×4×4×3` coefficient
grid. At colour `x`, its channel velocity is:

```text
v_i(x) = x_i * (1 - x_i) * f_i(x)
```

The boundary factor makes every face normal-invariant: a channel at exactly
zero or one stays there, while finite coefficients cannot cross that boundary
under the continuous flow. The final operator is the time-one solution of
`dx/dt=v(x)`, approximated by a fixed 24-step float64 RK4 integrator. Its
inverse uses the same integrator with negative time. This produces RGB directly
through an explicit deterministic numerical operator; there is no neural RGB
generator.

The coefficient grid is projected into `[-6,6]`. Identity is the exact
all-zero grid.

## Data and split

Only three existing synthetic explicit targets are allowed:

- identity;
- U5.R2E0 cyan-shadow/warm-highlight at strength `0.50`;
- U5.R2J0 warm-highlight positive-film response at strength `0.35`.

Fit uses an `8³` midpoint grid. Confirmation uses an untouched `11³` midpoint
grid. Jacobians are measured on a separate `7³` interior grid. No image or
film-scan payload is accessed.

## DoR

- K0–K3 and N1 results remain immutable.
- The completed CT5 sliced-transport bank remains untouched.
- This config, capacity, optimizer and every gate are committed before
  implementation.
- No real-film pixel, image download or external implementation is used.

## Frozen gates

- Identity confirmation error at most `1e-12`.
- Cube corners/endpoints fixed within `1e-12`.
- All outputs remain in `[0,1]` without a clamp.
- Maximum absolute coefficient at most `6`.
- Confirmation RGB RMSE at most `0.015` on every target.
- Each nonlinear target improves at least `20%` over its fitted global-affine
  confirmation baseline.
- Sampled Jacobian determinant at least `0.02`; spectral norm at most `8`.
- Numerical inverse roundtrip maximum error at most `2e-5`.
- Serialization replay and partition errors exactly zero.
- Two complete reports byte-identical.

## Branches

- **All gates pass:** retain as a numerical representation; a later
  simplest-model or visual frontier requires its own frozen contract.
- **Fidelity fails with regularity intact:** close this fixed field; do not
  enlarge the grid or tune the integrator on confirmation.
- **Regularity, bounds or inverse fails:** close; no output clamp, determinant
  penalty or step-count rescue.
- **Only one nonlinear target passes:** report family-specific evidence but do
  not claim a general representation pass.
- **Later visual value fails:** preserve only as a negative/control
  representation.

## DoD

- focused operator and runner tests;
- two byte-identical formal reports;
- result and decision records under the exact claim ceiling;
- full CPU suite;
- tracker, plan, board and agent-log propagation;
- scoped commits and pushes.
