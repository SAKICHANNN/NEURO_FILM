# U5.R2K1 — Triangular Monotone Colour Coupling Contract

**Status:** frozen before implementation or result inspection  
**DRPT level:** L2  
**Parent:** U5.R2K0 regularity failure  
**Primary writer:** current Codex Goal session

## Question

Can a compact composition of analytically invertible, range-bounded
single-channel coupling stages reproduce the same frozen nonlinear explicit
controls without the colour-space folds observed in K0?

K1 is a synthetic representation audit. It does not learn from photographs,
infer film identity or reopen real-film operator fitting.

## Research basis and exclusion boundary

Coupling layers are established invertible-transform components: one subset of
coordinates parameterizes a monotone transform of another subset, producing a
triangular Jacobian with a tractable positive determinant. The 2022 CVMP paper
by Mustafa, Hanji and Mantiuk applies conditional invertible networks to global
colour/tone mappings and demonstrates that multiple artistic mappings cannot
be recovered from content alone.

K1 adopts only the mathematical coupling principle. It excludes that paper's
generative density objective, learned style latent, image-pair dataset,
checkpoints and code. The project operator is independently implemented,
deterministic and explicit after fitting.

## Candidate

Each stage changes exactly one channel `c`. The other two current channels
condition a fixed `3x3` bank of 2D Gaussian basis functions plus an affine
basis. Their weighted sum produces `delta`. Channel `c` is updated by:

```text
x_c + (1-x_c) * tanh(max(delta, 0))
    + x_c * tanh(min(delta, 0))
```

Because `delta` does not depend on `x_c`, the stage derivative with respect to
`x_c` is either `1-tanh(delta)` or `1+tanh(delta)`, always positive for finite
parameters. Other channels are unchanged. Every stage is therefore
orientation-preserving, bounded and analytically invertible. A composition of
such stages retains those properties.

The frozen twelve-stage channel order is:

```text
R, G, B, G, B, R, B, R, G, R, G, B
```

All conditioner coefficients are projected into `[-1.5,1.5]`. Fitting uses
deterministic CPU float64 Adam only to estimate these explicit coefficients;
the optimizer is not part of inference and no network predicts RGB.

## Data and split

Only the three existing synthetic explicit targets are used:

- identity;
- U5.R2E0 cyan-shadow/warm-highlight at strength `0.50`;
- U5.R2J0 warm-highlight positive-film at strength `0.35`.

Fit uses the `8^3` midpoint grid. Confirmation uses the previously unseen
`11^3` midpoint grid; K0's `12^3` confirmation grid is not reused for threshold
tuning. The Jacobian grid is a separate `7^3` interior grid.

## DoR

- K0 is closed under its frozen regularity branch.
- K0 configs/results remain immutable.
- K1 config, optimizer, capacity and gates are committed before implementation.
- No external code, checkpoint, photo, LUT or film scan is used.

## Frozen gates

- Identity confirmation error at most `1e-12`.
- Output remains analytically in `[0,1]`; no clamp.
- Maximum absolute coefficient at most `1.5`.
- Confirmation RGB RMSE at most `0.015` for every target.
- Each nonlinear target improves at least `20%` over its best global affine
  confirmation baseline.
- Minimum analytic per-stage updated-channel derivative at least `0.02`.
- Every sampled finite-difference Jacobian determinant strictly positive;
  maximum spectral norm at most `8`.
- Analytic inverse roundtrip maximum error at most `1e-10`.
- Serialization replay and partition errors exactly zero.
- Two full reports byte-identical.

## Branches

- **All gates pass:** retain K1 as a numerical representation and open a
  separately frozen simplest-model comparison; no real-image frontier opens
  automatically.
- **Fidelity fails while regularity passes:** close the fixed capacity. Do not
  add stages/basis functions on the same confirmation grid.
- **Regularity or inverse fails:** close the formulation; do not use clamp or
  numerical inversion as rescue.
- **Optimizer repeat fails:** close reproducibility before interpreting
  fidelity.
- **Later visual value fails:** keep only as a representation/negative control.

## DoD

- focused tests;
- two byte-identical formal reports;
- result/decision records with exact claim ceiling;
- full CPU suite;
- tracker and agent-log propagation;
- scoped commits and pushes.
