# U5.R2K2 — Rational-Quadratic Colour Coupling Contract

**Status:** frozen before implementation or result inspection  
**DRPT level:** L2  
**Parent:** U5.R2K1 structural pass / fidelity failure  
**Primary writer:** current Codex Goal session

## Question

Can a compact composition of conditionally parameterized monotonic
rational-quadratic spline stages preserve K1's boundedness, positive
orientation and analytic inverse while passing the same frozen nonlinear
explicit-control fidelity gates?

K2 is a synthetic representation audit. It does not learn from photographs,
infer film identity, reopen real-film operator fitting or authorize a visual
frontier.

## Independent research basis

Durkan, Bekasov, Murray and Papamakarios, *Neural Spline Flows* (NeurIPS
2019, arXiv `1906.04032v2`) derives monotonic rational-quadratic scalar
transforms with positive derivatives and a quadratic closed-form inverse. The
paper shows that this scalar bijection can replace less expressive
affine/additive transforms inside triangular coupling layers without giving up
one-pass invertibility.

The exact paper PDF is retained as an ignored research source:

```text
bytes:   4,913,781
sha256:  1f20e349d8e77125fc918ca82039f42aa831d0289379fc650960289d9df9b6a0
```

K2 adopts only the Appendix-A scalar formula and the established triangular
coupling principle. It independently implements the formula. It does not copy
external code, parameters or checkpoints and excludes the paper's density
objective, generative sampling and neural conditioners.

This is not a prohibited K1 capacity rescue. K1's scalar stage was a single
bounded piecewise-affine shift controlled by fixed basis coefficients. K2
tests a separately published nonlinear scalar bijection whose internal knot
geometry and derivatives are explicit.

## Frozen candidate

Each of six stages changes exactly one channel in the order:

```text
R, G, B, R, G, B
```

The other two current channels are expanded into a fixed `2x2` normalized
Gaussian basis plus bias and affine terms. Those seven values linearly
parameterize one four-bin spline:

- four width logits;
- four height logits;
- three internal derivative offsets.

Widths and heights are normalized with fixed floors of `.04`. Internal
derivatives use a positive parameterization with floor `.02`; endpoint
derivatives are exactly one. Zero coefficients produce equal knots,
unit derivatives and exact identity.

Because stage parameters depend only on the two unchanged channels, the stage
Jacobian is triangular. Positive spline derivative makes its determinant
positive. The spline maps `[0,1]` onto `[0,1]` exactly, with no clamp, and the
inverse solves the paper's numerically stable quadratic root. Composition
retains these properties.

All explicit conditioner coefficients are projected to `[-2,2]`. Fitting uses
deterministic CPU float64 Adam only; the optimizer is absent at inference and
no network predicts pixels.

## Data and split

Use exactly the same existing synthetic explicit targets as K1:

- identity;
- U5.R2E0 cyan-shadow/warm-highlight at strength `.50`;
- U5.R2J0 warm-highlight positive-film at strength `.35`.

Fit uses the `8^3` midpoint grid. Confirmation remains the untouched `11^3`
midpoint grid and Jacobians use the separate `7^3` interior grid. No threshold,
capacity or optimizer change is allowed after viewing confirmation.

## DoR

- K1 results/config remain immutable and its same-form capacity branch stays
  closed.
- The exact primary paper and clean-room boundary are recorded.
- This contract and config are committed and pushed before implementation.
- No external code, checkpoint, photo, LUT or film scan is used.

## Frozen gates

- Identity confirmation maximum error at most `1e-12`.
- Output remains analytically in `[0,1]`; no clamp.
- Maximum absolute coefficient at most `2`.
- Observed width and height of every spline bin at least `.04`.
- Confirmation RGB RMSE at most `.015` for every target.
- Each nonlinear target improves at least `20%` over its best global affine
  confirmation baseline.
- Minimum analytic per-stage derivative at least `.02`.
- Every sampled finite-difference Jacobian determinant strictly positive;
  maximum spectral norm at most `8`.
- Analytic inverse roundtrip maximum error at most `1e-10`.
- Serialization replay and partition errors exactly zero.
- Two full reports byte-identical.

## Branches

- **All gates pass:** retain K2 as a numerical representation and freeze a
  simplest-model/complexity comparison before any real-image use.
- **Fidelity fails while structure passes:** close this fixed candidate. Do not
  add bins, stages or conditioner features on the same confirmation grid.
- **Derivative, orientation, inverse or range fails:** close; do not rescue
  with clamp, finite-difference inversion or post-hoc smoothing.
- **Optimizer/repeat fails:** close reproducibility before interpreting fit.
- **Any later severe artifact:** veto the candidate regardless of style.

## DoD

- focused formula, identity, inverse, serialization and determinism tests;
- two byte-identical formal reports;
- result and binding decision records;
- full CPU suite;
- authority/tracker/log change propagation;
- scoped commits and pushes.

## Claim ceiling

Clean-room, data-independent representation evidence for a finite explicitly
parameterized monotonic rational-quadratic colour coupling fitted only to
existing synthetic explicit controls. No density model, generative sampling,
neural conditioner, learned style latent, real-film operator, named stock,
calibration, authenticity, preference or production claim.
