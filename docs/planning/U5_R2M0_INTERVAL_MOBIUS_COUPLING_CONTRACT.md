# U5.R2M0 — Bounded Interval-Möbius Coupling Contract

**Status:** frozen before implementation or result inspection  
**DRPT level:** L2  
**Parent:** U5.R2K1/K3 representation failures plus U5.R2L1 policy close  
**Primary writer:** current Codex Goal session

## Question

Can a lower-capacity, explicitly invertible colour coupling represent the two
frozen nonlinear film-inspired controls while keeping RGB bounds and local
gain safe by construction?

M0 is a clean-room synthetic representation audit. It does not fit real images,
real-film pixels, preference labels or stock metadata. A pass opens only a
separately frozen real-image Style-safe frontier.

## Independent motivation

K1 proved that triangular single-channel couplings can be orientation
preserving and analytically invertible, but its one-sided piecewise-affine
channel update underfit both controls. K3 proved that a strong palette
factorization can be diverse and invertible, but its explicit polar gamut
boundary amplified one local derivative beyond the frozen norm gate.

M0 changes the update equation rather than increasing either failed model:

1. a six-term polynomial conditioner reads only the two unchanged channels;
2. two bounded non-negative terms independently lift the lower endpoint and
   compress the upper endpoint;
3. a bounded log-odds/Möbius term bends the channel monotonically inside that
   interval;
4. six triangular stages are composed in the frozen order `R,G,B,G,B,R`.

For input channel `x`, conditioner outputs `a,b,k` define:

```text
lower = Lmax * tanh(max(a, 0))
upper = 1 - Umax * tanh(max(b, 0))
m(x,k) = x*exp(k) / (1 - x + x*exp(k))
y = lower + (upper-lower)*m(x,k)
```

The inverse is analytic on the transformed interval: undo the interval affine
map and apply `m(z,-k)`. Because a stage conditioner never reads the channel
being changed, reverse-order inversion sees exactly the same conditioning
values. Positive interval width and the positive Möbius derivative make every
stage orientation preserving. No output clamp or projection is allowed.

The official ENNeLUT paper is used only for the high-level precedents of
bounded-coordinate colour modeling and invertibility-constrained residual
composition. M0 does not reimplement its MLP, copy code, weights, LUTs,
normalization constants or training data.

## DoR

- K1, K3 and L1 configs, gates and conclusions remain immutable.
- The exact equations, stage count, features, optimizer, grids and gates are
  frozen before implementation.
- The fixed 108 coefficients are fewer than K1's 144; this is not a same-form
  capacity rescue.
- No real-image or real-film access is allowed.

## Frozen targets and split

The development grid is uniform `8^3`; confirmation is uniform `11^3`.
Normalization and fitting use development only. Targets are immutable:

- identity;
- U5.R2E0 `cyan_shadow_warm_highlight_like` at strength `0.50`;
- U5.R2J0 `warm_highlight_like` at strength `0.35`.

## Gates

- identity maximum error `<=1e-12`;
- all outputs in `[0,1]`, with no clamp;
- coefficient absolute maximum `<=2`;
- both nonlinear confirmation RMSE `<=.015`;
- each nonlinear target improves at least 20% over its best global affine fit;
- minimum analytic stage derivative `>=.02`;
- sampled Jacobian determinant strictly positive;
- sampled Jacobian spectral norm `<=8`;
- inverse roundtrip maximum error `<=1e-10`;
- exact serialization replay and exact partition parity;
- two formal reports byte-identical.

## Branches

- **All gates pass:** retain M0 and freeze one fixed-operator real-image
  frontier with inherited style, non-basic, clipping and severe gates.
- **Fidelity fails:** close; do not add stages, features, steps or optimizer
  variants.
- **Bounds/inverse/orientation/norm fails:** close; do not add clamp,
  projection, smoothing or weaken gates.
- **Only floating exactness fails:** report it as a failure; no epsilon change.
- **Later real-image style fails:** close without fitting images.
- **Any severe artifact appears:** veto regardless of style.

## DoD

- pure explicit operator, inverse and serialization;
- deterministic development-only fitter and formal audit;
- formula/property/replay/partition tests;
- two byte-identical reports;
- full CPU suite;
- result/decision propagation;
- scoped commits and pushes.

## Claim ceiling

Clean-room, data-independent representation evidence for a bounded explicit
interval-Möbius coupling fitted only to existing synthetic explicit controls.
No external LUT/code/weight reuse, neural renderer, real-film operator
identification, named stock, calibration, authenticity, preference or
production claim.
