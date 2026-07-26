# U5.R2K0 — Bounded Gaussian Residual Representation Contract

**Status:** frozen before implementation or result inspection  
**DRPT level:** L2  
**Parent:** U5.R2 explicit colour-operator research  
**Primary writer:** current Codex Goal session  

## Question

Can an independently implemented, fixed-geometry bank of localized Gaussian
affine residuals compactly reproduce already frozen nonlinear explicit colour
controls while remaining analytically range bounded, replayable and locally
regular?

This is a representation question only. It does not identify a digital-to-film
operator and does not reopen any real-film pixel pool, stock-learning gate,
latent-mode gate or closed visual frontier.

## Source audit

The May 2026 paper *GLUT: 3D Gaussian Lookup Table for Continuous Color
Transformation* describes an explicit mapping formed by normalized Gaussian
weights, local affine transforms and one global affine transform. The v1 PDF is
fixed at SHA-256
`2a71064ec5db9ad1cc899ccf9a6d769acbecb9cb342e6b2deebf5d997ba09c3f`;
its arXiv source archive is fixed at
`e66f18acc71a76788dd2da1a0c95664a04f3e93086e923fc62cf23ebc05676d5`.

The source review found four boundaries that prevent a direct project
transplant:

1. the paper clamps final RGB to `[0,1]`, which is not a structural safety
   guarantee and is incompatible with this project's no-hidden-clamp standard;
2. the appendix says affine transforms initialize as identity, while the
   published mapping adds local and global affine branches without specifying
   the global initialization; an identity global branch would start near
   `2*x`;
3. the paper does not constrain monotonicity, Jacobian orientation, coefficient
   magnitude or pre-clamp range;
4. no official implementation or lineaged 300-LUT corpus was found in the
   paper, source archive or bounded primary-source search. The paper is
   distributed under arXiv's non-exclusive distribution licence, not a software
   licence.

Accordingly, K0 may use the published mathematical idea as prior art but must
be a clean-room implementation. It may not copy code, learned parameters,
unverified LUT assets or the conditional generator.

## Candidate

For input `x` in `[0,1]^3`, fixed regular-grid Gaussian primitives define
normalized weights. Their weighted local affine functions and one global affine
function predict a residual `delta(x)`. The final mapping is:

```text
x + (1-x) * tanh(max(delta, 0)) + x * tanh(min(delta, 0))
```

This signed-headroom form is identity at zero residual and analytically remains
inside `[0,1]` without output clipping. Geometry is fixed; only affine residual
coefficients are fitted by deterministic ridge regression. This makes the fit
linear, CPU-only and exactly replayable.

The two frozen capacities are 8 and 27 primitives. K0 fits only three existing
synthetic explicit controls:

- identity;
- the fixed U5.R2E0 cyan-shadow/warm-highlight density witness at strength
  `0.50`;
- the fixed U5.R2J0 warm-highlight positive-film witness at strength `0.35`.

These controls are numerical targets, not film truth or preference labels.
Fit and confirmation colour grids are disjoint.

## DoR

- J1 is closed with no visual gain and its configs remain immutable.
- The GLUT PDF/source hashes and source/licence boundaries are recorded.
- No current real-film pixels are opened for fitting.
- The K0 config and all gates are committed before the formal run.

## Frozen gates

- Published-equation negative control must demonstrate near-identity with a
  zero global branch and a maximum output above `1.5` with an identity global
  branch before clamp.
- Identity confirmation maximum absolute error: `1e-12`.
- Analytic output range: `[0,1]`.
- Maximum absolute fitted coefficient: `4.0`.
- 27-primitive confirmation RGB RMSE: at most `0.015` for every target.
- For each nonlinear target, 27 primitives must reduce confirmation RMSE by at
  least `20%` relative to a best global affine RGB fit.
- On the frozen interior grid: determinant strictly positive, diagonal
  derivatives non-negative and spectral norm at most `8.0`.
- JSON replay and partition errors: exactly zero.
- Two complete reports: byte-identical.

## Branches

- **All gates pass:** retain the representation only and open a separately
  frozen K1 comparison against the simplest existing explicit operator. No
  visual tuning is automatic.
- **Fidelity fails:** close the compact representation at the tested capacity;
  do not increase capacity on the same confirmation grid.
- **Regularity fails:** close the candidate; do not rescue with post-hoc clamp
  or a neural generator.
- **Only identity/global-affine value passes:** close as no useful nonlinear
  representation gain.
- **Any real-image severe artifact in a later leaf:** reject that policy
  regardless of style.

## DoD

- focused tests pass;
- two formal reports are byte-identical;
- result and decision records state the exact claim ceiling;
- affected trackers and agent log are propagated;
- scoped commits are pushed under the existing Goal authorization.
