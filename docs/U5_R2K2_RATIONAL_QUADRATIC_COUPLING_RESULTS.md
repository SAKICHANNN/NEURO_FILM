# U5.R2K2 Rational-Quadratic Coupling Results

## Decision

**Closed on confirmation fidelity and Jacobian norm.** The six-stage
rational-quadratic coupling operator preserves every intended structural
property: exact identity, bounded output without clamp, positive sampled
orientation, finite positive analytic derivatives, a closed-form inverse and
exact partition/serialization replay. It also fits the density control below
the frozen `.015` confirmation ceiling.

The positive-film control reaches `.01965` confirmation RMSE and fails. The
density control reaches a maximum Jacobian spectral norm of `9.457`, above the
frozen `8.0` ceiling. No real-image frontier opens and the fixed candidate
must not be enlarged or retuned on the same confirmation grid.

## Reproducibility

| Item | Value |
|---|---|
| Primary source | Durkan et al., *Neural Spline Flows*, NeurIPS 2019 |
| Paper PDF SHA-256 | `1f20e349d8e77125fc918ca82039f42aa831d0289379fc650960289d9df9b6a0` |
| Software commit | `c626edfc32acc6af964a9dff6da86a5b5eac8511` |
| Config SHA-256 | `cefd751de2a79502e710284d6b6121912da80bf2072b39dc65301a0791cf2da5` |
| Report SHA-256 | `1e8082c4e94840e06fd01bad6440ec6273c1dd7e1f59b7eff5c69029674a2b47` |
| Repeated reports | byte-identical |
| Fit / confirmation / Jacobian points | `512 / 1,331 / 343` |
| Explicit parameters | 462 |
| Stages / bins per stage | 6 / 4 |
| Focused tests | 4 passed |

The project independently implements only the monotonic rational-quadratic
scalar formula and coupling structure. It uses no external code, weights,
neural conditioner, density objective or generative sampling.

## Results

| Target | Fit RMSE | Confirmation RMSE | Affine RMSE | Gain vs affine | Min det(J) | Max norm | Inverse max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 0 | 0 | ~0 | n/a | 1.0000 | 1.0000 | 0 |
| density cyan s0.50 | .00243 | **.01435** | .07016 | 79.55% | .03092 | **9.457** | `1.22e-15` |
| positive warm s0.35 | .00067 | **.01965** | .02755 | 28.67% | .14051 | 7.669 | `5.55e-16` |

The positive control's very small fit error but materially worse untouched-grid
error is direct generalization evidence, not an optimizer invitation. Both
nonlinear targets exceed the frozen 20% affine-relative gain floor, but the
fidelity gate applies to every target and therefore fails.

All spline-bin floors pass. Minimum observed widths/heights are:

- density: `.07939 / .06498`;
- positive: `.13278 / .13949`;
- identity: `.25 / .25`.

Minimum analytic stage derivatives are `.2903`, `.4838` and `1.0`
respectively. Every sampled determinant is positive. Range, coefficient,
inverse, serialization and exact partition gates pass.

## Interpretation

K2 answers the K0/K1 trade-off more sharply:

- K0 had high fidelity but folded colour space.
- K1 was safely invertible but underfit.
- K2 is safely invertible and expressive enough for the density target, but
  its finite fixed form does not generalize within the positive target and can
  create locally excessive colour gain.

This makes monotonic rational-quadratic coupling a useful representation
result, not a renderer candidate. Analytic invertibility alone does not impose
the perceptual smoothness needed by the severe-artifact budget; the independent
Jacobian-norm gate remains necessary.

## Binding branches

Forbidden:

- add spline bins, stages or conditioner features on this confirmation grid;
- change fit steps, optimizer or regularization after seeing the result;
- relax `.015` fidelity or `8.0` Jacobian-norm gates;
- clamp or smooth the fitted result after the fact;
- render real images, fit real-film pixels or claim stock response.

Allowed:

- retain the implementation and results as a clean-room explicit-operator
  control;
- choose a separately motivated film-colour factorization whose regularity is
  structural;
- continue searching for rights-cleared common-input film measurements.

## Claim ceiling

Clean-room, data-independent negative representation evidence. K2 is not a
density model, generative image system, learned style latent, identified
digital-to-film operator, named-stock response, calibration, authenticity,
preference result or production promotion.
