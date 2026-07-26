# U5.R2S1 Canonical-Histogram Case Retrieval Development Contract

Date frozen: 2026-07-26

Parent: `U5.R2S0`

Status: ready

## Question

Can the project recover a strong reference-specific palette-score flow using
the deliberately simple rule:

> this query colour distribution most resembles case X, therefore reuse case
> X's bounded explicit colour operator?

This is the closest controlled test of the intended FilmCase intuition that
does not pretend current unpaired film scans provide digital-to-film targets.

## Shortcut boundary

The query representation is an `8 x 8 x 8` normalized RGB histogram. It has
no pixel coordinates, texture, object, face, geometry, date, source or
scanner feature. Shuffling all sampled RGB rows must leave both the histogram
and selected operator bit-exact.

This blocks spatial content shortcuts, but it does not make colour
distribution a film-stock signal. The experiment is synthetic and tests only
recovery mechanics.

## Synthetic development design

The fixed generator creates diagonal RGB Gaussian mixtures with:

- two, three or four components;
- Dirichlet-`1.5` weights;
- means in `[.06,.94]`;
- standard deviations in `[.09,.24]`.

There are `384` independent case-bank palettes and `96` independent
development queries. Each observed histogram uses `8192` draws. The latent
analytic mixture supplies the S0 oracle flow only for evaluation.

Seed `27012` is reserved for confirmation. Code and formal development runs
must fail closed if asked to generate it before a separately committed
confirmatory contract exists.

## Fixed candidates

1. hard Top-1 by Hellinger histogram distance;
2. hard Top-1 by Jensen-Shannon distance;
3. inverse-distance Top-3 Hellinger coefficient blend;
4. query-only histogram KDE with bandwidth `.08`, `.12`, `.16` or `.20`;
5. the global mean velocity grid.

Hard retrieval is nonparametric ML. KDE is an important no-training control.
Top-3 explicitly tests whether blending nearby cases averages away the
direction. The global mean represents the failure mode already observed in
earlier style learners.

All candidates predict only a fixed `4^3 x 3` bounded velocity grid. The
32-step deterministic diffeomorphic integrator remains the sole RGB renderer.

## Development outputs

Report velocity and operator error against the latent oracle, direction
cosine, style and non-affine retention, query-density attraction, reference
separation, exact permutation invariance and the inherited range,
Jacobian-norm, positive-orientation, inverse and replay diagnostics.

Selection is preregistered:

1. reject any structurally ineligible method;
2. minimize median operator RMSE;
3. minimize p90 operator RMSE;
4. maximize affine-residual retention;
5. prefer the simpler method on a tie.

Development may set numerical confirmatory gates, but only before seed
`27012` is generated or evaluated.

## Branches

- An eligible primary opens one frozen untouched confirmation.
- If query KDE materially wins, retain it and close retrieval unless hard
  retrieval independently preserves materially more diversity.
- If only the global mean or blend is competitive, close the hard-case
  hypothesis under this observation model.
- If no method is eligible, close without adding a neural network.

## Claim ceiling

This node can establish only development evidence about recovering synthetic
analytic palette-score flows from canonical colour histograms. It cannot
establish film appearance, stock identity, real-image routing, preference,
calibration, production quality or an identified unpaired operator.
