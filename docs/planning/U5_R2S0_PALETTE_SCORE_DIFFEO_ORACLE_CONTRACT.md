# U5.R2S0 Palette-Score Diffeomorphic Oracle Contract

Date frozen: 2026-07-26

Parent evidence: `U5.R2O0`, `U5.R2R1`

Status: ready

## Scientific question

Can the strong direction supplied by a reference colour-density score be
expressed through a bounded cube-preserving diffeomorphic RGB operator without
becoming bland, affine or reference-insensitive?

This separates two causes of failure:

- if an exact analytic score cannot pass, training a score network is
  unjustified;
- if it passes, a later leaf may test whether a small ML model can recover the
  required score from a canonical colour histogram.

## Fixed candidate

Three preregistered diagonal-Gaussian-mixture palette densities represent
warm-earth, cool-aqua and cyan-shadow/warm-highlight directions.

For each density:

1. evaluate its exact `grad(log p(rgb))` on a `4^3` velocity grid;
2. map each score vector `s` to
   `2 * s / (1 + ||s||_2)`;
3. use these vectors as the coefficients of U5.R2O0's boundary-vanishing
   stationary velocity;
4. integrate with fixed 32-step float64 RK4;
5. evaluate on untouched confirmation, Jacobian and palette-attraction grids.

No coefficient is fitted to an output transform.

## DoR

- R2O0 already validates the representation and inverse;
- R2R1 establishes that raw stochastic node transport is strong but folded;
- all mixtures, grids, coefficient mapping, thresholds and branches are
  frozen;
- no external asset, image, GPU or film pixel is required.

## Gates

Every palette must:

- remain exactly inside `[0,1]`;
- have minimum finite-difference determinant `>1e-5`;
- have maximum Jacobian spectral norm `<=8`;
- invert within `1e-5`;
- replay and partition exactly;
- move at least `.05` RMSE from identity;
- retain at least `.02` RMSE beyond the best global affine;
- improve its own mean analytic log density by at least `.1`.

The minimum pairwise output RMSE across palette operators must be at least
`.04`. Identity must remain bit-exact and coefficient vector norms cannot
exceed `2`.

## Stop branches

- Structure failure closes without changing cap, grid or integration steps.
- Style/non-affine/reference separation failure closes without adding ML.
- Density-attraction failure closes because the operator no longer carries
  the intended score semantics.
- A complete pass opens only a new histogram-to-score recovery contract. It
  does not open real images, film fitting or stock claims.

## Claim ceiling

This is an analytic synthetic mechanism test. Gaussian-mixture palettes are
not photographs, film stocks, latent modes or identified operators.
