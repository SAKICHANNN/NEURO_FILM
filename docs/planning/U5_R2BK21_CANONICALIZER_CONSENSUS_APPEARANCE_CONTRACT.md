# U5.R2BK21 Canonicalizer-Consensus Appearance Contract

Date: 2026-07-31  
Status: frozen before implementation

## Question

BK20 proves that even perfect unpaired source and target colour sets do not
identify the transform that paired their samples. BK21 therefore asks a
strictly weaker product-method question:

> Can two independently specified appearance objectives agree strongly enough
> to authorize one bounded explicit look, while weak evidence or disagreement
> produces an exact identity fallback?

This is an appearance-matching policy, not an operator-identification method.

## Fixed policy

Two existing, independently parameterized objectives fit the existing O0
cube-preserving flow:

1. 48 fixed sliced-quantile projections;
2. 144 fixed random Fourier MMD features.

Both see identical generated unpaired development sets. On an independent
validation set, the policy computes each candidate's normalized residual under
both objectives. It hard-selects the one candidate with the lower mean ratio
only when:

- the reference change is material;
- the two full-cube operators disagree by at most `0.06` RGB RMSE;
- the selected candidate improves the two-objective validation score by at
  least 25%;
- the selected flow passes the existing range, Jacobian, inverse and replay
  gates.

Otherwise it returns exact identity. It never averages operators, velocity
grids or rendered RGB.

## Discriminating scenarios

- `material_shared_look`: independent source/target samples from the same
  generated population, with the target transformed by one bounded explicit
  style. The policy must retain one hard candidate.
- `appearance_already_matched`: independent samples from the same unstyled
  population. The weak-change gate must return identity.
- `content_confounded_reference`: source and reference come from deliberately
  shifted populations. Canonicalizer sensitivity or insufficient shared
  validation value must return identity.

The third case is a content-confound stress test, not a model of film.

## Branches and ceiling

All gates passing retains only a synthetic mechanism prior for future
rights-cleared appearance matching. Any failure closes this frozen policy
without threshold, optimizer, loss, capacity or scenario rescue.

No branch identifies a digital-to-film operator, opens current stock pixels,
proves a latent mode, trains a router, changes AO6, or promotes a product
renderer. The maximum label is `film-inspired appearance matching`.
