# U5.R2R1 Published D-LUT Asset Structural Contract

Date frozen: 2026-07-26

Parent: `U5.R2R0`

Status: ready

## Question

Does any nonidentity step in the one published D-LUT trajectory satisfy the
project's pre-existing explicit-operator range and orientation requirements
without modifying the source asset?

This is a deterministic characterization after a disclosed development
inspection. It is not a blind confirmatory experiment.

## DoR

- official revision and Apache-2.0 licence fixed;
- 41 contiguous official LUT files and one checkpoint hash fixed;
- `.cube` axis order independently specified;
- paper/demo epsilon contradiction recorded;
- no film pixel, stock claim or image rendering is needed.

## Fixed audit

For steps `0..40`:

1. validate path, byte size, SHA-256 and canonical manifest hash;
2. parse standard red-fastest `.cube` rows into project RGB axes;
3. verify step 0 against exact identity;
4. report node range and identity displacement;
5. compute exact determinants for all six tetrahedra of every cell;
6. compute analytic trilinear Jacobians on the frozen
   `{0, 0.5, 1}^3` subcell grid;
7. report minimum determinants, nonpositive fractions and maximum spectral
   norm;
8. identify steps satisfying every structural gate;
9. repeat the complete run and compare report bytes.

## Frozen gates

- identity step maximum absolute error `<=1e-6`;
- every LUT node in `[0,1]`;
- minimum trilinear and tetrahedral determinant `>0`;
- maximum sampled trilinear and tetrahedral Jacobian spectral norm `<=8`;
- at least one nonidentity step passes all structural gates;
- two reports are byte-identical.

The thresholds are inherited from prior bounded global-operator work. They
were not selected after viewing D-LUT's values.

## Branches

- No nonidentity survivor: close before real images. Do not rescue with
  clipping, smoothing, projection, regularization, strength interpolation,
  retraining or another seed.
- Survivor: freeze the earliest and strongest survivors for a separately
  preregistered A0 style/severe audit.
- Integrity/parser failure: close as non-reproducible; do not repair external
  assets.

## Claim ceiling

One official reference image and one released trajectory can establish only a
method-artifact structural result. It cannot establish film stock, arbitrary
reference safety, owner preference, calibration, unpaired operator
identification or product readiness.
