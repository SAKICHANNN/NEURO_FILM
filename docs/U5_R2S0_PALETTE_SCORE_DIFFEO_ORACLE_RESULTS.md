# U5.R2S0 Palette-Score Diffeomorphic Oracle Results

Date: 2026-07-26

Decision: **all frozen gates pass; open only a separately frozen
histogram-to-score recovery experiment**

## Reproducibility

The evaluator ran twice at software commit
`a229f8dadc2beadbc3690b2207ff5e2bb372e35e`.

- config SHA-256:
  `17f4c377dcb2a534d1bf7ebf5c613bf980b51e1e7aa47db688acceb9ff714502`;
- both report SHA-256 values:
  `eeaf7b2b72973f7b837e1c3a94d1fda0aca03e008358114b70eab0a09cc11b2d`;
- identity maximum absolute error: `0`;
- minimum pairwise operator-output RMSE: `.10784`;
- five focused tests pass.

## Gate result

| Palette | Identity RMSE | Affine residual | Log-density gain | Min determinant | Max norm | Inverse error |
|---|---:|---:|---:|---:|---:|---:|
| warm earth | .15268 | .06599 | 3.3754 | .01416 | 3.6968 | 7.89e-7 |
| cool aqua | .14931 | .06647 | 3.1868 | .01545 | 4.1616 | 1.17e-6 |
| cyan shadow / warm highlight | .13219 | .05322 | 2.4025 | .01626 | 3.6928 | 4.72e-7 |

Every palette also:

- remains strictly inside the RGB cube, with observed extrema between
  `.07055` and `.93097`;
- stays below the coefficient-vector cap of `2`;
- has exact replay and partition parity;
- passes every preregistered style, attraction, range, orientation, norm and
  inverse gate.

All three pairwise operator comparisons exceed the `.04` separation floor;
the weakest pair is `.10784`.

## Interpretation

This is the first clean mechanism result in this branch showing that the
useful part of D-LUT's idea can be separated from its folded node motion. A
reference colour-density score can supply a large, distinct and materially
non-affine direction, while a boundary-vanishing stationary flow supplies a
bounded, positive-orientation and invertible explicit RGB operator.

The result does not show that a model can infer this score from a photograph.
It uses three exact analytic Gaussian-mixture densities and no image pixels.
It therefore does not establish reference retrieval, film style, stock
identity, unpaired digital-to-film identification, preference or production
quality.

## Next leaf

The only opened action is a separately preregistered synthetic recovery test:

```text
canonical colour histogram
  -> small bounded score/velocity predictor
  -> fixed cube-preserving flow
  -> deterministic explicit RGB operator
```

That experiment must use group-separated development and untouched
confirmatory palette families, be exactly invariant to spatial permutation,
compare against a simple non-learned density baseline and retain all current
structural gates. Current real-film pixels, operator fitting and latent-mode
work remain forbidden.
