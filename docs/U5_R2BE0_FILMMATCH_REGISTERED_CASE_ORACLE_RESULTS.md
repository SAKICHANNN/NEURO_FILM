# U5.R2BE0 — Registered FilmMatch case Oracle

The fit-forbidden validation pair is geometrically usable, but the frozen case
bank fails its diversity gate and does not open retrieval.

Two executions are byte-identical at report SHA-256
`2c07628ec3dc07e5e1f4512739b6a74542d49b0b799cc99ed9a02079c279c06d`.
SIFT/RANSAC retains 706/779 inliers (90.63%), with 0.519 px median and
2.109 px P95 reprojection error. The fixed low-gradient evaluation mask covers
38.74% of the target.

The registered evaluator supplies real descriptive value:

- global RMSE is `.17280`, 13.67% below identity;
- the best three-regime member is `positive`, at `.15352` RMSE and 11.15%
  improvement over global;
- the best exact-EV member is `EV +2`, at `.15537` and 10.09% improvement;
- no candidate adds a source-relative boundary.

The preregistered combined-bank diversity gate nevertheless fails. The
`regime_zero` and `EV 0` fits are necessarily identical, and `EV -5` versus
`EV -4` also collapses to `3.20e-10` output RMSE. The three regime members
alone are distinct (`.01278`–`.07726`), but the contract does not allow
post-result removal of failed cases.

BE0 therefore closes without a selector, visual shortlist or product route.
The positive-regime result remains evaluator-Oracle evidence only. A later
retrieval experiment needs a genuinely independent paired validation source;
the consumed target cannot be reused as confirmation.
