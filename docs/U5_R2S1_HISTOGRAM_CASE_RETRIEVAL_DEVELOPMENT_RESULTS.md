# U5.R2S1D Canonical-Histogram Recovery Development Results

Date: 2026-07-26

Decision: **select query-KDE bandwidth `.12` for one untouched
confirmation**

## Reproducibility

- software commit:
  `0477a04799cd8284a14898edc6f789a8bd04d44b`;
- config SHA-256:
  `5af8fc68e0cc75f97555161b8adaf707dd597a73dc5f8ca200a22b50c8fff766`;
- two reports are byte-identical at
  `f4804c4ff5dfa5b81f77f2fd918fbdc4d18049d87c80c2beeed9ed6d7004df11`;
- five focused tests pass without warnings;
- reserved confirmation seed `27012` was not accessed.

## Development ranking

| Method | Median operator RMSE | p90 operator RMSE | Median style retention | Median non-affine retention | Median reference separation |
|---|---:|---:|---:|---:|---:|
| query KDE `.12` | .03268 | .05076 | .984 | .909 | .935 |
| query KDE `.08` | .03341 | .04363 | .961 | .903 | .896 |
| query KDE `.16` | .04185 | .05914 | - | - | - |
| query KDE `.20` | .05094 | .07833 | - | - | - |
| Top-3 Hellinger blend | .07042 | .09912 | .924 | .772 | .818 |
| hard Hellinger Top-1 | .08727 | .12829 | 1.030 | .928 | .956 |
| hard JS Top-1 | .09075 | .12996 | - | - | - |
| global mean | .11279 | .13585 | .552 | .352 | 0 |

Every method passes the development structural eligibility checks. Exact
sample-row permutation changes neither the histogram nor any predicted
operator.

## What the result says

The strongest method is not a neural network and not hard case retrieval. A
fixed KDE over the query's canonical colour counts estimates the analytic
palette score directly, then the S0 flow turns it into a deterministic
explicit operator.

KDE `.12` has median direction cosine `.955`, retains `98.4%` of oracle style
strength, `90.9%` of non-affine residual and `93.5%` of reference separation,
and improves query-palette log density by median `2.789`. Its sampled minimum
determinant is `.01259`, maximum norm `4.819` and worst inverse error
`1.19e-6`.

Hard Hellinger Top-1 retains slightly more pairwise separation (`.956` versus
`.935`), but its median operator error is about `2.67x` larger. This small
diversity difference does not justify the large recovery loss. Top-3 blending
also loses accuracy and separation, consistent with the concern that mixing
cases averages away distinctive directions.

The global mean shows the clearest averaging failure: only `55.2%` style
retention, `35.2%` non-affine retention and zero query-dependent separation.

## Boundary

This is synthetic development. The query histogram was sampled from the same
latent density whose exact analytic score defines the evaluation oracle. Real
film scans do not reveal their digital-to-film operator, and raw palette
colour remains strongly content-dependent.

Therefore this result closes only **raw-histogram hard retrieval for this
synthetic score-recovery question**. It does not close future within-stock
retrieval over evidence-backed operator signatures, and it does not authorize
using current film pixels, fitting a stock, or treating scene colour as mode
evidence.

One untouched confirmation of KDE `.12`, hard Hellinger Top-1 and global mean
may now be frozen before seed `27012` is generated.
