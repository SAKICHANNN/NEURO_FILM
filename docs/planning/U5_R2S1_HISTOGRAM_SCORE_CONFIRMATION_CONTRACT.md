# U5.R2S1C Canonical-Histogram Score Confirmation Contract

Date frozen: 2026-07-26

Parent: `U5.R2S1D`

Status: ready

## Fixed primary and controls

The primary is the development-selected query-only KDE with bandwidth `.12`.
It converts a normalized `8^3` RGB histogram into a bounded `4^3 x 3`
velocity grid. The same 32-step explicit diffeomorphic renderer is used.

Only two controls remain:

- hard Hellinger Top-1 from the fixed 384-case bank;
- the fixed global-mean velocity grid.

No other bandwidth, blend, metric or learned model may be inspected after
confirmation begins.

## Untouched population

Confirmation uses `128` new palettes from reserved seed `27012`. The palette
generator, `8192` histogram samples, case bank seed `27010`, histogram bins,
operator representation and coefficient cap are inherited exactly from
development.

This seed was not generated during development. The committed contract is the
authority that now permits its first access.

## Frozen conjunction

The primary must:

- have median/p90 oracle-output RMSE no greater than `.04/.065`;
- have median velocity-direction cosine at least `.90`;
- retain median style in `[.80,1.20]`;
- retain median non-affine residual at least `.75`;
- improve query log density by median at least `1.5`;
- retain median reference separation at least `.80`;
- reduce median error by at least `45%` versus hard Top-1 and `50%` versus
  global mean;
- preserve exact histogram/operator permutation invariance;
- remain within the cube, have sampled minimum determinant `>.005`, maximum
  norm `<=8`, inverse error `<=1e-5`, exact replay and coefficient norms
  `<=2`;
- produce two byte-identical reports.

## Branch discipline

A complete pass retains only a synthetic explicit-operator mechanism prior.
Before any photograph can be rendered, another leaf must show how content
palette nuisance is separated from desired reference style. A failed
accuracy, retention, control or structural gate closes this fixed route
without switching to KDE `.08`, tuning bandwidth or adding a neural model.

## Claim ceiling

Even a complete pass does not identify a film stock or an unpaired
digital-to-film operator. It does not authorize current film pixels, LSM,
production integration or a real-image style claim.
