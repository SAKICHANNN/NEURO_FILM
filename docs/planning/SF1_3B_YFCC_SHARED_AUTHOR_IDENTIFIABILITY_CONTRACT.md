# SF1.3B shared-author stock-identifiability contract

Date: 2026-07-16

Node: `ULT > RF1 > SF1.3B`

Status: frozen before diagnostic execution

## Question

Within the same eight Flickr UIDs, can Ektar100 versus Velvia50 be separated
on held-out UIDs by global RGB evidence that is stronger than content,
low-frequency scene colour, geometry/border/source and basic standardization
controls?

This is an identifiability diagnostic, not training or operator fitting.

## Frozen design

- exact SF1.3A manifest SHA-256 `33395e5b...68ba3`;
- 21 Ektar100 and 16 Velvia50 files;
- group is Flickr UID; all pixels from a held-out UID are excluded from fit;
- equal-weight UID-by-stock units prevent authors with four files dominating;
- descriptors: RGB distribution, luma distribution, grayscale HOG, 4x4
  low-frequency RGB, standardized RGB distribution and geometry/border/source;
- 499 group-label permutations, seed 1801;
- 4,000 paired group bootstraps, seed 1802;
- preprocessing, centroids and scaling are fit without each held-out UID.

## Frozen gates

The stock signal passes only if all are true:

1. at least five groups per label;
2. RGB balanced accuracy >=0.70;
3. RGB permutation p<=0.05;
4. RGB accuracy exceeds the best nuisance control by >=0.10;
5. paired bootstrap 95% lower bound for that delta is >0;
6. HOG accuracy <=0.65.

The nuisance set includes luma, HOG, 4x4 scene colour, standardized RGB and
geometry/border/source. No threshold may change after results are read.

## Branches

- pass: open only a separately frozen residual/stock evidence design; no
  operator fitting, training or LSM automatically opens;
- fail: mark this shared-author pool unidentified and close it for stock
  learning; do not increase model capacity or reinterpret clusters;
- ambiguous/small-sample instability: same as fail for authorization while
  retaining the result as evidence.

Claim ceiling: shared-author, group-held-out stock-signal diagnostic only. No
stock response, physical process, latent mode, calibration, authenticity,
weights or release claim.
