# SF2.7A scanner-nuisance quantification contract

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.7A`

Status: **frozen before colour-result inspection**

## Question

How large and stable are scanner-hardware/software colour differences when the
same five physical Velvia 100F test slides are observed through four pipelines,
and how much survives simple global nuisance canonicalization on a held-out
slide?

This question measures a nuisance control. It does not identify a stock
operator.

## Frozen evidence and colour boundary

- inherit the exact 11-asset, 71,068,957-byte SF2.7R freeze;
- use all four complete pipelines and all five shared slide identities;
- use the recorder-space source TIFF only for geometric registration;
- preserve the native 16-bit scan samples and normalize by the integer range;
- label scan colours `unknown_scanner_device_rgb`;
- do not treat the untagged TIFF values as calibrated sRGB or colourimetric
  measurements.

The primary distance is Euclidean distance in normalized scanner device RGB.
CIEDE2000 under an explicit sRGB-display assumption is reported only as a
sensitivity diagnostic and is not called measured Delta E.

## Patch and alignment contract

The fixed source chart contains a 22 by 12 grid bounded by source coordinates
`x=[56,704]`, `y=[61,415]`. The inner 50% of each cell supplies one robust
channel median.

Each source chart is registered to its matching scan with SIFT, a fixed 0.72
ratio test and a 3-pixel RANSAC homography. Every pair must have at least 50
good matches, 40 inliers and no more than 1.5 pixels median inlier reprojection
error. Alignment failure closes the quantitative node rather than allowing
manual outcome-dependent correspondence.

## Frozen evaluation

Evaluate all six unordered pipeline pairs in both mapping directions using
five leave-one-slide-out folds. Four slides fit a nuisance mapping and the
fifth is test-only. Report raw and held-out residual median, p90 and worst-patch
distances for:

1. identity;
2. independent per-channel affine;
3. bounded full 3x3 affine plus bias.

The full affine coefficients are bounded to `[-2,2]`, bias to `[-0.5,0.5]`,
and test output to `[0,1]`. No spatial, per-slide, per-patch, neural or
result-tuned mapping is allowed.

## Pre-registered interpretation

- aggregate raw median RGB distance at least 0.04 and every pair at least 0.02:
  material stable scanner/software nuisance in this target set;
- best global canonicalizer reducing aggregate median by at least 50%:
  globally controllable nuisance component;
- best residual aggregate median no more than 0.02:
  small residual under this bounded test;
- larger residual:
  future stock/mode evidence must explicitly beat the retained scanner floor.

These engineering thresholds classify this audit only. Passing any threshold
does not open stock learning, operator fitting or LSM.

## Integrity and branches

- exact parent config/report hashes are verified;
- all ZIP/member hashes and slide roles come from the parent audit;
- fitting occurs only on four development slides in each fold;
- two complete reports must be byte-identical;
- alignment failure closes SF2.7A as unidentified;
- material nuisance is recorded as a negative control, not a film mode;
- complete canonicalization does not make the images stock truth;
- partial canonicalization records both removable and residual components.

## DoD

- contract/config committed before quantitative execution;
- isolated module/CLI and focused tests;
- two deterministic reports;
- visual alignment sheet reviewed without changing gates;
- complete CPU suite;
- decision and stock-first/LSM control propagation;
- scoped commit and push;
- Ultimate Goal continues to the next ready algorithm/data leaf.
