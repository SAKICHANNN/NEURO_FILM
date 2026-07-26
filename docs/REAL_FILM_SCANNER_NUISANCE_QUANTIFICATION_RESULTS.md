# SF2.7A scanner-nuisance quantification results

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.7A`

Decision: **partial control pass; scanner nuisance is real but not universally
large, and simple global control removes most of the median difference**

## Integrity and alignment

All 20 source-to-scan pairs pass the frozen automatic geometry gates. The
minimum RANSAC support is 75 inliers and the maximum median inlier reprojection
error is 0.644 pixels. Both complete reports are byte-identical at SHA-256
`bd093389e4e3808979ada7380308b6fe0e573a8421e4ec10c0de797c781993ad`.
The deterministic 20-panel overlay sheet is
`fedd19ff325b885983b2a648be3361d14e202ab4db505cf79f1b8f2162e9513c`;
visual review confirms that every magenta 22x12 grid follows the chart cells
without a displaced slide or pipeline.

The analysis preserves native 16-bit scan samples. Because the TIFFs have no
embedded ICC profile, the primary measure is normalized scanner-device-RGB
distance. Any sRGB-assumed CIEDE2000 field is sensitivity-only, not a
colourimetric measurement.

## Held-out result

All six pipeline pairs are evaluated in both directions with five
leave-one-slide-out folds and 264 held-out patches per slide.

| Mapping | Median RGB distance | P90 | Maximum |
|---|---:|---:|---:|
| identity/raw | 0.06055 | 0.15093 | 0.28520 |
| per-channel affine | 0.02549 | 0.14083 | 0.26470 |
| bounded 3x3 affine + bias | 0.01446 | 0.06120 | 0.20549 |

The bounded full affine reduces the aggregate median by 76.11% and passes the
frozen small-residual threshold.

## Important non-uniformity

The raw nuisance is not uniformly large:

- the two LS50/NikonScan devices differ by only 0.01464 median RGB distance,
  below the frozen 0.02 every-pair floor;
- LS50 VueScan versus LS9000 NikonScan reaches 0.09865;
- the remaining cross-software/cross-model pair medians span 0.04638–0.09293.

Therefore the compound `material_raw_nuisance` gate is false even though the
aggregate and most cross-pipeline differences are substantial. The correct
conclusion is not “every scanner creates a separate mode.”

## Meaning

This target set demonstrates two facts future stock/mode studies must respect:

1. scanner model/software can create large, stable appearance differences from
   identical physical slides;
2. a bounded global colour canonicalizer can remove most median difference,
   so raw scanner clusters must not be promoted to film modes.

The p90 residual remains 0.0612 and the worst patch 0.2055, so a global affine
is not a universal per-patch correction. These tails are nuisance stress
evidence, not proof of emulsion behaviour.

## Branch

Record SF2.7A as a scanner/software negative control and canonicalization
baseline. Do not train a stock expert, fit a digital-to-film operator or open
LSM from this single target set. A future eligible stock study must group
scanner/source, report raw and canonicalized sensitivity, and treat
scanner-aligned clusters as nuisance unless independent evidence says
otherwise. The Ultimate Goal continues.
