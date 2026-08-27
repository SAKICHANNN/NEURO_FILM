# U4.3A three-stock structural diagnostics contract

## Question

On the already-rendered RF3.D0 population, which fixed three-stock Look
Approximation outputs most need original-resolution review for colour-induced
edge, texture, flat-region or chroma-speckle anomalies?

## Frozen inputs

- the exact 16-row RF3.D0 report and its 64 existing renders;
- the exact AO7S decoded-source manifest and its existing RGB8 PNGs;
- the four unchanged RF3.D0 arms: Velvia, Portra, Ektar and the AO6 Velvia
  display-proxy comparator.

No image is rendered, fitted, downloaded or relabelled by this leaf.

## Diagnostics

All calculations use linear-sRGB luminance derived from the encoded RGB8
source and output.

1. Source-edge direction consistency: Sobel gradient cosine on source pixels
   at or above the 75th gradient percentile. Report p05 and median.
2. Source-edge gain: output/source Sobel magnitude ratio on the same mask.
   Report p05, median and p95.
3. Fine-texture retention: absolute one-pixel Gaussian high-pass ratio on
   source pixels at or above the 75th high-pass percentile. Report median and
   p95.
4. Flat-region new high frequency: positive output-minus-source absolute
   high-pass on source pixels at or below the 25th percentile. Report p95,
   p99 and maximum.
5. Reuse the existing U4.3 high-frequency chroma-island diagnostic without
   changing its thresholds.

For each metric, publish an independently sorted review queue. Do not combine
the metrics into a learned or hand-weighted scalar score.

## Gates and claim ceiling

The formal gates are exact input/output hashes, all 16 sources, all four arms,
finite metrics, deterministic forward/reverse reports, no writes outside the
owned report path and no pixel rendering. These diagnostics never authorize
or veto a candidate automatically. A severe decision still requires the
frozen full-resolution adjudication protocol.

This is product-safety diagnostic evidence for existing Look Approximation
outputs. It is not stock identification, stock calibration, preference,
multi-stock completion or product promotion.
