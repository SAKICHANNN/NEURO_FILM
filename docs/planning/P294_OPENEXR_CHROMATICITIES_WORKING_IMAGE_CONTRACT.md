# P294 — OpenEXR chromaticities WorkingImage contract

Status: prospectively frozen before either EXR body is requested or decoded.

## Question

Can a strict private ingress use the OpenEXR `chromaticities` attribute to
convert two official ASWF reference images into the same unclipped
`linear_rec2020_d65` WorkingImage representation?

This is a file-boundary and colour-arithmetic question. It is not a rescue of
the P293 EyefulTower file, whose missing container identity remains closed.

## Exact sources and rights

- repository: `AcademySoftwareFoundation/openexr-images`;
- revision: `e38ffb0790f62f05a6f083a6fa4cac150b3b7452`;
- repository licence blob: `268d234adc762429a5d7db6a456651d7d0933dd1`;
- source declaration blob: `Chromaticities/README.rst`,
  `8d94d9e2211874955faa6c55021174283335921c`;
- inputs: `Chromaticities/Rec709.exr` (908,168 bytes, Git blob
  `13b3fca94da1426e4793bd66a20299c955b73d00`) and
  `Chromaticities/XYZ.exr` (930,048 bytes, Git blob
  `cd4423ab12669b4caaaeb157ddd08fe430f5a775`).

The official declaration says that the four files in this directory test the
OpenEXR chromaticities attribute and should look the same when displayed
properly. P294 uses only the two ordinary RGB files; the luminance/chroma files
and JPEG previews are forbidden.

## Frozen implementation

The ingress accepts exactly one non-deep scanline part with separate R/G/B
HALF channels, equal data/display windows, unit sampling, finite pixels and an
exact recognized chromaticity tuple. It supports only the two source-locked
tuples:

- Rec.709 primaries with D65 white;
- XYZ identity primaries with equal-energy white.

For either tuple it constructs the RGB-to-XYZ matrix from xy primaries,
Bradford-adapts the declared white to D65, converts XYZ-D65 to linear Rec.2020,
and returns an owned writable contiguous float32 `WorkingImage`. It never
clips, normalizes, tone-maps or reads a sidecar.

The existing Rec.709/D65 conversion primitive is the independent control for
the Rec709 row. The official paired-source invariant is the independent
cross-representation control: converted Rec709 and XYZ results must agree.

## Frozen gates

1. exact revision, Git blob, byte count and SHA-256 for both acquired files;
2. exact OpenEXR 3.4.15 runtime wheel identity;
3. one scanline part, HALF R/G/B, equal 610x406 windows, unit sampling;
4. exact source-locked chromaticities and no missing-metadata fallback;
5. finite output, owned/writable/contiguous float32, input files immutable;
6. Rec709 candidate versus existing Rec.709/D65 primitive: max absolute error
   at most `2e-6`;
7. converted Rec709 versus converted XYZ: RMSE at most `5e-4`, p95 at most
   `1e-3`, and maximum absolute error at most `5e-3`;
8. no clipping or normalization; negative/highlight support must be retained
   when present;
9. missing/wrong chromaticities, FLOAT channels, mismatched windows,
   non-unit sampling, multipart/deep and nonfinite inputs reject before a
   WorkingImage is returned;
10. two fresh-process reports in opposite source order are byte-exact and all
    formal temporary outputs are removed.

## Stop and claim boundary

Any failed gate closes exact P294 without changing source, tuple, precision,
adaptation, threshold, source order or channel family. Do not add the P293
EyefulTower file, YC files, JPEGs, an inferred sidecar or a gamut/tone rescue.

A pass establishes only a private two-file chromaticity-aware OpenEXR ingress
mechanism. It is not arbitrary EXR, DCI-P3, ACES, HDR quality, display
rendering, a default loader, package/schema/capability/product admission,
stock evidence or candidate-3 progress.
