# P296 — standards-aware OpenEXR chromaticity ingress contract

Status: prospectively frozen before P296 pixel reads or comparison metrics.

## Question

Can a separately versioned opt-in loader apply the OpenEXR standard's default
Rec.709/D65 chromaticities when the attribute is absent, honor explicit source
chromaticities when present, and map the official Rec709/XYZ reference images
into one consistent linear Rec.2020/D65 WorkingImage representation?

This is not a rescue or rewrite of P294. P294's explicit-only policy remains
failed and unchanged. P296 tests a new, standards-aware policy on the same
already-consumed official sources and therefore opens no fresh-data claim.

## Exact sources and runtime

- ASWF `openexr-images` revision
  `e38ffb0790f62f05a6f083a6fa4cac150b3b7452`;
- licence blob `268d234adc762429a5d7db6a456651d7d0933dd1`;
- declaration blob `8d94d9e2211874955faa6c55021174283335921c`;
- exact P294 `Rec709.exr` and `XYZ.exr` objects and source manifest;
- exact OpenEXR 3.4.15 CPython 3.12 wheel already pinned by P294.

## Frozen arithmetic and file policy

1. Accept exactly one non-deep scanline part and one packed RGB group, with no
   alpha or extra channel, float16 or float32 storage, positive bounded size,
   finite decoded samples and absolute component no greater than 65,504.
2. If `chromaticities` is present, use the exact eight xy values. If absent,
   use Rec.709/D65 only when the caller explicitly enables
   `allow_standard_rec709_default`; otherwise reject before output.
3. Build the source RGB-to-XYZ matrix analytically from xy primaries and white,
   reject singular/non-finite/invalid xy systems, Bradford-adapt the source
   white to D65, then apply the fixed XYZ-D65-to-linear-Rec.2020 matrix.
4. Return an owned, writable, C-contiguous float32 WorkingImage, preserve
   negative/highlight values without clipping, retain the exact source path and
   record whether identity was embedded or standard-defaulted.
5. Never infer DCI-P3/ACES from filenames or prose, never use a sidecar, ICC,
   JPEG, display rendering, normalization or gamut clipping.

## Frozen gates

- both official files load and produce exact 610x406 linear Rec.2020/D65 rows;
- repeat and opposite-order output hashes are exact;
- cross-source RMSE <= 0.002, p99 absolute error <= 0.005 and maximum absolute
  error <= 0.05; these bounds are frozen before P296 pixel reads;
- finite/ownership/source-immutability/negative-highlight preservation pass;
- missing-default authorization, alpha/extra channel, malformed xy,
  non-finite pixels, excessive dimensions and wrong runtime fail atomically;
- P294 and P295 evidence/source hashes remain unchanged.

Any failed gate closes P296 without changing thresholds, sources, matrices,
channel policy or default authorization. Do not substitute another image or
add a special-case correction.

## Claim ceiling

At most one private two-source standards-aware OpenEXR-to-linear-Rec.2020
mechanical result. No arbitrary EXR, DCI-P3/ACES, HDR or display quality,
default dispatch, public dependency/API/package/schema/capability/product,
stock evidence or candidate-3 change.
