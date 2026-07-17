# U1.3B default float-internal renderer results

Date: 2026-07-17

Decision: **pass — promote the existing float32 colour/effect core for default 8-bit export**

## Implemented boundary

`render_film.py` now sends every supported `WorkingImage` through
`working_image_to_srgb_float`, `style_transfer_rgb` and the float32 procedural
compositor. The existing suffix-aware encoder performs the only output
quantization, to sRGB8 by default or sRGB16 when explicitly requested.

The PIL `style_transfer` wrapper and legacy input adapter remain available for
compatibility tests and external callers, but neither is on the renderer's main
path. Metrics report `legacy_8bit_adapter=false` and
`internal_color_precision=float32` for both output depths.

## Frozen-gate evidence

- deterministic sRGB8 colour parity is exact;
- deterministic sRGB8 combined grain/halation/dust parity is bounded to one
  code per channel, with no pixel above one code;
- a 16-bit E2E regression proves default sRGB8 output equals the float-core
  result and differs from the early-quantized legacy result;
- PNG/JPEG/TIFF8 format, dimensions, ICC, output label and bit depth remain
  unchanged; PNG/TIFF16 and JPEG16 rejection regressions pass;
- focused colour/ingress tests: 15 passed;
- complete CPU suite: 223 passed.

## Full-resolution RAW audit

The existing 24,969,216-byte Sony ARW fixture renders at 6024x4024 from
`linear_srgb/scene_linear` 16-bit source provenance. The default Velvia sRGB8
render is ICC-tagged, bounded to [4, 251], records the float32 internal path and
retains the `film-inspired/look-approximation` claim. Autonomous full-resolution
review finds no new banding, posterization, colour blocks, geometry corruption
or clipping failure.

| Style | Max code delta | Mean code delta | Fraction >1 code | Fraction >4 codes | Bounds old/new |
|---|---:|---:|---:|---:|---|
| Velvia 50 | 24 | 0.2953 | 0.0960% | 0.00191% | [4,251] / [4,251] |
| HP5 | 114 | 0.1556 | 0.0111% | 0.00699% | [4,251] / [4,251] |

The HP5 comparison exposes a pre-existing B&W profile defect: sparse orange
residual chroma is visible around extreme highlights in both paths. The float
path does not worsen aggregate contamination: pixels with channel spread above
20 codes fall from 0.1233% to 0.1195%; 392 pixels are newly above that threshold
while 1,322 legacy pixels are resolved. This is not a migration regression,
but it remains a separate severe-artifact candidate that must be fixed or
explicitly excluded before promoting B&W profiles.

## Claim ceiling and remaining work

This closes the default renderer's early-quantization debt and makes
`WorkingImage -> float32 sRGB look core -> final quantization` the single main
path. It does not add HDR, HEIF, wide gamut, a calibrated RAW scene-to-display
transform, stock calibration, operator fitting, training or latent modes.

The nearest safety follow-up is a separately frozen B&W chroma-invariant leaf;
it must not change the completed U1.3B gates or hide the existing evidence.
