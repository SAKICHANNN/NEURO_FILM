# U1.3B default float-internal renderer migration contract

Date: 2026-07-17

Node: `ULT > U1 > U1.3B`

Status: frozen before implementation

## Question

Can the default 8-bit export keep its existing file formats and established
8-bit appearance while removing the renderer's early sRGB8 quantization, so
all supported inputs use the existing float32 safe-Lab and procedural-effect
core and quantize only at final export?

This is an internal precision and architecture migration. It does not change
the default output bit depth, create HDR or wide-gamut support, calibrate RAW
scene-to-display rendering, or authorise stock learning.

## Parent evidence

- U1.1 routes every supported input through `WorkingImage`, but default 8-bit
  output still calls `working_image_to_legacy_srgb8` before colour rendering;
- U1.3A already exposes the same `style_transfer_rgb` and effect compositor as
  a float32 path for opt-in PNG/TIFF16;
- the PIL colour wrapper is only an sRGB8 input/output compatibility boundary
  around that float core;
- the preregistration diagnostic below was run before implementation against
  three deterministic image families, five styles and both no-effect and
  combined grain/halation/dust conditions.

## Frozen pre-implementation diagnostic

| Input | Condition | Cases | Worst max code delta | Worst mean code delta | Worst fraction above 1 code |
|---|---|---:|---:|---:|---:|
| sRGB8 | colour only | 15 | 0 | 0 | 0 |
| sRGB8 | combined effects | 15 | 1 | 0.211 | 0 |
| sRGB16 | colour only | 15 | 88 | 0.240 | 0.00104 |
| sRGB16 | combined effects | 15 | 45 | 0.268 | 0.00101 |

The sparse high-bit-depth outliers occur after removing the old input
quantization and are strongest at black-and-white transform discontinuities.
They are not hidden by an average-only gate and require explicit regression
and visual review.

## Allowed implementation

- make `working_image_to_srgb_float` the sole colour-render input adapter;
- use `build_color_render_float` for both 8-bit and 16-bit exports;
- keep one final quantization in the existing suffix-aware encoder;
- retain the public PIL wrapper for compatibility consumers and its tests;
- record `legacy_8bit_adapter=false` and `internal_color_precision=float32`
  for every renderer output;
- add deterministic parity tests at the renderer boundary.

## Frozen gates

1. **8-bit colour compatibility:** synthetic and fixed sRGB8 inputs must be
   exactly equal after final uint8 quantization with no procedural effects.
2. **8-bit effect compatibility:** deterministic combined-effect output must
   differ by at most one uint8 code per channel; no pixel may exceed one code.
3. **High-bit-depth honesty:** a regression fixture must demonstrate that the
   default 8-bit export consumes high-bit-depth source detail through the float
   core rather than reproducing the early-quantized legacy path. No exact
   compatibility claim is allowed for 16-bit or RAW input.
4. **Outlier safety:** representative high-bit-depth colour and B&W renders
   must receive full-output numerical comparison and autonomous visual severe-
   artifact review. Any new posterization, banding, clipping block, colour
   speckle or geometry corruption rejects the migration.
5. **Format/provenance:** PNG/JPEG/TIFF8 defaults, dimensions, ICC, output
   claim and bit-depth metadata remain unchanged except the two internal-path
   fields above. PNG/TIFF16 and invalid JPEG16 behavior remain unchanged.
6. **Verification:** targeted tests, complete CPU suite, diff check and a real
   high-bit-depth/RAW smoke must pass before promotion.

## Stop branches

- If an sRGB8 colour result differs, preserve the default legacy path and fix
  the adapter/core mismatch before adding features.
- If deterministic effects exceed one code on sRGB8, preserve the legacy path
  until the amplification is understood.
- If high-bit-depth outliers create a confirmed severe artifact, keep U1.3A
  opt-in only and record the failing fixture; do not relax the artifact veto.
- If only sparse non-severe high-bit-depth differences remain, document them as
  removal of premature quantization and promote the float-internal default.

## Claim ceiling

At most: the default Style-safe renderer uses one float32 internal colour and
effect path and performs final 8- or 16-bit sRGB quantization at export. Output
remains `film-inspired/look-approximation`. This is not HDR, wide gamut,
camera-accurate RAW, calibrated scene-to-display, a stock response, or evidence
for operator fitting, training or latent modes.
