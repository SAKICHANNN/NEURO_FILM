# U2.5A B&W chroma-invariant contract

Date: 2026-07-17

Node: `ULT > U2 > U2.5 > U2.5A`

Status: frozen before implementation

## Defect and parent evidence

The U1.3B full-resolution RAW audit found sparse orange residuals around extreme
highlights in HP5 output. The old early-quantized and new float paths both show
the defect, so it is pre-existing safe-Lab behavior rather than a U1.3B
regression. On the audited RAW, channel spread above 20 codes affects 0.1233%
of legacy pixels and 0.1195% of float-path pixels.

Code inspection identifies three leakage paths: the B&W branch retains a small
fraction of Lab chroma; neutral/skin guards may blend source chroma back into
the target; and source-directed gamut compression may move an outlying target
back toward coloured source Lab. Per-channel internal grain/dither can also
break a final neutral-axis invariant.

## Required invariant

For styles `hp5` and `tri_x_400`, final RGB from `style_transfer_rgb` must be
achromatic before output quantization: R, G and B are equal at every pixel
within float tolerance and exactly equal after uint8/uint16 quantization.

This is a semantic invariant for B&W profiles, not a new aesthetic model. It
does not require colour styles to become less stylized and must not alter them.

## Frozen gates

1. Both B&W styles pass high-chroma, neutral-ramp and randomized RGB fixtures.
2. The invariant holds with configured guardrails, dither and nonzero internal
   grain, not only in a stripped-down test.
3. Quantized uint8 channels are exactly equal; float channel spread is at most
   `2e-6`.
4. A representative colour style is bit-exact before/after the patch on a
   frozen deterministic fixture.
5. Existing output bounds, deterministic seeds, format/ICC and U1.3B parity
   contracts remain intact.
6. Full CPU suite and full-resolution RAW HP5 visual smoke pass with no orange
   speckle, posterization, banding, clipping block or geometry corruption.

## Allowed implementation

- enforce a final neutral-axis projection only for `B_AND_W_STYLES` inside the
  existing explicit safe-Lab core;
- use a deterministic luminance-preserving scalar RGB value;
- retain current B&W tone/contrast, output margin and deterministic noise
  behavior as closely as the invariant permits;
- add focused property/regression tests in the existing colour safety suite.

No new module, model, dataset, dependency, style ID or production branch is
allowed. Do not change colour-style profiles or U1.3B evidence.

## Stop branches

- If colour-style output changes, reject the patch and isolate the projection.
- If neutral projection creates banding or a material tone regression, preserve
  the evidence and redesign the scalar projection before promotion.
- If the RAW speckle persists despite an exact core invariant, trace the later
  effect/export stage rather than loosening the invariant.

## Claim ceiling

At most: current explicit HP5/Tri-X look approximations produce deterministic
achromatic RGB without residual colour artifacts. This is not calibrated B&W
film response, developer simulation, spectral modeling or stock authenticity.
