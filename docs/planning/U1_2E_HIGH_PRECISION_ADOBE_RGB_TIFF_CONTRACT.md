# U1.2E — High-precision Adobe RGB TIFF ingress contract

## Question

Can the existing RGB16 TIFF ingress accept a strictly identified
matrix-shaper profile compatible with Adobe RGB (1998), preserve all 16-bit
samples and TIFF orientation, and return an owned unclipped
`linear_rec2020` `WorkingImage` without weakening the unknown-profile gate?

## Frozen scope

- Single-page contiguous RGB16 TIFF only.
- Embedded RGB/XYZ ICC matrix-shaper profiles only.
- Shared one-parameter gamma fixed to `563/256` within ICC quantization.
- ICC colourants must match the Adobe RGB (1998) primaries adapted to the D50
  PCS; the media white tag may be D50 or D65 because both interoperable v2
  profile encodings exist.
- TIFF orientation 1–8 is applied through lossless axis operations before the
  colour transform.
- Decode encoded samples through the ICC gamma and colourant matrix, then the
  existing D50 PCS to linear Rec.2020/D65 transform. Do not clip or gamut map.

## Gates

1. A CC BY-SA ClayRGB v2 g22 profile and an independently generated
   compatible matrix-shaper profile both pass semantic identification.
2. Decoder output is exact to an independent float64 matrix oracle after the
   same final float32 conversion.
3. Orientation 6 is sample-exact before the colour transform.
4. Source sample multiset, source file bytes and embedded profile bytes remain
   unchanged.
5. Nonuniform TRCs, wrong primaries, wrong gamma, malformed ICC, alpha,
   non-RGB16 and multi-page inputs fail closed.
6. Existing sRGB16, ProPhoto RGB16 and U1.2D orientation outputs remain exact.
7. Two committed-head forward/reverse fresh-process reports are byte exact and
   leave no scratch files.

## Claim ceiling

This is only a private high-precision TIFF input compatibility boundary. It
does not distribute Adobe's ICC profile, implement arbitrary ICC transforms,
add PNG/RAW/HDR support, change a film look, or establish stock calibration.

