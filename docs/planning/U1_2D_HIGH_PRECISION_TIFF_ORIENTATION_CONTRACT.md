# U1.2D high-precision TIFF orientation contract

## Question

Can the existing 16-bit TIFF ingress apply the standard TIFF/EXIF Orientation
values 1--8 without resampling or reducing sample precision, while preserving
all existing colour-profile and raster safety gates?

## Frozen scope

- Change only the existing single-page, three-channel `uint16` TIFF paths.
- Apply Orientation in encoded sample space before sRGB linearization or the
  supported ProPhoto-to-linear-Rec.2020 transform.
- Use only axis permutation and reversal. Interpolation, resizing, cropping,
  colour conversion and quantization are forbidden in the orientation step.
- Orientation mappings are frozen as: 1 identity; 2 horizontal mirror; 3
  180-degree rotation; 4 vertical mirror; 5 main-diagonal transpose; 6
  90-degree clockwise rotation; 7 anti-diagonal transpose; 8 90-degree
  counter-clockwise rotation.
- Reject missing-invalid values outside 1--8. Do not reinterpret arbitrary
  TIFF tags or change generic 8-bit, PNG, JPEG, RAW or HDR ingress.

## Frozen gates

1. All eight orientations exactly match an independent Pillow transpose oracle
   on a non-square, channel-distinct `uint16` fixture.
2. Source samples are preserved exactly up to the frozen axis permutation and
   reversal; no interpolation or quantization occurs.
3. sRGB and supported ProPhoto TIFF paths both produce the expected oriented
   geometry and colour-domain result.
4. Identity-orientation output remains byte-exact to the pre-change path.
5. Unsupported ICC, invalid orientation, multi-page, alpha and non-RGB16
   inputs retain their existing fail-closed behaviour.
6. A renderer subprocess consumes an Orientation=6 RGB16 TIFF and publishes
   the correctly rotated dimensions without changing the default product look.

## Stop rule and claim ceiling

Any oracle, precision, legacy or adjacent safety failure closes U1.2D without
metadata inference, tolerance relaxation or decoder replacement. A pass claims
only deterministic single-page RGB16 TIFF Orientation 1--8 ingress for the
already supported sRGB and ProPhoto profiles. It does not claim arbitrary TIFF,
PNG orientation, RAW camera orientation, HDR, multi-page, alpha, new ICC
support, stock calibration or a new film operator.
