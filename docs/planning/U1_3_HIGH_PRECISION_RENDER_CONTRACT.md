# U1.3 opt-in high-precision renderer contract

Date: 2026-07-16

Node: `ULT > U1 > U1.3A`

Status: frozen before implementation

## Objective

Add an opt-in 16-bit PNG/TIFF render path that keeps sRGB-encoded RGB and the
existing deterministic safe-Lab transform in float32 until the final encoder.
The current 8-bit CLI path remains the compatibility default.

## DoR

- `WorkingImage` raster and generic RAW ingress exists;
- standard sRGB ICC and exact uint16 PNG/TIFF encoders pass pixel tests;
- safe-Lab is deterministic but currently quantizes both its input adapter and
  return value to uint8;
- generic RAW is explicitly linear-sRGB/scene-linear and carries an
  uncalibrated scene-to-display warning.

## Allowed changes

- extract a float-returning safe-Lab core while preserving the public PIL
  wrapper;
- add a float sRGB adapter for known `linear_srgb` scene/display-linear input;
- add explicit `--output-bit-depth 16` for `.png`, `.tif` and `.tiff` only;
- retain float32 through procedural layers and encode once at the end;
- record internal precision, output bit depth, ICC and claim provenance.

## Forbidden shortcuts

- putting an already uint8 result into a uint16 container;
- silently changing the default 8-bit compatibility path;
- allowing 16-bit JPEG;
- treating linear-sRGB RAW gamma encoding as calibrated tone mapping;
- widening to HDR, HEIF, wide gamut, Reference claims or a new renderer.

## DoD

1. the existing PIL safe-Lab wrapper remains pixel-equivalent to explicit
   quantization of the new float core;
2. default 8-bit render tests remain unchanged;
3. 16-bit PNG and TIFF E2Es decode to uint16, retain the exact sRGB ICC and
   contain more than 256 sample levels on a suitable fixture;
4. invalid 16-bit suffixes fail before output creation;
5. metrics say whether the legacy uint8 adapter was used and whether the
   internal color path stayed float;
6. targeted tests, full CPU suite and one real-image visual smoke pass.

This is product precision work only. It does not change stock evidence,
operator fitting, LSM eligibility or the `film-inspired/look-approximation`
claim ceiling.
