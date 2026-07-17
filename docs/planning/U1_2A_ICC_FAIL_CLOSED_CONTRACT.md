# U1.2A ICC conversion fail-closed contract

Date: 2026-07-17

Node: `ULT > U1 > U1.2 > U1.2A`

Status: frozen before implementation

## Defect

An 8-bit PNG with an invalid embedded ICC profile is inspected as profiled, but
`load_working_image` catches the LittleCMS failure, discards the profile and
continues via plain RGB conversion. The resulting `WorkingImage` still reports
an ICC source while its pixels were never colour-managed.

This is a proven provenance/pixel contradiction and violates the fail-closed
colour-state contract. High-precision PNG/TIFF already reject unsupported ICC;
the ordinary raster path must use the same epistemic boundary.

## Frozen policy

- no embedded ICC: retain the explicit assumed-sRGB warning and current decode;
- valid ICC convertible through the pinned Pillow/LittleCMS path: convert to
  sRGB and retain ICC provenance;
- malformed, unsupported or conversion-failing ICC: raise before returning a
  `WorkingImage` and before renderer output creation;
- never label fallback pixels as ICC-managed.

## Gates

- malformed embedded ICC rejects for PNG and JPEG;
- renderer subprocess creates no output for the malformed-profile fixture;
- unprofiled assumed-sRGB, standard sRGB ICC and existing high-precision ICC
  cases remain unchanged;
- valid non-sRGB profile behavior remains whatever LittleCMS can explicitly
  convert; no blanket wide-gamut claim is added;
- targeted and full CPU suites pass.

## Boundaries

This changes only error handling in the existing raster colour-conversion
boundary. No new profile engine, OCIO/ACES config, wide-gamut working space,
format, dependency or Reference claim is allowed.
