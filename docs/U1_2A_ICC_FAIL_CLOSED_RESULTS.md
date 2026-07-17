# U1.2A ICC conversion fail-closed results

Date: 2026-07-17

Decision: **pass — embedded ICC conversion failures now reject**

## Change

The ordinary raster path no longer catches a Pillow/LittleCMS embedded-profile
failure and continues with unprofiled RGB pixels. It raises before constructing
a `WorkingImage`; `render_film` therefore fails before output creation.

No-profile inputs retain the explicit assumed-sRGB warning. Valid profiles that
LittleCMS can convert still enter the existing sRGB/linear-sRGB path. Existing
high-precision PNG/TIFF profile restrictions are unchanged.

## Evidence

- malformed embedded ICC rejects for both PNG and JPEG;
- inspection still reports that the source contains an ICC profile, without
  pretending conversion succeeded;
- renderer subprocess produces no output for the malformed-profile fixture;
- ordinary assumed-sRGB, standard profiled SDR, PNG/TIFF16, RAW and U1.5A
  rejection tests remain green;
- 32 focused preprocessing/renderer tests pass;
- the complete CPU suite passes: 233 tests.

## Claim ceiling

This repairs colour-state/provenance honesty. It does not prove arbitrary ICC
support, introduce a wide-gamut working space, or provide OCIO/ACES, HDR or
calibrated Reference rendering.
