# P94 — DNG dual-illuminant ForwardMatrix mechanics contract

Status: frozen before implementation and before any selected raster/sample decode.

## Question

Can a private, versioned metadata-only primitive reproduce the Adobe DNG SDK
1.7.1 dual-illuminant `ForwardMatrix` camera-to-PCS construction on every exact
three-channel DNG in the existing CC0 raw.pixls.us cohort that exposes both
forward matrices?

This is a bounded RAW engineering leaf.  It is not another after-only reference
matching candidate and does not evaluate image appearance, film style, stock
identity, sensor calibration quality, or product preference.

## Authoritative inputs

- Adobe DNG Specification 1.7.1.0, exact local SHA-256
  `abdecfd8e104e8b86cc054d3a40b677bb2ebe87131bfde080a1df3ee16eb2f8d`.
- Adobe DNG SDK 1.7.1 build 2652 archive, exact local SHA-256
  `73499b47f4683e12120a234bd0946f02e52ab2ff9834bcbd0e9f8ab4f923360e`.
- SDK equations and constants in `dng_color_spec.cpp`,
  `dng_camera_profile.cpp`, `dng_temperature.cpp`, and `dng_xy_coord.*`.
- Five exact existing CC0 DNGs: Motorola moto g(7) play, LG-H850, Xiaomi
  M2010J19CG, Blackmagic Pocket Cinema Camera 4K, and Huawei EML-L29.  Every
  row must expose `ColorMatrix1/2`, `ForwardMatrix1/2`,
  `CalibrationIlluminant1/2`, and `AsShotNeutral` before it is admitted.

The SDK's required attribution and DNG patent-notice obligations remain
binding.  Project code derived from these equations must retain an explicit
Adobe notice.  This contract does not broaden project licensing or authorize a
public release.

## Frozen implementation boundary

The primitive may:

1. read TIFF/DNG metadata tags without reading image samples;
2. support exactly three camera channels and exactly the standard A/D65
   illuminants (`17` and `21`), in either tag order;
3. use explicit `CameraCalibration1/2` and `AnalogBalance` when present;
4. use only the DNG-defined identity/ones defaults when those optional tags are
   absent;
5. normalize and interpolate `ForwardMatrix1/2`, solve the DNG neutral white,
   and construct the camera-to-D50 PCS matrix using the SDK equations.

It must reject unsupported illuminants, malformed/non-positive neutrals,
wrong matrix shapes, singular transforms, non-finite values, missing required
tags, nonmatching non-empty calibration signatures, and non-three-channel
profiles.  No fallback to LibRaw color output, generic sRGB, learned color,
external DCP profiles, raster decode, or heuristic matrix repair is allowed.

## Frozen gates

All gates are mandatory:

- exactly five admitted source rows and five distinct camera makes;
- exact source byte count and SHA-256 for every row;
- both forward matrices present for every row;
- all derived matrices/vectors finite and camera-to-PCS determinant nonzero;
- normalized forward matrices map camera one to D50 XYZ with max absolute
  error at most `1e-12`;
- derived camera-to-PCS maps its reconstructed reference neutral to D50 XYZ
  with max absolute error at most `1e-10`;
- camera-to-PCS inverse roundtrip max absolute error at most `1e-10` on the
  fixed identity-plus-six-axis probe;
- neutral-to-xy iteration terminates in at most 30 passes and repeated
  construction is exactly deterministic;
- forward/reverse row enumeration produces the same canonical scientific
  payload;
- two fresh processes produce byte-identical canonical reports;
- raster/sample/RGB decode calls equal zero.

Any mechanics, source, or replay failure is fail-closed.  There is no matrix,
epsilon, illuminant, cohort, or tolerance rescue on these rows.

## Claim ceiling

PASS means only a private, metadata-only, five-device DNG 1.7.1
ForwardMatrix camera-to-D50 PCS mechanics result.  It does not establish
arbitrary-DNG support, real sensor/IDT calibration accuracy, demosaic or render
quality, ACES output, default-loader behavior, public API/schema/package,
capability, product admission, film identity, or stock authenticity.

