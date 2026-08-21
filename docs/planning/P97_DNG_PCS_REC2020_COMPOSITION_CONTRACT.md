# P97 — DNG PCS-to-Rec.2020 composition contract

Status: frozen before implementation or formal evaluation.

## Question

Can the already-qualified D50-PCS-to-linear-Rec.2020 segment be extracted from
the supported ProPhoto decoder as one reusable high-precision primitive,
without changing any existing decoder bytes, and compose deterministically
with all five P94 DNG camera-to-PCS matrices?

This is RAW colour-state plumbing for a private explicit operator. It does not
integrate a loader, decode a DNG raster, or rescue P95/P96.

## Frozen parents and regression oracle

- P94 evidence SHA-256:
  `28dbe488dac052ca374d35b883d9eaec93d3f2adb63223bc04ab87290c4ba355`.
- P94 report SHA-256:
  `f5a4aeb31070efcc7bddd9ff485dccadaa6b04e95d0251483c4bc1c24a1ae216`.
- Exact official ROMM profile SHA-256:
  `96b2f2987f83e2a545e607799fbfdff43ef8158fb9b215b187c574db8f145aaf`.
- Existing 65,536-triplet official-ROMM regression output SHA-256:
  `379c29f2a231c4297e34ac46abef05380b076739436ccd00c08d1f1bc58aa752`.

The existing Bradford D50-to-D65 and XYZ-D65-to-linear-Rec.2020 matrices are
unchanged. The new primitive may only factor those two existing operations out
of `decode_prophoto_rgb16_to_linear_rec2020`.

## Frozen evaluation

- Re-run the exact 65,536-triplet official-ROMM regression and require the
  frozen float32 output SHA.
- Compose each of the five P94 matrices with the exact 257 P96 camera probes,
  for 1,285 triplets total.
- Compare staged camera-to-PCS then PCS-to-Rec.2020 against the direct composed
  matrix with max absolute error `<=5e-15`.
- Require each metadata camera white to reach Rec.2020 unit white with maximum
  absolute error `<=1e-3`; this is a matrix-mechanics check, not calibration.
- Invalid shape/type/non-finite PCS inputs fail before output.
- Forward/reverse row order and two fresh reports are byte exact.
- DNG raster/sample/RGB reads remain zero.

Any regression, composition, validation, replay, binding, or white-mechanics
failure closes the exact primitive. No matrix, constant, tolerance, profile,
probe, row, or precision rescue is allowed.

## Claim ceiling

PASS means only a private reusable D50 PCS-to-linear-Rec.2020 matrix primitive,
exact existing ProPhoto decoder parity, and five-profile synthetic composition
mechanics. It does not establish arbitrary DNG support, DNG raster decode,
sensor/IDT calibration, scene-to-display tone mapping, photographic or
colorimetric accuracy, gamut mapping, loader integration, package/schema,
public capability, product, film, stock, or preference admission.
