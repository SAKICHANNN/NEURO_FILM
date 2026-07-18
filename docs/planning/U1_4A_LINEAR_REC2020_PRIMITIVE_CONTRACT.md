# U1.4A Linear Rec.2020 Primitive Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.4 > U1.4A`

**Status:** frozen / implementation ready

**Config SHA-256:** `655906434bc92f4e0a4edab68f1a6588763b7f890cc47516ed61e5d241c10f62`

## Purpose

The current production path is linear-sRGB only. U1.4A establishes the first
validated wide-gamut mathematical primitive without installing a new colour
library or pretending the renderer already supports HDR/ACES.

The leaf adds dependency-free, explicit float32 conversions between
`linear_srgb` and `linear_rec2020`, both relative to D65. It also adds a
`WorkingImage` conversion helper that preserves source state and provenance.
The production renderer remains unchanged and accepts only linear-sRGB.

## Authorities and frozen math

- ITU-R BT.2020-2 defines the Rec.2020 primaries and D65 white:
  <https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en>.
- W3C CSS Color 4 publishes high-precision linear-sRGB/XYZ and
  linear-Rec.2020/XYZ matrices and explicitly performs the D65 conversion
  without chromatic adaptation:
  <https://www.w3.org/TR/css-color-4/#color-conversion-code>.

The config freezes the composed column-vector matrices derived from those
published rational matrices. The implementation must independently retain the
source XYZ matrices and verify composition rather than treating rounded values
as unexplained constants.

## Contract

- input and output are finite `float32` HxWx3 arrays;
- transform math uses float64 matrices and returns float32;
- no clamping, gamut mapping, transfer function or tone map is allowed;
- negative and above-one extended values are preserved;
- input bytes and writeability are unchanged;
- unsupported/nonlinear state fails closed;
- same-space conversion returns an independent copy;
- `WorkingImage` conversion preserves transfer/source/profile/HDR/orientation/
  alpha/bit-depth/path fields and adds one explicit conversion warning.

## Verification

- source matrices match the published W3C rational values;
- composed matrices match the frozen config within `5e-15`;
- forward/inverse matrix identity error is `<=5e-15`;
- sRGB primaries, white and Rec.2020 extended-gamut vectors match golden values;
- seeded extended-range float32 roundtrip error is `<=1e-6`;
- neutral-axis error is `<=2e-7`;
- no clamp, mutation or nonfinite acceptance;
- focused tests followed by the complete CPU suite.

## DoR / DoD and rollback

DoR: U1.2B/C pass; current head is
`3da2cd53a6a1a0641baecb58c96120861ccafe84`; worktree is clean; no relevant
download/training process exists; `PyOpenColorIO` and `colour-science` are not
installed and will not be added in this leaf.

DoD: versioned module/API, config-driven audit tests, WorkingImage preservation
tests, complete CPU suite, result document and tracker/log/state propagation.
Rollback is the scoped implementation commit; current renderer behavior never
changes.

## Branches

- **Pass:** retain the primitive; design a separate operator/renderer-space
  compatibility gate before production use.
- **Math/roundtrip failure:** correct the primitive; do not add approximations
  or clipping to force a pass.
- **Renderer pressure:** keep integration closed; U1.4A is not an authorization.

## Claim ceiling

Pass proves only a validated D65 linear-sRGB/linear-Rec.2020 conversion
primitive. It does not prove HDR/PQ/HLG, OCIO, ACES, gamut mapping, wide-gamut
file output, physical fidelity or production renderer integration.
