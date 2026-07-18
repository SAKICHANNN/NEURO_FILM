# U1.4A Linear Rec.2020 Primitive Results

**Date:** 2026-07-18

**Decision:** pass; retain as a validated wide-gamut mathematical primitive

## Evidence

- contract config SHA-256:
  `655906434bc92f4e0a4edab68f1a6588763b7f890cc47516ed61e5d241c10f62`;
- implementation commit: `bd5b193`;
- transform version: `linear-d65-srgb-rec2020-v1`;
- forward matrix SHA-256:
  `36fdc6ef41e6c7dcccaa46c6bc2652c1757f0270c3d8acb32bec9a22cd5add12`;
- inverse matrix SHA-256:
  `6eb56d4ebcb48b66aa0308f2123751c63675422479f19d6f602f7d34ca40a21c`.

The module composes the published D65 XYZ matrices rather than introducing an
unexplained colour transform. Observed gates:

| Check | Observed | Frozen gate |
|---|---:|---:|
| forward/config max abs | 0 | `<=5e-15` |
| inverse/config max abs | 0 | `<=5e-15` |
| inverse × forward identity max abs | `2.3593e-16` | `<=5e-15` |
| extended float32 roundtrip max abs | `2.3842e-7` | `<=1e-6` |
| neutral-axis max abs | 0 | `<=2e-7` |

The frozen seeded wide image retains values from approximately `-0.2133` to
`3.9691`, proving the transform does not silently clamp to `[0,1]`.

## API and ownership

`src/preprocess/color_management.py` provides:

- `linear_rgb_matrix`;
- `convert_linear_rgb`;
- `convert_working_image_space`.

Inputs must be finite float32 HxWx3 and explicitly linear. Same-space output is
an independent copy. WorkingImage conversion preserves source transfer/profile,
HDR metadata, orientation, alpha policy, bit depth and path; mutable metadata is
not aliased and one versioned conversion warning is added.

## Verification and propagation

- 53 focused colour/preprocess/renderer/claim tests pass;
- 576 complete CPU tests pass in 33.94 seconds;
- the production renderer, output encoder, schemas, effects, stock evidence and
  frozen experiments are unchanged;
- `docs/PROJECT_STRUCTURE.md` records the new module in the existing preprocess
  boundary.

## Claim ceiling

This proves a dependency-free D65 linear-sRGB/linear-Rec.2020 conversion
primitive. It does not prove HDR/PQ/HLG, OCIO, ACES, gamut mapping, wide-gamut
file input/output, renderer integration, physical calibration or stock style.
