# U1.4C1A Working-Space Lab Primitive Results

**Date:** 2026-07-18

**Decision:** pass as the first U1.4C1 implementation child; shared safe-Lab
kernel extraction remains open

## Evidence

- parent config SHA-256:
  `e8dcf71c26e021dad6675068c4e355e5e89eecbf5939f967f26cef015982a18c`;
- implementation commit: `9abc3083b16e1d06691a47191a48ac0b44b5d7fc`;
- focused tests: 33 pass;
- complete CPU suite: 593 pass in 34.01 seconds.

| Gate | Observed | Frozen gate |
|---|---:|---:|
| legacy sRGB/skimage Lab max abs | `0` | `<=3e-4` |
| extended linear-sRGB Lab roundtrip | `1.0729e-6` | `<=3e-6` |
| extended linear-Rec.2020 Lab roundtrip | `4.7684e-7` | `<=3e-6` |
| same-colour cross-space Lab max abs | `6.1035e-5` | `<=1e-4` internal guard |

The first implementation attempt used the official W3C sRGB XYZ matrix
directly and produced `0.02045` maximum Lab disagreement with the frozen
skimage convention. The frozen gate was not widened. The retained design uses
the exact legacy skimage sRGB PCS and maps Rec.2020 physical colours into that
same PCS through the validated U1.4A matrices. This preserves current style
statistics while keeping U1.4A's official conversion math independent.

## Structure and scope

`src/color_engine/lab.py` now owns unclipped D65 Lab conversion for explicit
linear sRGB and linear Rec.2020. It does not own WorkingImage, file I/O, FilmFX
or renderer orchestration. Existing safe-Lab code has not yet been delegated to
the module, so all frozen legacy output remains untouched at this child.

## Claim ceiling

Validated working-space-aware D65 Lab primitives only. This does not yet prove
a shared safe-Lab kernel, Rec.2020 style output, gamut handling, visual benefit
or production integration.
