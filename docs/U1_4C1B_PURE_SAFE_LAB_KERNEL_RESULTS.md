# U1.4C1B Pure Safe-Lab Kernel Results

**Status:** pass

**Parent:** `ULT > U1.4 > U1.4C > U1.4C1`

**Contract:** `docs/planning/U1_4C1_WORKING_SPACE_LAB_KERNEL_CONTRACT.md`

**Implementation commit:** `12222bd877fbfaad0520ade1600ca63be379c700`

**Claim ceiling:** pure reusable Lab-domain look kernel with frozen legacy sRGB parity

## Change

- Added `src/color_engine/safe_lab.py` as the single maintained implementation
  of the safe-Lab statistics transfer, B&W Lab branch, chroma curve, luma-detail
  preservation, colour guardrails and tone rolloff.
- Moved the immutable full-image Lab statistics context into the colour-engine
  package.
- Kept RGB conversion, sRGB gamut compression, output margin, dither, grain and
  neutral-axis projection in the legacy adapter.
- Made `scripts/pipeline_color_baseline.py` delegate to the shared kernel without
  changing the existing public rendering path.

## Frozen evidence

| Gate | Result |
|---|---|
| Seeded float hash, dither 0 | exact `3229cf98e4691964b0101c9e2b0288d9fed27e04eac84a9aa6922c0bfa8bfd0a` |
| Seeded float hash, dither 0.35 | exact `10eea738bf9c673ffb017ff093699dcb8425dbb79d5e5a0f1b5650f3bb3a9d7a` |
| Eight safe-rich full/tiled styles | pass at frozen maximum/seam tolerance `1e-6` |
| Kernel determinism | byte-identical repeated output |
| Source mutation | none |
| Full-image context on a tile | pass |
| Invalid context/statistics | fail closed |
| Focused regression | 56 passed |
| Complete CPU suite | 600 passed |
| Diff/compile checks | pass |

No tolerance or historical configuration was changed after observing results.

## Decision

Retain the extraction. U1.4C1B establishes a colour-space-independent Lab
transformation core, but it does not establish destination-space gamut handling
or a Rec.2020 renderer. The next permitted child is U1.4C1C: freeze and implement
destination-working-space gamut tests/compression plus the isolated six-colour-
style Rec.2020 adapter and preregistered out-of-sRGB witness.

Grain, dither, output margin, HP5/Tri-X, FilmFX, production renderer/profile/
schema changes, HDR, ACES/OCIO, calibration and preference claims remain closed.
