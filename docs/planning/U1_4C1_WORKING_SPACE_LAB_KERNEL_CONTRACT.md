# U1.4C1 Working-Space-Aware Lab Kernel Contract

**Status:** frozen before implementation
**Parent:** ULT > U1.4 > U1.4C
**Risk:** R1 local refactor with strict regression stop
**Primary writer:** `dev-research-reliability`

Frozen config SHA-256:
`e8dcf71c26e021dad6675068c4e355e5e89eecbf5939f967f26cef015982a18c`.

## Objective

Extract the colour-space-independent D65 Lab transform from the legacy
safe-Lab engine, preserve every frozen sRGB output, and add an isolated
colour-only path from display-linear Rec.2020 WorkingImage back to
display-linear Rec.2020 WorkingImage.

This is not renderer integration. U1.4C0 proves that existing effects and public
boundaries remain sRGB-specific.

## Definition of ready

- U1.4A validated D65 linear-sRGB/Rec.2020 conversion passes;
- U1.4B deterministic RGB16 PNG CICP boundary passes;
- U1.4C0 identifies the portable Lab subset and rejects direct wiring;
- existing frozen full-frame hashes and all-style full/tiled parity tests exist;
- no new dependency or schema migration is required.

## Required structure

Reusable Lab conversion/kernel code belongs in a dedicated `src/color_engine/`
package, not a second renderer or a research-script copy. The legacy script may
delegate to it only after exact frozen output is proved. Preprocess continues to
own WorkingImage and file I/O; FilmFX remains separate.

## Allowed implementation

1. D65 XYZ/Lab conversion for explicit linear sRGB and linear Rec.2020;
2. a pure Lab transform accepting source Lab, frozen context/statistics and
   existing safe-Lab parameters;
3. destination-working-space gamut tests and source/chroma binary compression;
4. an isolated Rec.2020 adapter supporting the six colour styles;
5. preservation of WorkingImage provenance without mutable aliasing;
6. tests and non-production diagnostic outputs.

The initial Rec.2020 path fixes grain, dither and output margin to zero and
forbids all FilmFX. HP5 and Tri-X remain excluded until a destination-space
neutral-axis contract is separately frozen.

## Frozen regression gates

- the existing seeded float outputs retain SHA-256
  `3229cf98...bfd0a` at dither 0 and `10eea738...9d7a` at dither 0.35;
- all eight legacy safe-rich full/tiled results remain within `1e-6`, with the
  current seam and determinism gates;
- complete existing renderer/profile/recipe behavior remains unchanged.

Any frozen sRGB mismatch stops the refactor; tolerances may not be widened after
seeing the result.

## Rec.2020 gates

- sRGB-to-Lab reference parity `<=3e-4` and Rec.2020 Lab roundtrip `<=3e-6`;
- result is finite and within Rec.2020 to `2e-6` after the selected gamut policy;
- at least one preregistered styled wide-colour witness remains outside sRGB by
  `>=0.02`, proving the path did not collapse through sRGB;
- repeated output is byte-identical and the source is not mutated;
- incompatible space/state and nonzero grain/dither/margin/effects fail closed;
- focused and full CPU suites pass.

## Forbidden shortcuts

- no Rec.2020-to-clipped-sRGB roundtrip presented as wide-gamut processing;
- no duplicate safe-Lab algorithm maintained in parallel;
- no direct encoded RGB generation model;
- no production CLI/profile/recipe/schema/default change;
- no FilmFX, HDR, ACES/OCIO, calibration or visual-superiority claim.

## Branches

- **Legacy parity fails:** revert extraction and close U1.4C1.
- **Lab math passes but styled witness collapses into sRGB:** retain conversion
  utilities only; reject the wide-gamut style operator.
- **Colour-only path passes:** retain as isolated research API and open a
  separate visual/OOD audit before any product integration.
- **Artifacts appear:** reject candidate regardless of gamut retention.

## Claim ceiling

At most: isolated colour-only working-space-aware safe-Lab research path with
frozen legacy parity. No effect, renderer, HDR, calibration or preference claim.
