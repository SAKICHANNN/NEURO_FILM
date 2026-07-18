# U1.4C1C Rec.2020 Safe-Lab Adapter Contract

**Status:** frozen before implementation

**Parent:** `ULT > U1.4 > U1.4C > U1.4C1`

**Parent config:** `configs/u1_4c1_working_space_lab_kernel_v1.json`

**Risk:** R1 isolated research API with strict legacy-regression stop

**Primary writer:** `dev-research-reliability`

## Objective

Add destination-working-space gamut checks/compression and one isolated
colour-only adapter from a display-linear Rec.2020 `WorkingImage` to another
display-linear Rec.2020 `WorkingImage`. This child consumes the C1A conversion
primitive and C1B pure Lab kernel. It is not production renderer integration.

## Frozen input and style boundary

- input is a finite, non-empty, float32 `WorkingImage` with
  `working_space=linear_rec2020` and `transfer_state=display_linear`;
- input samples must already lie in Rec.2020 `[0,1]` within `2e-6`; extended,
  scene-linear, display-referred, unknown and other-space inputs fail closed;
- allowed styles are exactly `ektar_100`, `portra_400`, `portra_800`,
  `velvia_50`, `vision3_250d` and `vision3_500t`;
- destination mean/std and every existing Lab-look parameter are explicit API
  inputs; the adapter does not load profiles or infer a stock;
- grain, dither, output margin, HP5/Tri-X and all FilmFX are absent from the API.

## Frozen gamut algorithms

- `source`: binary search on the source-Lab to styled-Lab segment in the
  destination working space;
- `chroma`: binary search from a same-L/hue neutral to the styled Lab value;
- both use 24 fixed iterations and the existing D65 Lab backend;
- source mode requires the source endpoint to be in destination gamut;
- chroma mode requires the same-L neutral endpoint to be in gamut and does not
  silently change luminance;
- a result outside destination gamut by more than `2e-6` fails closed;
- only final numerical excursions within `2e-6` may be clipped to `[0,1]`.

No sRGB conversion, sRGB gamut test or encoded-component operation is allowed
inside this path.

## Preregistered styled witness

The witness is a deterministic `12x16` display-linear Rec.2020 field. Its
channels are constructed from fixed horizontal/vertical ramps spanning
`R=0.02..0.70`, `G=0.20..1.00`, and `B=0.01..0.50`, including saturated
Rec.2020 green/cyan cells. Apply `velvia_50` with the tracked film statistics
and safe-rich colour parameters except that grain, dither and output margin are
zero, using `source` gamut mode.

After rendering, convert a diagnostic copy from linear Rec.2020 to linear sRGB
without clipping. The maximum excursion below 0 or above 1 must be at least
`0.02`. This diagnostic never becomes the render path.

## Gates

1. both legacy seeded hashes remain exact;
2. all eight legacy full/tiled style gates remain unchanged at `1e-6`;
3. both destination gamut modes produce finite Rec.2020 samples within `2e-6`;
4. the preregistered styled witness retains at least `0.02` linear-sRGB
   excursion;
5. repeated adapter output is byte-identical and input pixels are unchanged;
6. provenance fields and nested HDR metadata are preserved without mutable
   aliasing; a research limitation warning may be appended;
7. all incompatible state/space/style/gamut cases fail closed;
8. focused and complete CPU suites pass.

## Stop and branch rules

- legacy mismatch: revert C1C and close the adapter branch;
- witness collapses into sRGB: retain C1A/B only and close C1C;
- source or chroma policy cannot meet its frozen safety gate: reject that
  policy rather than widening tolerance;
- all gates pass: retain only as an isolated research API and open a separate
  visual/OOD audit before any renderer integration.

## Claim ceiling

At most: isolated colour-only Rec.2020 safe-Lab research adapter with frozen
legacy parity and a mathematical wide-colour witness. No visual superiority,
film-stock authenticity, calibration, HDR, ACES/OCIO, effect, product default,
renderer or user-facing wide-gamut support claim.
