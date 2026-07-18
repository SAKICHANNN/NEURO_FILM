# U1.4B BT.2020 SDR PNG cICP Contract

**Status:** frozen before implementation
**Parent:** ULT > U1.4
**Risk:** R1 local, reversible
**Primary writer:** `dev-research-reliability`

Frozen config SHA-256:
`e14c0fd6eb72a97f34f37f104e7594d7a1514fcb8acc08482cb6556b51c83215`.

## Purpose

U1.4A proves reversible linear-sRGB/linear-Rec.2020 mathematics but no real file
boundary. U1.4B may add one standards-backed boundary: deterministic 16-bit RGB
PNG carrying full-range BT.2020 SDR samples and the PNG Third Edition `cICP`
signal. It must not fabricate an ICC profile or imply HDR/ACES support.

The frozen CICP tuple is `09 0F 00 01`:

- colour primaries 9: BT.2020;
- transfer characteristics 15: BT.2020-2 12-bit-system curve, used here as the
  functionally equivalent exact BT.2020 curve for 16-bit PNG sample storage;
- matrix coefficients 0: RGB identity, required by PNG;
- full-range flag 1.

## Definition of ready

- U1.4A is committed, tested and passed;
- PNG Third Edition defines `cICP`, requires it before `IDAT`, requires matrix
  coefficient 0 for PNG RGB and gives it highest colour-chunk precedence;
- H.273 identifies BT.2020 primaries as 9 and the BT.2020 transfer as 15;
- the existing OpenCV encoder can losslessly write uint16 RGB PNG;
- no dependency installation, external binary profile or schema migration is
  required.

## Allowed work

- explicit signed BT.2020 OETF/inverse helpers, with clipping only at file save;
- a deterministic PNG chunk parser/writer with CRC validation;
- exact recognition of the frozen `09 0F 00 01` tuple;
- 16-bit RGB ingress to `WorkingImage(linear_rec2020, display_linear)`;
- 16-bit RGB egress only from `WorkingImage(linear_rec2020, display_linear)`;
- focused golden, corruption, roundtrip, provenance and regression tests.

## Forbidden work

- no production renderer flag, render-profile/schema change or default change;
- no scene-linear save without an explicit scene-to-display transform;
- no alpha, animation, arbitrary ICC/CICP, TIFF, HEIF, AVIF, PQ or HLG support;
- no per-channel clipping inside U1.4A conversion math;
- no claim of display calibration, gamut mapping, ACES or HDR.

Unsupported or malformed CICP must fail closed before pixels are interpreted.
Existing sRGB PNG behavior must remain byte/provenance compatible.

## Definition of done

1. output contains exactly one valid `cICP` before `IDAT`, with payload
   `090f0001`, and contains neither `iCCP` nor `sRGB`;
2. repeat output is byte-identical;
3. stored uint16 samples survive decode exactly;
4. seeded linear Rec.2020 roundtrip max absolute error is at most `5e-5`;
5. a Rec.2020 primary witness remains outside sRGB by at least `0.05` after
   U1.4A conversion, proving the boundary did not silently narrow gamut;
6. malformed, duplicate, late or unsupported CICP; non-RGB16; alpha; and
   incompatible WorkingImage state fail closed;
7. focused tests and the complete CPU suite pass;
8. evidence and claim ceiling propagate before a scoped commit and push.

## Branches

- **Pass:** retain as a non-production wide-gamut file primitive; audit operator
  compatibility separately.
- **Decoder ambiguity or metadata loss:** close input support; output-only is
  allowed only if its own frozen gates pass and the claim is narrowed.
- **Non-determinism or sample corruption:** close U1.4B; do not add dependencies
  merely to force a pass.
- **Viewer inconsistency:** record as product compatibility risk; it does not
  alter the byte-level standards claim.

## Claim ceiling

At most: deterministic BT.2020 SDR 16-bit RGB PNG `cICP` ingress/egress
boundary. Renderer integration, HDR, display calibration and arbitrary profile
support remain open.
