# U1.3D DNG-to-LibRaw metadata conformance contract

Date frozen: 2026-08-21  
Node: `ULT > U1 > U1.3D`  
Status: frozen before confirmation-file TIFF/LibRaw inspection

## Question

Which standardized DNG capture facts survive the current `rawpy`/LibRaw
inspection path with enough fidelity to be used by future explicit operators?

U1.3C remains the exact source-of-truth receipt. U1.3D is a conformance audit,
not a renderer change. It calls `rawpy.imread` for metadata but never calls
`postprocess`, `raw_image`, `raw_image_visible`, TIFF `asarray`, or any RGB
render/decode path.

## Role separation

The four U1.3C rows are consumed development diagnostics. Their observed
relationships may define comparison logic but cannot decide the result.

The mandatory confirmation rows below were selected from the already
hash/rights-frozen U5.R2BH1S manifest by fixed manifest order before any U1.3D
TIFF or LibRaw inspection:

1. `motorola_moto_g_7_play`
2. `lg_lg_h850`
3. `autel_robotics_xb015`
4. `xiaomi_m2010j19cg`
5. `blackmagic_pocket_cinema_camera_4k`

No replacement is permitted.

## Fixed comparisons

For each confirmation file:

- exact source SHA and U1.3C receipt construction must pass;
- LibRaw raw width/height must equal the selected DNG raw IFD dimensions;
- a uniform DNG WhiteLevel must equal LibRaw's scalar white level;
- DNG BlackLevel values are compared to LibRaw's integer per-channel values
  after nearest-integer projection, with at most one code absolute error;
- expected camera-WB multipliers are derived only from exact AsShotNeutral as
  `[G/R, 1, G/B]`; LibRaw's first three values are normalized by its green
  value and must have maximum relative error at most `1e-5`;
- LibRaw `visible_width/height` must match either the raw IFD dimensions or the
  exact DNG DefaultCropSize dimensions. The classification is recorded; it is
  not silently called a crop match; and
- all metadata must be finite and LibRaw inspection warnings must be empty.

If a DNG level is non-uniform in a way that cannot be mapped without CFA or
channel guessing, that row fails. No vendor/model special cases are allowed.

## Gates and stop rules

All five confirmation rows and every comparison above must pass. Forward and
reverse enumeration reports must be byte exact. Any missing file/hash, U1.3C
receipt failure, LibRaw warning, unsupported level shape, mismatch, non-finite
value, pixel access or replay difference closes the conformance claim.

Do not fix a failure by changing LibRaw parameters, rounding tolerance,
AsShotNeutral algebra, crop interpretation, source rows or thresholds.

## Claim ceiling

A pass establishes only that these exact standardized facts agree between DNG
and LibRaw on five held confirmation devices. It may open a separate opt-in
inspection-enrichment leaf. It does not establish raster decode, vendor render,
tone map, arbitrary-camera/DNG coverage, calibrated sensor noise, a public
schema/capability, film/stock identity or product admission.
