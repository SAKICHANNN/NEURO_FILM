# U1.4B BT.2020 SDR PNG cICP Results

**Date:** 2026-07-18

**Decision:** pass; retain as an isolated non-production wide-gamut file boundary

## Frozen identity

- config SHA-256:
  `e14c0fd6eb72a97f34f37f104e7594d7a1514fcb8acc08482cb6556b51c83215`;
- implementation commit: `627a0d9678315742cbe72defa4a5af72c3bcfacf`;
- transfer version: `bt2020-2-oetf-v1`;
- CICP payload: `09 0F 00 01`.

The payload identifies BT.2020 primaries, the functionally equivalent
BT.2020-2 transfer characteristic 15, RGB identity matrix coefficients and
full-range samples. Output contains exactly one valid `cICP` before `IDAT` and
contains neither `iCCP` nor `sRGB`.

## Gate evidence

| Gate | Observed | Frozen requirement |
|---|---:|---:|
| repeat output | byte-identical | byte-identical |
| stored uint16 samples | exact | exact |
| seeded linear roundtrip max abs | `1.4424e-5` | `<=5e-5` |
| Rec.2020-green sRGB excursion | `0.5876` | `>=0.05` |
| cICP CRC/order/payload | pass | pass |
| unsupported/corrupt signalling | fail closed | fail closed |
| focused tests | 52 pass | pass |
| complete CPU suite | 584 pass in 35.01s | pass |

The deterministic seeded 11x13 evidence file was 953 bytes with SHA-256
`edd98b5dc722d334f41de8260be6105ed5eceaf584976078c1beecd8a349248e`.
That file is a generated diagnostic, not a committed product asset.

## Boundary behavior

- ingress recognizes PNG by signature rather than filename suffix;
- only single-frame, alpha-free, 16-bit RGB with the exact supported CICP tuple
  becomes `WorkingImage(linear_rec2020, display_linear)`;
- malformed CRC, duplicate/late CICP, unsupported tuples, 8-bit or alpha inputs
  fail closed;
- egress accepts only display-linear Rec.2020 WorkingImage and clips only at the
  explicit file boundary before BT.2020 transfer and uint16 quantization;
- existing sRGB encoders, renderer profiles, recipes and defaults are unchanged.

## Claim ceiling and remaining risk

This proves deterministic BT.2020 SDR 16-bit RGB PNG CICP ingress/egress. It
does not prove renderer compatibility, HDR/PQ/HLG, arbitrary ICC/CICP, TIFF,
OCIO/ACES, gamut mapping or calibrated display output. CICP-aware viewer support
is an external product-compatibility concern and remains to be tested before a
user-facing output option is considered.
