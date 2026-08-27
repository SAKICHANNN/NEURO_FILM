# P279 — HDR+ merged-DNG ingress safety contract

Status: **frozen before any P279 result-pixel decode**  
Parent: P269 exact Google HDR+ one-burst source lock  
Claim ceiling: private finite ingress/safety evidence only

## Question

Do the two already-retained official HDR+ `merged.dng` results enter the current
WorkingImage path without silently selecting the known 480x640 JPEG-compressed
metadata page instead of the full 4032x3024 burst result?

This is not an HDR reconstruction or quality experiment. It does not compare
the two HDR+ versions, fit an operator, read a fresh group, change the loader or
consume candidate 3.

## Frozen inputs

- P269 burst `0382_20150912_131056_666` only.
- `result_20161014/merged.dng`, 25,288,270 bytes, SHA-256
  `ab3a6bf3de83a3d9af62421af446fc2d0de1b3b9a563455f5ef6e412894624f2`.
- `result_20171023/merged.dng`, 24,885,864 bytes, SHA-256
  `3a21417ff7fba508c7fc13ca2cb758329e39ac914e55522f12f2acdae66bfbdd`.
- Parent P269 evidence SHA-256
  `2f9fc7a4875031b16d0488bcde7fe4e107bed03f80ba0147870e4b16e5da5505`.

P269 already established metadata-only facts: each file reports a 480x640x3,
uint8, JPEG-compressed page, while the source burst is 4032x3024. P279 freezes
the expected full-result dimensions as 4032x3024 before any new pixel read.

## Frozen execution and gates

Two fresh processes enumerate the two versions in forward/reverse order and
call the unchanged public `load_working_image` entry. Reports normalize order
and must be byte-exact.

For each file exactly one safe outcome is admissible:

1. explicit fail-closed rejection before returning pixels; or
2. an owned, finite, writable, C-contiguous WorkingImage of exactly 3024x4032.

Returning 480x640, another dimension, or an unowned/nonfinite image fails the
primary no-silent-thumbnail gate. Source bytes must remain exact, no network or
persistent pixel/media artifact is allowed, and a 64-byte truncation must
reject atomically. No alternate decoder, rawpy option, IFD selection, resize,
dimension tolerance or same-file rescue is allowed.

## Stop and claim boundary

Any gate failure closes this exact two-file/loader pair. A pass proves only
private ingress safety for these two official result DNGs. Neither outcome
opens arbitrary DNG, Google HDR+ quality, calibrated scene/display state,
package/schema/capability, product admission or candidate-3 change.
