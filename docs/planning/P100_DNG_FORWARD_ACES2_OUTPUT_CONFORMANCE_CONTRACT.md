# P100 DNG ForwardMatrix to ACES 2 output conformance contract

## Question

Can the already-retained private P98 DNG ForwardMatrix raster path feed the
already-retained official ACES 2 WorkingImage adapter on all five exact P98
DNGs while preserving source identity, scene-linear input bytes, finite bounded
SDR/HDR output, target separation and process/order replay?

This is a mature explicit-operator engineering leaf. It is not another
automatic single-reference candidate, does not consume or reset the bounded
`2/3` candidate counter, and does not reopen P99 BaselineExposure after its
exact-boundary failure.

## Frozen inputs and information flow

- Use exactly the five source identities and byte hashes already bound by P98.
- Decode only through unchanged `load_dng_forward_working_image`.
- Apply only unchanged `apply_working_image_aces2_output` for
  `sdr_rec709` and `hdr_rec2020_pq`.
- Read no reference, target, preferred render, downstream application image or
  P99 exposure output.
- Write no decoded image or encoded output artifact. Formal output is metrics
  and array identities only.
- The generic LibRaw ACES photographic smoke is historical context, not a
  quality baseline: it uses a different camera-colour policy and cannot define
  truth for P98.

## Frozen gates

For all five rows:

1. source byte count and SHA-256 remain exact before and after execution;
2. the P98 WorkingImage remains finite float32 `linear_rec2020` /
   `scene_linear`, with its pixel SHA unchanged by both output transforms;
3. both ACES 2 outputs are finite float32, have the same shape as the P98
   input, and lie in `[0, 1]`;
4. SDR and HDR output hashes differ for every row;
5. the adapter output is byte-exact with an independently invoked official
   Rec.2020-to-ACEScg plus display/view processor chain for every row/target;
6. canonical and reverse row order reports are byte-exact;
7. two fresh-process reports are byte-exact.

Any failure closes this exact five-DNG composition. No clipping, gamut map,
tone curve, exposure rescue, row replacement, backend change, tolerance,
partition selection or target-specific exception is allowed.

## Claim ceiling

At most this leaf can establish a private exact-five-DNG Windows CPU
composition from P98 camera-linear LibRaw plus DNG ForwardMatrix to
scene-linear Rec.2020 and the pinned official ACES 2 SDR/HDR numerical outputs.
It cannot establish arbitrary DNG support, sensor/IDT calibration,
vendor/Adobe rendering parity, photographic or colorimetric quality, P99
BaselineExposure safety, tone mapping quality, HDR files or metadata, display
calibration, default loader/renderer integration, public package/schema/
capability, product, film, stock or preference admission.
