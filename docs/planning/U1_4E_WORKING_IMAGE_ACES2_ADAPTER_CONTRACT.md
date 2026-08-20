# U1.4E WorkingImage to official ACES 2 adapter contract

## Question

Can the existing scene-linear `WorkingImage` boundary feed the pinned official
OpenColorIO 2.5.2 / ACES 2 output transforms without changing the global
working-space vocabulary or the default renderer?

## Frozen implementation

- Accept only finite float32 `WorkingImage` pixels in `linear_srgb` or
  `linear_rec2020` with `transfer_state == "scene_linear"`.
- Map those spaces through the exact built-in ACES CG config source spaces
  `Linear Rec.709 (sRGB)` and `Linear Rec.2020`, respectively.
- Convert source -> `ACEScg` with the official CPU processor, then invoke the
  unchanged U1.4D ACES 2 SDR or HDR output processor.
- Return a new contiguous float32 HxWx3 array. Do not mutate the input, clip,
  gamut-map, encode a file, or add `acescg` to `WorkingImage.working_space`.
- Reject display-linear/nonlinear/unknown inputs before processor execution.

## Frozen audit

The formal fixture contains deterministic extended linear RGB cube/ramp rows
for both supported source spaces. For each source and both output targets:

1. composed source -> ACEScg -> display output is finite and input-preserving;
2. packed and scalar official source conversion agree within `2e-6`;
3. the adapter and an independently constructed direct official display/view
   transform agree within `2e-5`;
4. neutral output ramps are nondecreasing and maximum neutral channel spread
   is at most `2e-5`;
5. unsupported working space, transfer state, dtype, shape and non-finite data
   fail closed;
6. canonical/reversed enumeration and two fresh process reports are byte exact.

Any gate failure closes this exact adapter. No threshold, color-space alias,
range, clipping or processor rescue is allowed on the scored fixture.

## Claim ceiling

A pass establishes only a private Windows CPU numerical adapter from the two
existing scene-linear `WorkingImage` spaces into the pinned official ACES 2
output transforms. It does not establish ACES certification, a camera IDT,
photographic quality, HDR file encoding or metadata, display calibration,
GPU/macOS parity, default-renderer integration, a public capability, film or
stock fidelity, or product admission.
