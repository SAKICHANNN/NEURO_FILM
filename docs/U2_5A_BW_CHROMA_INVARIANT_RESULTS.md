# U2.5A B&W chroma-invariant results

Date: 2026-07-17

Decision: **pass — promote exact neutral-axis output for HP5 and Tri-X**

## Change

The existing explicit safe-Lab core now performs a final B&W-only projection
onto the display-sRGB neutral axis. It converts the current RGB result to
linear-light relative luminance, re-encodes that scalar to sRGB and repeats it
across R/G/B. The projection runs after deterministic internal grain, dither
and output margin, so those stages cannot reintroduce colour.

Colour styles do not enter this branch. No profile, style ID, dataset,
dependency, renderer module or product route was added.

## Evidence

- HP5 and Tri-X pass randomized, neutral-ramp and saturated-primary fixtures
  with configured guardrails, dither and nonzero internal grain;
- float maximum channel spread is below the frozen `2e-6` limit;
- uint8 and uint16 channels are exactly equal at every tested pixel;
- the frozen Velvia uint8 output remains bit-exact at SHA-256
  `72a7e30e1b3f9640ed764c8ddee40e6da028c8afda4e78dd40d88b2d27e85307`;
- 17 focused safety/ingress tests pass;
- the complete CPU suite passes: 225 tests.

## Full-resolution RAW smoke

The same 6024x4024 Sony ARW used to expose the defect renders through the
default float32 HP5 path to an ICC-tagged PNG8 with bounds [4,251]. All
24,240,576 pixels (6024x4024) have exactly equal R/G/B channels; automated count
of non-neutral pixels is zero. Autonomous visual review finds the prior orange
highlight speckles removed and no new posterization, banding, clipping blocks,
geometry corruption or objectionable tone discontinuity.

## Claim ceiling

This proves an implementation invariant for the current HP5/Tri-X
`film-inspired/look-approximation` profiles. It does not establish calibrated
B&W film response, developer behavior, spectral sensitivity or stock
authenticity.
