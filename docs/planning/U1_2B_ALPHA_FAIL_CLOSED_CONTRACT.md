# U1.2B Raster Alpha Fail-Closed Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.2 > U1.2B`

**Status:** frozen / implementation ready

**Config SHA-256:** `8d5274a383f290aacaa837fac079c257040e6745f3c9c0cdbf29d02779d755f0`

## Defect

The current raster loader converts RGBA/LA images to RGB, discarding alpha,
but returns `alpha_policy="preserved"`. Palette transparency is also silently
discarded. The three-channel `WorkingImage` contains no alpha plane, so this
claim is false. Hidden RGB beneath transparent pixels can enter colour and
effect processing without an explicit matte decision.

## Frozen policy

- Inputs without alpha decode unchanged.
- An alpha-bearing raster whose resolved alpha is exactly 255 everywhere may
  strip that redundant channel, return `alpha_policy="absent"` and record the
  `opaque_alpha_discarded` warning.
- If any resolved alpha is below 255, loading fails before RGB conversion with
  an explicit message that preservation/compositing is unimplemented.
- RGBA, LA and palette/`tRNS` PNG fixtures are required.
- No background matte, alpha output, CLI option, HDR, wide-gamut or schema
  expansion is added in this leaf.

## DoR / DoD

DoR: U1.2A and U1.5A fail-closed foundations pass; G4I is closed; the renderer
still emits RGB only; current head is `723ba75bd067f48257229258251f09a0759aece3`
and the worktree is clean.

DoD: focused inspection/load/hidden-RGB/opaque-equivalence tests; no
`preserved` claim can be emitted without stored alpha; ordinary SDR regression;
full CPU suite; result propagation; scoped commit/push.

## Branches

- **Pass:** retain honest fail-closed alpha ingress; a future explicit matte or
  alpha-preserving output is a separate product contract.
- **Fixture mismatch:** repair inspection/decode agreement before changing
  policy.
- **Regression:** do not weaken rejection; diagnose the affected SDR path.

## Claim ceiling

U1.2B can establish only honest alpha handling for the current SDR RGB raster
ingress. It cannot claim alpha preservation, correct creative compositing,
HDR/gain-map support, wide gamut, calibrated Reference or complete colour-state
support.
