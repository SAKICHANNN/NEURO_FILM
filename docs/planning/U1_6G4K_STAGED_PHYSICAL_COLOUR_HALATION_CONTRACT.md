# U1.6G4K Staged Physical-Colour Halation Contract

**Date:** 2026-08-28

**Node:** `ULT > U1.6 > U1.6G4K`

**Status:** frozen / implementation ready

**Config SHA-256:** `9bb37a60ebcef011dbdcf838e6fa6d05e411409d4610d97eda69018f64320a07`

## Purpose

U1.6G3 froze the `physical-colour-v1` field graph and U1.6G4D proved its
gradient and row-staged global providers. U1.6G4E-G4J executed and integrated
only the separate density family. G4K fills the remaining colour-film
mechanics gap with a research-only staged executor for the unchanged default
`physical_halation_layer` controls.

This leaf is not a new film-stock model. It makes the existing red/green
backscatter operator executable without persistent full-frame derived scalar
fields; it does not promote that operator into the product or use Velvia 50 as
a proxy for all stocks.

## Fixed operator identity

Version: `staged-physical-colour-halation-v1-defaults`.

Input is finite float32 display-sRGB `HxWx3` in `[0,1]`, `H,W >= 2`. A separate
scene-linear source, alternative profile, or parameter expansion is outside
this leaf.

The fixed controls are the current `physical_halation_layer` defaults:
percentiles 99.7/99.8, limiter 2.0 stops, softness .42, gamma 1.45, local/global
diffusion 1.0/.18, CineStill-style red/green backscatter 1.0/.34, green hue
.28, background gain/target 1.25/.20, skin protect .55, impact .85 and alpha
cap .32.

## Frozen execution schedule

1. Validate source and execution geometry before public output allocation.
2. Resolve both source percentiles through bounded repeatable row factories.
3. Build only `local_mean`, `red_tail`, and `red_glare` as coarse global-grid
   contexts using the existing row-stage provider.
4. Traverse output tiles in row-major order. On halo-expanded windows compute
   exact coordinate gradients, `local_abs`, red near/mid, and green near/mid.
5. Reconstruct global contexts only for the requested original-coordinate
   window, crop tile cores, then evaluate visibility, channel exposure,
   normalized colour, and alpha in the legacy formula order.
6. Return a full required `FilmLayer` plus immutable resource metadata.

The input and final RGB/alpha are full-frame by contract. Hidden persistent
full-frame luma, source, edge, visibility, red, green, or blur scalar fields are
forbidden. Row/tile recomputation, two percentile histograms, and three coarse
contexts are allowed.

## Reference hierarchy

1. A materialized-v2 oracle uses the identical formulas plus G2/G4C/direct
   kernels on full derived arrays.
2. The current legacy `physical_halation_layer` is a compatibility control.
   Global-grid v2 is deliberately a different blur implementation, so legacy
   bytes need not be exact.
3. Identity/no-effect is context only.

## Frozen evidence and gates

The two synthetic cases and two pre-existing, hash-locked real inputs are
listed in `configs/u1_6g4k_staged_physical_colour_halation_v1.json`. No source,
threshold, parameter, or case replacement is permitted after first formal
execution.

- Staged versus materialized-v2 alpha, RGB, composite, and tile-seam maximum
  error must be `<=2e-6`; rounded sRGB8 composites must be byte-identical.
- Repeat float outputs and metadata must be exact for each frozen policy.
- Legacy alpha/RGB/composite maximum drift must be `<=7e-4`; mean alpha and
  composite drift `<=3e-5`; changed uint8 fraction `<=.005`; max code delta 1.
- Every derived source request must remain below full source height.
- Metadata must disclose input/output/context/histogram/source-reread/tile
  workspace bytes, with persistent derived full scalar bytes and scratch disk
  bytes both zero.
- Invalid shape/dtype/range/geometry, nonfinite providers, and injected provider
  failure must publish no partial layer.
- Confirmed severe seams, bands, blocks, clipping expansion, or unstable colour
  reject the leaf regardless of numerical proximity.

## Branches

- **Pass:** retain the private colour-family executor and stop; renderer/recipe,
  24MP resource, stock calibration, and product exposure each require a new
  independently frozen reason.
- **Numerical failure:** repair window/order logic without changing gates.
- **Legacy drift or severe failure:** close this executor version without
  threshold or cohort rescue.
- **Resource failure:** retain only bounded numerical evidence.

## Claim ceiling

A pass proves only deterministic execution of the existing default
physical-colour research operator on the frozen cases. It does not prove
physical calibration, a named-stock response, multi-stock distinction,
population preference, arbitrary-resolution production readiness, or product
promotion.
