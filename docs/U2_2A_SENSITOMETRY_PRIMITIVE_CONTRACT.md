# U2.2A monotone sensitometry primitive contract

Date frozen: 2026-07-24  
Node: `ULT > U2.2A`  
Parent: `U2.2` monotone exposure/sensitometry curves

## Purpose

Create the missing explicit bridge between nonnegative scene/display-linear
channel exposure and film-like layer density. The primitive must represent
toe, straight-line and shoulder behaviour without hiding exposure, white
balance, clipping or an RGB display interpretation inside a generic spline.

This is a clean-room numerical primitive. The frozen witness is descriptive
and is not fitted to a named stock. Manufacturer characteristic curves remain
priors/constraints, not end-to-end RGB targets.

## Representation

### Exposure encoder

For linear channel value `x >= 0`, reference value `r > 0` and explicit black
offset `b > 0`:

```text
h = log10((x + b) / (r + b))
x = (r + b) * 10**h - b
```

The offset makes zero finite while preserving an analytic inverse. Negative
linear input and inverse values below the zero-exposure boundary fail closed;
there is no max/epsilon clamp.

### Characteristic curve

Each layer is a strictly increasing rational-quadratic spline from log
exposure to density with positive analytic derivative and linear tails. A
declared neutral anchor `(h=0, density=1)` is mandatory and verified, fixing
the curve's exposure/density gauge.

### RGB layer operator

Three independently versioned anchored curves map linear R/G/B exposures to
three layer densities. Forward, inverse, diagonal Jacobian determinant,
canonical JSON replay and explicit intermediate log exposure are required.
The output is `layer_density`, not display RGB; negative-film inversion, dye
mixing, print interpretation and output gamut are deliberately separate.

## Frozen descriptive witness

The fixed clean-room curves have one shared neutral anchor but different
toe/shoulder shapes. The audit uses 16,384 generated RGB samples across
`[0,16]` plus exact boundary/anchor probes and requires:

- forward/inverse max absolute RGB error `<=1e-11`;
- encoder forward/inverse max error `<=1e-13`;
- neutral anchor density error `<=1e-12` for all layers;
- minimum Jacobian determinant strictly positive;
- minimum scalar curve derivative strictly positive;
- toe and shoulder derivative each lower than the midscale derivative for all
  three layers;
- maximum pairwise layer-density separation away from the neutral anchor at
  least `0.05`, proving the witness is not one replicated curve;
- JSON replay byte-exact output;
- negative exposure and below-bound inverse fail closed;
- two complete result hashes identical.

## Engineering boundary

Reusable pure math belongs in `src/color_engine/sensitometry.py`. Formal
evaluation belongs in `src/eval`; the CLI stays under `scripts/`. Existing
roll2film splines may be reused, but their schemas and behaviour cannot be
changed. No renderer, profile schema, stock data, frozen experiment or default
output may be modified.

## Decisions

- **Pass:** U2.2A numerical primitive closes and may support a separately
  frozen U2.2B composition/interpretation audit. It does not by itself complete
  U2.2 product integration.
- **Fail:** preserve the result and keep U2.2 pending; no more capacity or
  relaxed gate without a distinct hypothesis.
- **Invalid:** repair only implementation/provenance contract failures and
  rerun unchanged gates.

## Claim ceiling

Deterministic numerical validation of an uncalibrated monotone
linear-exposure-to-layer-density sensitometry primitive.

