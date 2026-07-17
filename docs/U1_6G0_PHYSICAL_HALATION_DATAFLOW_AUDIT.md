# U1.6G0 Physical/Density Halation Dataflow Audit

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6G0`

**Decision:** **close direct tiled-adapter shortcut** — do not wrap the current
physical/density layer in a nominal halo. Open only prerequisite global-field
primitives before reconsidering staged execution.

## Dependency inventory

The colour physical-halation path contains:

- a full-image linear-light luminance/log-exposure field;
- source percentile normalisation (default 99.7%);
- full-image gradients and colour/skin/source confidence fields;
- a background mean blur and a second blur of local absolute deviation;
- four red source blurs and two green source blurs;
- a second source percentile (99.8%);
- final visibility, exposure, energy, colour and alpha fields.

At default diffusion it invokes eight Gaussian fields. Sigma reaches 52 for
red glare. Density halation uses one percentile and six Gaussian fields, with
sigma reaching 70. These maxima enter the shape-dependent downsample branch of
`gaussian_filter_safe`: approximately 8x and 11x respectively before the
downsample factor is clipped by full-image dimensions.

## Decisive counterexample

A deterministic `257 x 389 x 1` float32 field was blurred at sigma 52. The
full path was compared with 64-pixel tiles using nominal halo
`round(3 * 52) = 156`, larger than any direct-kernel support requirement.

- tile count: 35;
- maximum expanded tile: `257 x 353 x 1`;
- maximum error: `0.0005808472633361816`;
- mean error: `0.00013795234553981572`;
- two-sided seam maximum: `0.0005808472633361816`;
- byte-identical: no.

The mismatch persists because each expanded tile chooses a different BOX
downsample shape/grid and BILINEAR reconstruction grid from the full image.
Adding nominal halo cannot make a shape-dependent global resample local.

## Identifiability and resource boundary

Percentiles also depend on all pixels and must be fitted once from a declared
full field. Chained fields such as `abs(background - local_mean)` require the
preceding global result before their own blur. Unlike U1.6F grain, there is not
one raw field plus one fixed-radius high-pass; there is a multi-stage field DAG
with several simultaneously live sources and different global resample grids.

Therefore a direct effect adapter would either:

- silently change legacy pixels and seam behaviour;
- hide multiple full-frame arrays;
- or duplicate the whole physical algorithm inside tiling code.

All three violate current contracts.

## Required prerequisites before implementation

Any later U1.6G1+ route must separately freeze and prove:

1. an exact or explicitly versioned staged percentile primitive;
2. a full-grid shape-stable BOX/downsample/direct-blur/BILINEAR operator whose
   tiled output is defined by original-image coordinates, not tile shape;
3. a scratch-field DAG/lifetime planner with declared disk and RAM peaks;
4. coordinate-exact gradients and boundary modes;
5. colour and density families as separate graphs;
6. legacy parity or a newly versioned effect with fresh visual/severe gates.

No physical-halation code, renderer path or threshold changed in U1.6G0.

## Claim and branch boundary

This audit does not reject physical/density halation as a visual effect. It
rejects only the unsupported claim that the current implementation is a finite
halo-local operator. U1.6C simple halation remains independently eligible.

Training, stock claims, renderer integration, physical calibration, streaming,
bounded total memory and 100MP readiness remain unchanged. U1.6 and Ultimate
continue through prerequisite dataflow or another legal product/data leaf.
