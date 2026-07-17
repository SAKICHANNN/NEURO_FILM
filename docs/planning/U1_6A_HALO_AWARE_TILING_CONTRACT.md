# U1.6A Halo-Aware Tiling Primitive Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6A`

**Status:** frozen / ready

## Question

Can the project add one deterministic, reusable tiled-execution primitive whose transient operator working set is bounded by tile size and whose stitched result is equivalent to full-frame execution for operators with an explicit finite spatial support?

This leaf does not claim that the complete renderer is already tile-safe or that 100MP end-to-end memory is bounded. Current safe-Lab colour statistics, percentile-normalized physical halation, full-frame grain normalization and coordinate-seeded dust have global-state requirements that must be adapted separately before product integration.

## Scope and placement

Add `src/inference/tiled_render.py`. The primitive owns only:

- strict HWC float-array validation;
- row-major core-tile planning;
- clipped image-space halo extraction;
- same-shape callback validation;
- core-region stitching;
- deterministic execution metadata.

It must not duplicate colour/effect algorithms, allocate hidden full-frame intermediates, infer an operator radius, or silently tile a callback that has undeclared global dependencies.

## Frozen interface

The implementation exposes:

1. an immutable tile-window record containing core and expanded image-space bounds plus core offsets inside the expanded tile;
2. a pure row-major planner for `(height, width, tile_size, halo)`;
3. a tiled executor accepting an HWC float array and a callback that returns an array with the exact expanded-tile shape;
4. execution metadata containing input shape, tile size, halo, tile count and maximum expanded tile shape.

The output array is full-frame by design. “Bounded” in U1.6A means the callback sees at most `(tile_size + 2 * halo)^2` pixels, excluding clipped borders, and the primitive creates no additional full-frame working image beyond its output. Streaming encode/cache is a later U1.6 leaf.

## DoR

- U1.3B float32 renderer is complete;
- input/output/profile contracts remain unchanged;
- no data, training, GPU, network or licence action is required;
- current global-dependency operators are explicitly excluded from integration.

## DoD and frozen gates

All gates are required:

1. planner covers every output pixel exactly once with no gap or core overlap;
2. identity and pointwise callbacks are bit-exact against full-frame execution across irregular shapes, one-pixel dimensions and non-divisible tiles;
3. direct finite-support Gaussian callback is numerically equivalent when `halo >= ceil(truncate * sigma)`, with maximum absolute error `<= 1e-6` and no visible seam peak above that tolerance;
4. repeated execution is byte-identical;
5. callback shape/dtype/rank violations, invalid tile/halo values and non-finite output fail closed;
6. metadata proves the expanded callback input never exceeds the frozen bound;
7. focused tests and the complete CPU suite pass.

## Forbidden fallback

- no automatic padding that changes full-frame boundary semantics;
- no overlap averaging that can hide an inadequate halo;
- no accepting NaN/Inf, shape mismatch or implicit RGB/channel changes;
- no claiming safe-Lab, grain, dust or percentile-normalized halation parity from primitive-only tests;
- no wiring the primitive into `scripts/render_film.py` in this leaf;
- no 100MP product or bounded-total-memory claim.

## Branches

- **Pass:** retain the primitive and open a separately frozen operator-locality/global-state audit before renderer integration.
- **Numerical seam or planner failure:** fix within the frozen interface and rerun; do not weaken tolerance.
- **Global-state mismatch:** keep that operator full-frame and design a two-pass/global-context contract in a child leaf.
- **Memory contract failure:** close U1.6A; do not disguise full-frame intermediates as tiled execution.

## Evidence bundle

- implementation and focused tests;
- planner coverage and maximum-window evidence;
- full-frame/tiled error and seam report;
- repeat hash;
- complete CPU-suite result;
- software commit and scoped propagation record.
