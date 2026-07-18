# U1.6G1 Shape-Stable Global-Resample Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G1`

**Status:** frozen / implementation ready

## Purpose

U1.6G0 proved that the current large-sigma branch of
`gaussian_filter_safe` is not halo-local: every expanded tile chooses a
different BOX/downsample/BILINEAR grid. U1.6G1 freezes the smallest independent
primitive needed to remove that ambiguity. It does not integrate physical or
density halation.

## Versioned operator

The candidate is named `shape-stable-global-resample-v1` and has one immutable
plan per original image shape and requested spatial sigma.

1. Choose the downsample factor and coarse shape once from the original image.
2. Area-average the source field onto that one coarse grid. Source pixels are
   piecewise-constant unit cells; coarse cells use exact overlap weights.
3. Apply the existing direct float32 reflect Gaussian on the coarse grid, with
   sigma divided independently by the vertical and horizontal scale factors.
4. Reconstruct any requested original-coordinate window with one frozen
   half-pixel bilinear mapping and edge clamping.
5. A full render and every tiled assembly must call the same window
   reconstruction routine. Tile shape may not affect the plan or coordinates.

This is an explicitly new, versioned numerical operator. It is not required to
match Pillow's legacy BOX/BILINEAR approximation byte-for-byte. Existing
halation remains unchanged until a later visual and severe-artifact gate.

## Scope

Allowed:

- a reusable implementation under `src/filmfx/`;
- a public direct-Gaussian wrapper around the current tested implementation;
- immutable plan/stage metadata and focused CPU tests;
- synthetic mechanical fields, including the U1.6G0 `257x389`, sigma-52 case.

Forbidden:

- changing `gaussian_filter_safe`, current effect defaults, renderer or CLI;
- claiming legacy-effect parity, physical accuracy, calibrated film response,
  complete-renderer tiling, streaming input, bounded total memory or 100MP;
- hiding a full-resolution reconstructed field inside the staged context;
- training, operator fitting or opening any closed RF/SF/LSM branch.

## Interface and invariants

The implementation must expose an immutable plan, a read-only coarse stage,
strict original-coordinate window reconstruction, and full/tiled convenience
paths with auditable metadata. Inputs are finite float32 2-D or HWC fields.
Channel-axis blur is forbidden. Shape, dtype, version, finiteness, bounds and
plan/stage mismatches fail closed. Tiled assembly may allocate the final output
and one output window, but no hidden full-resolution intermediate.

## DoR

- U1.6G0 counterexample and prerequisite list are frozen;
- U1.6A tile-window planning is present;
- the branch is clean at pre-contract HEAD `c1ebb42`;
- no renderer, training or download process is active.

## DoD and gates

All gates are required:

1. the U1.6G0 `257x389`, sigma-52 case has full/tiled maximum and seam error
   exactly zero for at least two non-divisor tile sizes;
2. irregular 2-D and multi-channel fields, anisotropic sigma, identity/coarse
   edge cases and repeated execution pass;
3. tiled bytes equal full bytes for every frozen case;
4. plan and stage are invariant to tile size;
5. invalid dtype, shape, sigma, bounds, stage and plan combinations fail closed;
6. metadata reports version, tile count, maximum output-window shape, coarse
   bytes and output bytes;
7. focused tests and the complete CPU suite pass.

## Branches

- **Pass:** retain the independent primitive and open U1.6G2 percentile/DAG or
  effect-family staging; do not integrate effects automatically.
- **Numerical mismatch:** repair the coordinate implementation without relaxing
  exact full/tiled equality.
- **Unbounded intermediate:** close the implementation and retain the contract.
- **Visual regression after later integration:** keep current simple halation
  and reject the new effect version; this primitive may remain independently.

## Evidence and rollback

Contract, implementation and results use separate scoped commits. Required
evidence records the software commit, test command, frozen shapes/sigmas/tile
sizes, maximum/seam errors, stage/output byte bounds and claim ceiling. Each
commit is independently revertible.

## Claim ceiling

A pass proves only that a versioned low-frequency field can be staged on one
original-image grid and reconstructed window-by-window without tile-dependent
pixels. It does not prove old halation parity, physical realism, renderer
integration, stock authenticity, calibration, streaming decode or 100MP total
memory readiness.
