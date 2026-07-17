# U1.6A Halo-Aware Tiling Primitive Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6A`

**Decision:** **pass** — retain the finite-support tiled-execution primitive and open only a separately frozen operator-locality/global-context audit.

## Delivered implementation

`src/inference/tiled_render.py` provides:

- immutable core/expanded `TileWindow` records;
- a pure row-major tile planner;
- strict HWC floating-point input validation;
- read-only expanded tile views;
- exact callback shape/dtype/finite validation;
- core-only stitching with no overlap blending;
- immutable execution metadata with the maximum expanded tile shape.

The primitive is exported from `src.inference`. It is not wired into `scripts/render_film.py` and does not duplicate any colour or effect algorithm.

## Formal committed evidence

The formal deterministic audit used commit `5fd5036d42d897dea8cdf18c29e6e1dc50e63fe4` and a seeded synthetic float32 array. Synthetic values are sufficient here because the question is exact execution topology and numerical parity, not image style or stock evidence.

| Field | Result |
|---|---:|
| Input shape | `257 x 389 x 3` |
| Core tile size | `64` |
| Halo | `6` |
| Tile count | `35` |
| Coverage min/max | `1 / 1` |
| Maximum expanded tile | `76 x 76 x 3` |
| Frozen maximum spatial side | `76` |
| Full-frame/tiled maximum absolute error | `0.0` |
| Tile-seam maximum absolute error | `0.0` |
| Repeat output bytes | identical |
| Repeat metadata | identical |
| Output SHA-256 | `f1e386c346ce0c19769b50359d37fbc04c107bc14d988162738e0fcae738bca8` |

The tested callback was the project's direct finite-support Gaussian path with `sigma=2.0`, `truncate=3.0` and `halo=ceil(sigma * truncate)=6`. This stays below the frozen `1e-6` parity and seam gates.

## Verification

- focused suite: **21 passed**;
- complete CPU suite: **269 passed**;
- irregular/non-divisible grids, single-pixel inputs, pointwise bit parity and oversized clipped halos pass;
- invalid rank, zero dimensions, integer input, NaN input, tile/halo values, callback rank/shape/dtype and NaN/Inf output fail closed;
- callback mutation of the input view is rejected by the read-only array contract.

## Claim boundary and next branch

This result bounds the transient callback window, excluding the full-frame input and output. It does not establish bounded total memory, streaming encode, cache behavior or complete renderer parity.

Current safe-Lab source statistics, percentile-normalized physical/density halation, full-frame grain normalization and coordinate-seeded dust contain global-state dependencies. They remain full-frame until a child audit defines deterministic two-pass context or global-coordinate randomness. U1.6 remains in progress and no 100MP product claim is allowed.
