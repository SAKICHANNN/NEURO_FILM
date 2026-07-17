# U1.6D Sparse Dust/Scratch Tiling Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6D`

**Decision:** **pass for numerical execution only** — retain the experimental
sparse-context adapter; do not change the renderer default or claim realistic
film defects.

## Delivered implementation

`src/filmfx/tiled_effects.py` now provides:

- an immutable `DustScratchContext` with read-only compact int32 geometry and
  float32 alpha arrays;
- exact replay of the legacy PCG64 call order, event counts and clipping rules;
- exact global-coordinate alpha-window reconstruction;
- `composite_dust_scratch_tiled`, which intersects sparse events with zero-halo
  tiles and reuses the existing compositor and U1.6A executor;
- metadata for tile topology, event counts and context bytes.

The adapter requires finite float32 RGB and fails closed on invalid shapes,
strength, seed, margin, tile size or mismatched/mutable context. Context
integrity is checked once before tiled execution, not once per tile. No CLI,
profile, recipe, default renderer or legacy effect path changed.

## Exactness and bounded-context evidence

Four irregular/boundary variants reconstruct the legacy dense alpha layer
byte-for-byte. Four tiled composite variants, including zero strength, output
margin, overlaps, boundaries and a scratch longer than one tile, match the
legacy full-frame float result byte-for-byte. Repeated context construction and
execution are also byte-identical.

The context contains event geometry and alpha only. It holds no full-frame RGB
or alpha raster. The formal smoke used 179 specks plus four scratches in 3,660
bytes, compared with a 257 x 389 dense float32 alpha raster of 399,892 bytes.
This demonstrates sparse context for the tested shape; it is not an end-to-end
memory or 100MP performance result.

## Formal committed real-raster smoke

The smoke ran from implementation commit
`94e9e11ff769c1f6c3c16d422a2130dd4ebf976e`. It used the quarantined raster
only as decoded-image mechanics:

- source: `data/film_domain/velvia_50/fl_76ef574fbe793892.jpg`;
- source SHA-256: `11c546ede342a4e80cf75c71f7d9fcb31fa81950cd36627069076c747d6e5a29`;
- fixed crop `(y0,y1,x0,x1)`: `(17,274,23,412)`, shape `257 x 389 x 3`;
- base: full-frame heuristic safe-rich `velvia_50`, with colour-core grain off;
- dust parameters: strength `1.0`, seed `15`, output margin `4`;
- tile size / halo / count: `64 / 0 / 35`;
- maximum expanded tile: `64 x 64 x 3`;
- events/context: `179` specks, `4` scratches, `3,660` bytes;
- float full/tiled output: byte-identical, maximum error `0.0`;
- sRGB8 full/tiled output: byte-identical;
- sRGB8 SHA-256: `8f40d4c257c872aa9c0c29cf2efffd0c642db19c9c34550129ae0c809c08fd18`;
- active-effect evidence: `12,429` quantized channels change versus the base,
  with maximum float change `0.17914721369743347`.

The source contributes no stock, style, preference, rights or calibration
evidence.

## Visual adjudication

At stress strength 1.0, the inherited effect visibly produces square bright
specks and straight vertical scratches. These are identical in full and tiled
outputs, so tiling adds no seam, truncation or new glitch. However, the look is
plainly heuristic and can be conspicuous. U1.6D therefore does **not** promote
the effect's realism, strength policy or product default. A later product
quality leaf must separately judge or redesign defect shape and strength under
the severe-artifact veto.

## Verification

- dedicated and adjacent tiler/effect/renderer tests: **106 passed**;
- complete CPU suite: **328 passed**;
- compile and diff checks pass;
- legacy layer, CLI and default render paths remain unchanged.

## Branch and claim boundary

U1.6D closes as an exact sparse-execution pass. U1.6 remains active. Grain
normalisation, physical/density halation context, safe-Lab/effect/encode
integration, streaming/cache, total memory and 24MP/100MP performance remain
unresolved. No physical-defect, named-stock, calibrated or complete-renderer
claim opens.
