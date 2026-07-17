# U1.6B Safe-Lab Global-Context Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6B`

**Decision:** **pass** — retain the experimental two-pass safe-Lab API; do not switch the default renderer or claim complete tiled rendering.

## Delivered implementation

The existing `scripts/pipeline_color_baseline.py` now contains:

- immutable `SafeLabSourceContext` with source shape, pixel count and full-image Lab mean/std;
- exact full-image context construction with the legacy `1e-3` std floor;
- one context-aware safe-Lab implementation shared by full-frame and tiled entrypoints;
- `legacy_uniform_dither_window`, which jumps PCG64 to exact image-space row offsets and allocates only the requested window;
- experimental `style_transfer_rgb_tiled`, which reuses U1.6A with halo `5` for luma-detail preservation and rejects nonzero legacy colour-core grain.

No CLI flag, renderer default, profile asset, statistics file, guardrail, effect algorithm or output claim changed.

## Compatibility evidence

Before implementation, the current algorithm was frozen on a deterministic 17x19 input. The refactored full-frame path reproduces both hashes exactly:

| Dither | Frozen/refactored SHA-256 |
|---:|---|
| `0.0` | `3229cf98e4691964b0101c9e2b0288d9fed27e04eac84a9aa6922c0bfa8bfd0a` |
| `0.35` | `10eea738bf9c673ffb017ff093699dcb8425dbb79d5e5a0f1b5650f3bb3a9d7a` |

Standalone `scripts/pipeline_color_baseline.py --help` also succeeds after preserving repository-root import behavior.

## Eight-style committed audit

At software commit `600c3bdc51a48946da8a9e8962363064424644f9`, all eight tracked safe-rich styles were evaluated on the same seeded 31x47 float32 input with tile size `11` and halo `5`.

| Style | Max error | Seam max | Byte-identical | Tiled SHA-256 |
|---|---:|---:|---|---|
| `ektar_100` | `0.0` | `0.0` | yes | `69936ec56694fe0535b8bab872c072b4719b1345f3db98b62fe77a95a145905b` |
| `hp5` | `0.0` | `0.0` | yes | `706928c95ad937d71ad2fa6d0187f04f7c1614a1005813da4d3532e3670c70c9` |
| `portra_400` | `0.0` | `0.0` | yes | `5fd3f8297639589bf7e281f02e99a6e8deb85939ae195eb5a0d3e883a6c7434d` |
| `portra_800` | `0.0` | `0.0` | yes | `4d7d49bd399885f2efe09983308df49b492c977dd69bfb039549731a254648dd` |
| `tri_x_400` | `0.0` | `0.0` | yes | `e071a1a8e0a640d96f4b1cdccaafc5830eb12c24d6f3066a6fd6da8addff3fc6` |
| `velvia_50` | `0.0` | `0.0` | yes | `4830b105f35762db9ea853a485124eaa12d8341170e7ef5f8b16e879d10b50a8` |
| `vision3_250d` | `0.0` | `0.0` | yes | `48d364f056ee0629184acaaec1ea9163900a97a8af75cd738218357b1d6cf7d4` |
| `vision3_500t` | `0.0` | `0.0` | yes | `10a4083e906ed1074c2fae59a5cc0326a1e5ee8fad48be151bf6eb2774cdb7d1` |

These are numerical compatibility results for heuristic profiles, not evidence that the names reproduce physical stocks.

## Real-raster mechanical smoke

The committed smoke used quarantined `data/film_domain/velvia_50/fl_76ef574fbe793892.jpg` only as decoded raster mechanics. Source SHA-256 is `11c546ede342a4e80cf75c71f7d9fcb31fa81950cd36627069076c747d6e5a29`. It supplies no label, rights, stock-style, preference or calibration evidence.

- original shape: `576 x 1024 x 3`;
- fixed crop bounds `(y0,y1,x0,x1)`: `(17,274,23,412)`;
- crop shape: `257 x 389 x 3`;
- tile size / halo / count: `64 / 5 / 35`;
- maximum expanded tile: `74 x 74 x 3`;
- float max error / seam max: `0.0 / 0.0`;
- float outputs: byte-identical;
- quantized sRGB8 outputs: byte-identical;
- sRGB8 SHA-256: `f0dcc1e27190714419681781d6eca6d7d8e7080ee7e59346509638d7299da772`.

## Verification and fail-closed evidence

- dedicated U1.6B tests: **15 passed**;
- targeted safe-Lab/U1.6A/renderer compatibility set: **55 passed**;
- complete CPU suite: **284 passed**;
- exact coordinate dither slices pass across full, interior, edge and one-pixel windows;
- repeated tiled output and metadata are identical;
- invalid/empty/non-finite RGB, inconsistent context, out-of-range dither windows, invalid tile size and nonzero colour-core grain reject;
- compile and diff checks pass.

## Branch and claim boundary

U1.6B closes as a numerical two-pass colour pass. U1.6 remains active.

The following remain full-frame or unresolved:

- legacy/residual grain global normal RNG and normalization;
- dust/scratch full-shape counts and coordinates;
- simple-halation radius/locality integration;
- physical/density halation percentiles and shape-dependent downsample approximation;
- streaming decode/encode, cache, bounded total memory and 24MP/100MP performance.

The default renderer therefore remains full-frame. No complete-renderer, 100MP, real-film, named-stock or calibrated claim opens.
