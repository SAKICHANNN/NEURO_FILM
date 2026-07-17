# U1.6C Simple-Halation Tiled Composite Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6C`

**Decision:** **pass** — retain the experimental finite-support adapter; do not route physical halation or switch the renderer default.

## Delivered implementation

`src/filmfx/tiled_effects.py` adds:

- `simple_halation_required_halo`, which mirrors the current radius clamping and rejects the shape-dependent downsample branch;
- `composite_simple_halation_tiled`, which validates parameters, calls the existing `halation_layer`, composites through the existing `composite_layers`, and delegates core/halo execution to U1.6A.

The module contains no duplicate halation or compositor math. It is exported through `src.filmfx`, but no CLI/default/profile/preset path uses it yet.

## Locality and variant evidence

The helper returns halo `31` for the default `max_radius=10.0`: one gradient pixel plus direct Gaussian radius `round(3*10)=30`. It also reproduces the current resolved-radius clamp and accepts the largest tested direct radius `10.5` with halo `33`; a radius entering the downsample branch rejects.

Four irregular/non-divisible variants cover:

- default parameters;
- zero strength and a small three-scale radius;
- stronger halation with the minimum radius boundary and different radius gamma;
- lower highlight/edge thresholds, maximum accepted direct radius, nine scales and output margin.

Every variant has full/tiled maximum error and a two-sided tile-boundary seam maximum `<=1e-6`. Repeated execution is byte-identical and metadata respects `tile_size + 2*halo`.

## Formal committed real-raster smoke

The smoke ran at commit `a254b845880e9da7a5afd11470fbcf9a5cc500f7`. It used the same quarantined raster as U1.6B only for mechanics:

- source: `data/film_domain/velvia_50/fl_76ef574fbe793892.jpg`;
- source SHA-256: `11c546ede342a4e80cf75c71f7d9fcb31fa81950cd36627069076c747d6e5a29`;
- fixed crop: `(17,274,23,412)`, shape `257 x 389 x 3`;
- base: full-frame heuristic safe-rich `velvia_50` output;
- active smoke parameters: strength `0.3`, threshold `0.4`, edge threshold `0.02`, output margin `4`;
- tile size / halo / count: `64 / 31 / 35`;
- maximum expanded tile: `126 x 126 x 3`;
- full/tiled maximum error: `5.960464477539063e-08`;
- two-sided seam maximum: `5.960464477539063e-08`;
- quantized sRGB8 outputs: byte-identical;
- sRGB8 SHA-256: `d337fe9045b118cd4648221aa8a3ce634a8f482bb91ccde5ca3200c6a8aa5fee`.

This was an active-effect check: the composite changes 19,337 quantized channel values versus its base, with maximum float change `0.009850382804870605`. It is therefore not a zero-effect parity pass. The raster remains quarantined and contributes no stock, style, rights, preference or calibration evidence.

## Verification

- dedicated U1.6C tests: **21 passed**;
- targeted tiler/halation/safe-Lab compatibility set: **70 passed**;
- complete CPU suite: **305 passed**;
- invalid RGB, NaN/out-of-range strength, thresholds, radii, gamma, scale count, output margin and downsample-branch radius reject;
- compile/diff checks pass;
- existing full-frame effect and renderer paths remain unchanged.

## Branch and claim boundary

U1.6C closes as a numerical heuristic-effect pass. U1.6 remains active. Any later integration must separately freeze safe-Lab/effect ordering, encode behavior and recipe provenance.

Physical/density halation still depends on global percentiles and shape-dependent downsample approximations and is not eligible for this adapter. Grain, dust, streaming/cache, bounded total memory and 24MP/100MP performance also remain unresolved. No physical-accuracy, named-stock, complete-renderer or calibrated claim opens.
