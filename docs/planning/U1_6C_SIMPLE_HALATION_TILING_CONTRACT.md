# U1.6C Simple-Halation Tiled Composite Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6C`

**Status:** frozen / ready

## Role and parent evidence

U1.6A proves finite-support tiled execution. U1.6B proves byte-identical two-pass safe-Lab colour. U1.6C is the next effect-specific child: it asks only whether the current simple-halation fallback plus existing compositor can be executed tile-wise from its actual finite support.

Bounded reconnaissance on the unchanged code gives maximum full/tiled error `5.960464477539063e-08` and seam maximum `0.0` for an irregular 83x107 input, tile size 32 and halo 31. This observation selects the candidate but is not confirmatory evidence.

## Dependency proof

Current `halation_layer` uses:

- one-pixel luminance gradients;
- pointwise highlight/edge/scale weights;
- Gaussian scales ending at the resolved `max_radius`;
- `gaussian_filter_safe` with default truncate `3.0`.

For this path, the largest direct kernel radius is `round(3 * resolved_max_radius)`. The frozen required halo is therefore:

`1 + round(3 * resolved_max_radius)`

where:

- `resolved_min_radius = max(0.4, min_radius)`;
- `resolved_max_radius = max(resolved_min_radius + 0.1, max_radius)`.

The default `max_radius=10.0` requires halo `31`. The current maximum radius stays on the direct convolution branch because radius 30 is below `max_direct_radius=32`; no shape-dependent downsample approximation is involved.

## Goal and scope

Add `src/filmfx/tiled_effects.py` with:

- a pure required-halo helper mirroring the existing radius resolution;
- an experimental `composite_simple_halation_tiled` adapter;
- reuse of `halation_layer`, `composite_layers` and U1.6A execution;
- execution metadata from the generic tiled primitive.

The adapter returns the composited RGB image, not an independently tiled layer bank. It must not duplicate halation math.

## Non-goals

- no `scripts/render_film.py` integration or default switch;
- no physical/density halation support;
- no percentile, source-normalization or downsample-context solution;
- no change to halation parameters, profiles, presets or compositor semantics;
- no calibrated or physically validated claim;
- no total-memory, streaming or 100MP performance claim.

## DoR

- U1.6A and U1.6B pass;
- current branch is clean and no active renderer/test/download process exists;
- placement under `src/filmfx` preserves effect ownership while importing only the generic tiled primitive;
- rollback is one additive implementation commit;
- all existing full-frame paths remain available and unchanged.

## DoD and frozen gates

All gates are required:

1. required-halo calculation exactly matches the current radius clamping and direct Gaussian support;
2. default and at least three boundary/radius variants on irregular/non-divisible shapes have full/tiled maximum error and seam maximum `<=1e-6`;
3. one fixed real-raster mechanical crop has quantized sRGB8 byte parity;
4. repeated tiled composites are byte-identical;
5. metadata proves maximum expanded tiles remain within `tile_size + 2 * required_halo`;
6. invalid RGB, tile size, non-finite/out-of-range strength/radii/scale parameters and unsupported output margin fail closed before partial output;
7. existing halation, compositor, safe-Lab and renderer tests pass;
8. complete CPU suite passes.

## Forbidden fallback

- no overlap blending to hide seams;
- no manually enlarged full-frame context or hidden full-frame layer cache;
- no changing Gaussian truncate, radius or boundary mode to make tiling easier;
- no routing physical/density halation through this adapter;
- no weakening the `1e-6` gate after observing confirmatory outputs.

## Branches

- **Pass:** retain the experimental adapter; later integration must still freeze ordering with safe-Lab and output encode.
- **Parity failure:** repair halo derivation or close; do not alter the existing full-frame effect.
- **Radius enters downsample branch:** reject that parameter set and create a separate global-grid contract.
- **Real-raster quantization mismatch:** close promotion even if float metrics appear small.
- **Severe visual artifact:** reject the candidate/policy; tiling parity never overrides the global severe-artifact veto.

## Verification, commits and claims

Contract, implementation and result propagation are separate commits and pushes. Targeted tests precede the full CPU suite. Formal smoke runs only from the committed implementation.

A pass means only numerical parity of the current heuristic simple-halation composite. It does not establish physical accuracy, film-stock authenticity, complete renderer tiling or product performance.
