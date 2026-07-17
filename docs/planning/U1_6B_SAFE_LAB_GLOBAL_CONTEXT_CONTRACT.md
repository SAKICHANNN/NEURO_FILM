# U1.6B Safe-Lab Global-Context Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6B`

**Status:** frozen / ready

## Document role

This is the executable child contract opened by the U1.6A pass. It classifies the current renderer's locality dependencies and freezes the smallest two-pass path that can be proved without changing the production CLI or overstating 100MP readiness.

## Current state

- branch `research/fivek-auto-optimize-cache` is clean at `17b7e01`;
- no project download, pytest or renderer process is active;
- U1.6A provides exact finite-support halo planning and stitching;
- `scripts/render_film.py` still executes the complete renderer full-frame;
- current source pools remain closed for learning, and this numerical product leaf opens no fitting/training/LSM action.

## Operator dependency audit

| Stage | Dependency | U1.6B decision |
|---|---|---|
| safe-Lab RGB/Lab conversion, transfer, guardrails, gamut and tone | pointwise after source statistics | eligible |
| safe-Lab source Lab mean/std | full-image reduction | eligible through one explicit prepass context |
| `preserve_luma_detail` | Gaussian `sigma=1.1`, finite spatial support | eligible with frozen halo `5` |
| safe-rich dither | legacy full-frame PCG64 uniform sequence | eligible only through exact global-coordinate sequence replay |
| legacy colour-core grain | global normal RNG stream | forbidden in tiled path; current safe-rich profile uses zero |
| simple halation | gradients plus multiscale finite blur | defer to a separate effect-radius leaf |
| physical/density halation | global percentiles and shape-dependent downsampled blur | forbidden until a separate global-context/algorithm contract |
| residual grain layer | global RNG, blur, mean/std normalization | forbidden until coordinate RNG plus reduction contract |
| dust/scratch | full-shape counts and coordinate RNG | forbidden until a global-coordinate generator contract |
| compositor | pointwise once layers exist | eligible later; not integrated here |

This classification is based on the current code, not an assertion about the intended physics of future effects.

## Goal

Add an experimental two-pass safe-Lab API that:

1. computes immutable full-image source Lab statistics exactly once;
2. applies the existing safe-Lab colour operator to expanded tiles using that context;
3. uses U1.6A core stitching and a halo of `5` when luma-detail preservation is active;
4. reproduces the existing full-frame dither sequence by image-space coordinates without allocating a full-frame noise field;
5. refuses nonzero legacy colour-core grain;
6. leaves every existing CLI/default call on the full-frame path.

## Scope and placement

The safe-Lab implementation currently lives in `scripts/pipeline_color_baseline.py` and is imported by the renderer. U1.6B uses touch-and-normalize within that file rather than creating a second colour engine or making a lower-level `src` module import a script.

Allowed changes:

- add immutable source-context and coordinate-dither helpers beside the existing safe-Lab implementation;
- refactor the existing full-frame function through the same context-aware core only if frozen output compatibility remains exact;
- add an explicit `style_transfer_rgb_tiled` experimental function;
- add focused tests under `tests/`.

Forbidden changes:

- no new CLI flag or default-path switch;
- no modification of profile/statistics/guardrail assets;
- no renderer/effect integration;
- no change to frozen experiments, data gates or claims;
- no full-frame dither/noise cache disguised as bounded tiling.

## Frozen interface and invariants

The implementation must expose:

- an immutable context containing full-image shape, pixel count and finite three-channel Lab mean/std;
- a context constructor accepting only finite HxWx3 float RGB;
- a context-aware internal colour application path;
- `style_transfer_rgb_tiled(..., tile_size=...)` using the existing U1.6A executor.

Context std uses the existing floor `1e-3`. The tiled entrypoint computes its own context from the exact input; callers cannot inject unrelated statistics. Nonzero `grain` fails closed. Required halo is `5` when `preserve_luma_detail_strength > 0`, otherwise `0`.

Coordinate dither must reproduce the current `np.random.default_rng(seed + 1009).uniform(...)` row-major float64-to-float32 stream at every `(y, x, channel)`. Each tile may allocate only its expanded noise window. The legacy full-frame path's pixels must remain unchanged.

## DoR

- U1.6A pass and implementation are present;
- current safe-rich profile identity and asset hashes remain unchanged;
- rollback is the independent U1.6B implementation commit;
- tests cover the existing full-frame path and the new experimental path;
- no active process or user dirty file is present.

## DoD and frozen gates

All gates are required:

1. existing frozen full-frame regressions remain byte-identical;
2. full-frame context refactor output is bit-exact to the pre-refactor algorithm for zero and nonzero safe-rich dither;
3. all eight tracked safe-rich styles on an irregular finite synthetic image match tiled output at maximum absolute error and seam peak `<= 1e-6`;
4. at least one non-divisible real raster mechanical smoke has full/tiled quantized sRGB8 byte parity; it provides no style/stock/right evidence;
5. dither coordinate windows exactly equal slices from the legacy full-frame sequence across borders and irregular windows;
6. context, image, tile size and nonzero-grain violations fail closed;
7. repeated tiled execution is byte-identical and metadata stays inside the U1.6A expanded-window bound;
8. focused tests and the complete CPU suite pass.

## Branches

- **Pass:** retain the experimental safe-Lab two-pass API; open a separately frozen integration/streaming leaf or effect-specific context leaf.
- **Colour parity failure:** repair within this contract; do not loosen `1e-6` or alter style assets.
- **Dither incompatibility:** keep dither/full renderer full-frame and close tiled safe-rich promotion rather than changing historical pixels.
- **Global dependency leak:** fail closed and keep the affected stage full-frame.
- **Memory-contract failure:** close the candidate; do not precompute a hidden full-frame noise or Lab buffer beyond the explicit prepass input conversion already required by the current renderer.

## Verification, commit and rollback

- commit/push this contract before implementation;
- targeted context/dither/tile tests first;
- existing colour/renderer/output regression tests next;
- complete CPU suite after targeted pass;
- commit/push implementation before formal committed smoke;
- propagate results only from committed-code evidence;
- each commit can be reverted independently without modifying data or historical experiment outputs.

## Claim ceiling

A pass proves only numerical equivalence of an experimental two-pass safe-Lab colour path with bounded per-tile callback/dither windows. It does not prove complete-renderer tiling, bounded total memory, streaming encode, physical-effect parity, 100MP performance, real-film style, stock accuracy or calibration.
