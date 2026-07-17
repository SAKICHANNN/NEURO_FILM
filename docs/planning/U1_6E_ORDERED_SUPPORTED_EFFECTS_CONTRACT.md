# U1.6E Ordered Supported-Effects Tiling Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6E`

**Status:** frozen / ready

## Parent evidence and ordering fact

U1.6A proves strict halo-aware core stitching. U1.6C proves finite-support
simple-halation parity, and U1.6D proves an exact sparse global dust context.
The current renderer derives all effect layers from the same colour-rendered
base, then composites them in order `grain -> halation -> dust`, clips after
each layer, and applies output margin only after the final layer.

U1.6E asks only whether the already eligible subset `simple halation -> dust`
can share one tiled pass without changing that ordering or base dependency.

## Frozen implementation

Extend `src/filmfx/tiled_effects.py` with one experimental ordered adapter that:

- accepts a finite float32 colour-rendered base;
- validates every parameter before tile execution;
- uses halo zero when simple halation is disabled and the U1.6C halo otherwise;
- builds one U1.6D sparse dust context before execution when dust is enabled;
- generates both enabled layers from each expanded **base** tile;
- composites simple halation before dust in one existing `composite_layers`
  call;
- applies output margin only in that final tile composite;
- reports shared tile topology, enabled stages and sparse-context counts/bytes.

The generic executor stays effect-agnostic. Effect math stays in the existing
generators. No intermediate full-frame effect output or dense dust raster is
allowed.

## Non-goals

- no grain or grain RNG/normalisation solution;
- no physical/density halation, percentile or downsample context;
- no safe-Lab orchestration, input decode, encode, CLI, profile or recipe path;
- no change to current renderer defaults or effect policy;
- no claim that the inherited dust/scratch shapes are realistic;
- no streaming, total-memory, 24MP/100MP or product-latency claim.

## DoR

- U1.6A/C/D pass with committed evidence and 328 CPU tests;
- branch is clean and synced at `804a3b5`;
- no renderer/test/download process or user/Cursor dirty files exists;
- rollback is one additive implementation commit;
- experimental adapter remains under `src/filmfx` beside its effect owners.

## DoD and frozen gates

All gates are required:

1. none, simple-only, dust-only and combined variants cover irregular shapes,
   non-divisible tiles, boundaries, non-default seeds and output margin;
2. full reference constructs enabled layers from the unchanged base and calls
   the existing compositor once in renderer order;
3. tiled float maximum and two-sided seam error are `<= 1e-6` for every
   variant, with dust-only and no-effect paths byte-identical;
4. sRGB8 reference/tiled bytes are identical; uint16 quantized difference is
   at most one code value;
5. a test that reverses the two active layers proves the order is observable,
   so accidental commutation cannot pass;
6. repeat execution and metadata are byte-identical;
7. metadata proves exact tile coverage, the frozen halo bound, enabled-stage
   order and compact dust bytes without a dense context;
8. invalid RGB, strengths, radii, thresholds, scale count, seed, tile size and
   output margin fail before partial output;
9. one committed real-raster active-both smoke passes numerical and visual
   inspection without a new seam/truncation artifact;
10. targeted compatibility and complete CPU suites pass.

## Forbidden fallback

- no sequential full-frame adapter calls or hidden intermediate full frame;
- no deriving dust or halation from the output of the other effect;
- no changing layer order, applying margin per layer or blending tile seams;
- no per-tile dust reseeding or dense alpha cache;
- no lowering the numerical gates after confirmatory output;
- no wiring unsupported effects through approximate tiling.

## Branches and claim ceiling

- **Pass:** retain one experimental supported-effects adapter; full renderer
  integration still requires a separately frozen orchestration/provenance leaf.
- **Order/parity failure:** repair orchestration only or close; do not alter
  established effect/compositor pixels.
- **Visual failure:** reject the tested policy even if numerical parity passes.
- **Unsupported effect requested:** fail closed; never silently run it tiled.

A pass proves only ordered numerical equivalence for the current heuristic
simple-halation and sparse dust subset. It proves neither realistic film
effects nor bounded end-to-end rendering.
