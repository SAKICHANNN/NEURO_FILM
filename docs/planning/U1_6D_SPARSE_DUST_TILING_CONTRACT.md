# U1.6D Sparse Dust/Scratch Tiled Composite Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6D`

**Status:** frozen / ready

## Role and parent evidence

U1.6A proves deterministic halo-aware execution, U1.6B proves two-pass
safe-Lab parity, and U1.6C proves the current simple-halation composite can be
tiled. U1.6D asks whether the existing procedural dust/scratch layer can be
represented as a small deterministic event context and composited tile-wise
without allocating full-frame RGB and alpha layers.

This leaf is selected ahead of grain because the current grain path combines a
variable-consumption normal RNG, finite blur and full-image mean/std
normalisation. Pretending that grain is coordinate-local would change its
sequence or hide a full-frame context. Dust/scratch instead consists only of
sparse axis-aligned rectangles whose overlaps use `maximum`.

## Existing dependency contract

For an image `(height, width, 3)`, the unchanged generator uses one PCG64
sequence and creates:

- `max(1, int(height * width * 0.00018 * strength * 10.0))` specks;
- `max(0, int(width * 0.012 * strength))` scratches;
- speck rectangles of radius one or two pixels;
- scratch rectangles two pixels wide after boundary clipping;
- per-event alpha sampled from the existing ranges and multiplied by strength;
- order-independent `maximum` overlap, followed by clipping to `[0, 0.25]`;
- an all-white alpha layer composited with the unchanged alpha compositor.

The procedural event table is global context, but event count is sparse. It
must contain geometry and alpha only, never a full-frame raster.

## Goal and scope

Extend the existing `src/filmfx/tiled_effects.py` module with:

- an immutable `DustScratchContext` containing source shape, strength, seed and
  compact typed event arrays;
- a builder that replays the exact existing RNG call order and rectangle
  clipping rules;
- an experimental `composite_dust_scratch_tiled` adapter;
- reuse of `composite_layers` and the U1.6A executor with halo zero;
- auditable tiled-execution metadata plus sparse-context counts/bytes.

The implementation may factor a shared private event generator only if the
legacy `dust_scratch_layer` remains byte-identical for every frozen test case.

## Non-goals

- no CLI/default/profile/recipe switch;
- no change to dust count, geometry, alpha distribution, seed or compositor;
- no new physical-film or scanner-defect claim;
- no grain solution and no change to grain RNG/normalisation;
- no physical/density halation work;
- no complete renderer, streaming pipeline, 24MP/100MP latency or total-memory
  claim.

## DoR

- U1.6A-C pass and the branch is clean at `911e396`;
- no active renderer, test or download process exists;
- event context belongs beside the existing effect adapter, with the generic
  tiler remaining effect-agnostic;
- rollback is one additive implementation commit;
- full-frame and CLI paths stay unchanged.

## DoD and frozen gates

All gates are required:

1. context event counts exactly match the existing formulas for at least four
   irregular shapes, boundary strengths and multiple seeds;
2. event geometry/alpha reconstruct the legacy full-frame layer byte-for-byte;
3. tiled composites match full-frame float output byte-for-byte, including
   cross-tile specks, long scratches, overlaps and image boundaries;
4. one fixed real-raster active-effect smoke has sRGB8 byte parity and changes
   nonzero output values versus its base;
5. repeated context construction and tiled execution are byte-identical;
6. metadata reports halo zero, exact tile coverage, bounded expanded tiles,
   event counts and compact event bytes; no full-frame layer exists in context;
7. invalid RGB/shape, non-finite or out-of-range strength, invalid seed,
   context mismatch, tile size and output margin fail closed before partial
   output;
8. existing dust, compositor, tiler, safe-Lab, halation and renderer tests pass;
9. complete CPU suite passes.

## Forbidden fallback

- no per-tile reseeding or changing the random sequence;
- no hidden full-frame alpha/RGB cache;
- no overlap blending or seam feathering;
- no dropping or shortening events at tile boundaries;
- no changing the legacy effect to make the adapter easier;
- no weakening byte parity after confirmatory output is observed.

## Branches

- **Pass:** retain the experimental sparse-context adapter; integration still
  requires a separately frozen safe-Lab/effect/encode ordering contract.
- **RNG or layer parity failure:** repair exact event replay or close; do not
  alter the legacy layer.
- **Context grows as a dense raster:** close the design.
- **Performance ambiguity:** retain numerical evidence only; do not claim 100MP
  readiness.
- **Severe visual artifact:** reject the candidate/policy even if tiling parity
  passes.

## Verification, commits and claim ceiling

Contract, implementation and result propagation are separate commits and
pushes. Focused tests precede the full CPU suite. A formal real-raster smoke
runs only from the committed implementation.

A pass proves only exact sparse execution of the current heuristic procedural
dust/scratch composite. It does not establish realistic film defects, stock
authenticity, a complete tiled renderer or bounded end-to-end product memory.
