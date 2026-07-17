# U1.6F Grain Staged Global-Context Contract

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6F`

**Status:** frozen / ready

## Parent evidence and dependency audit

U1.6A provides exact halo-aware stitching. The current
`grain_residual_layer` is not coordinate-local: it generates one PCG64 normal
field, subtracts a sigma-1.2 Gaussian low pass, subtracts per-channel global
means, divides by one global standard deviation, applies a base-luminance
envelope, then subtracts per-channel residual means.

A bounded development probe establishes only feasibility inputs:

- row-chunked `Generator.normal` preserves the legacy sequence byte-for-byte;
- a float32 memmap has byte-identical NumPy mean/std reductions to the same
  ndarray on the probe;
- sigma 1.2 with truncate 3 uses direct radius 4 and never enters the
  downsample approximation.

These facts do not pass the node. They justify a preregistered staged pilot.

## Frozen implementation and structure

Add `src/filmfx/tiled_grain.py` rather than expanding the existing effect
adapter module. The experimental function must:

1. validate finite float32 base, strength, seed, colour mode, tile size,
   scratch directory and budget before writing;
2. create a private temporary directory under a caller-selected existing
   scratch root;
3. generate the unchanged PCG64 normal sequence in deterministic row chunks
   into a raw float32 memmap;
4. compute exact radius-4 high-pass tiles through U1.6A into a second memmap;
5. close/delete the raw memmap before later stages;
6. use the staged full-layout memmap for legacy-order means/std, then apply
   normalisation, luminance envelope and final channel mean in bounded chunks;
7. return the unchanged `FilmLayer` residual plus auditable execution/scratch
   metadata;
8. close handles and remove all scratch files on success and on injected
   failure.

This is bounded-RAM external staging, not a compact context. At peak it may use
two full-size float32 scratch fields plus input/output and tile working memory.

## Non-goals

- no new grain model, spectrum, density response or physical calibration;
- no change to legacy RNG, blur, normalisation, envelope or compositor;
- no direct coordinate jump for variable-consumption normal RNG;
- no persistent cache, project output, checkpoint or dataset artifact;
- no CLI/default/profile/recipe integration;
- no latency, SSD endurance, streaming, total-memory or 24MP/100MP product
  readiness claim.

## DoR

- U1.6A passes and U1.6E closes without code changes;
- branch is clean and synced at `d980756`;
- no active project process or user/Cursor dirty files exists;
- scratch is caller-owned local temporary space, not paid/cloud storage;
- rollback is one additive module/test commit.

## DoD and frozen gates

All gates are required:

1. at least four irregular shapes cover colour/monochrome, zero/active
   strengths, non-divisible row chunks, multiple seeds and image boundaries;
2. staged raw normal sequence is byte-identical to legacy generation;
3. high-pass staged output, both global means, scalar std and final residual are
   each byte-identical to the legacy layer for the frozen variants;
4. composited float and sRGB8 outputs are byte-identical to legacy;
5. metadata reports radius/halo 4, tile coverage, row chunk, raw/high-pass and
   peak scratch bytes, and proves no file remains after return;
6. repeated execution and metadata are byte-identical except private temporary
   path identity, which must not be exposed;
7. invalid inputs, nonexistent/non-directory scratch root, insufficient
   declared scratch budget and injected write/filter failure fail closed and
   leave no scratch files;
8. one committed real-raster active colour and B&W mechanics smoke passes
   numerical parity and visual inspection;
9. legacy effects, compositor, U1.6A-D and renderer tests pass;
10. complete CPU suite passes.

## Forbidden fallback

- no full ndarray noise allocation hidden behind the staged API;
- no different stat reduction, approximate std or changed random generator;
- no per-tile reseeding, hash noise or new grain appearance;
- no leaving scratch files for manual cleanup;
- no lowering byte parity after confirmatory output;
- no claiming that disk staging is bounded total memory or production-ready.

## Branches and claim ceiling

- **Pass:** retain the experimental staged legacy-grain adapter; production
  integration requires later resource policy, provenance and performance work.
- **RNG/reduction/filter parity failure:** close exact legacy staging; a new
  coordinate-hash grain would be a different separately judged effect.
- **Scratch cleanup/budget failure:** reject implementation even if pixels pass.
- **Visual failure:** no promotion of the corresponding grain policy.

A pass proves only exact legacy grain through bounded-RAM temporary disk
staging. It does not prove physically realistic grain or end-to-end product
scalability.
