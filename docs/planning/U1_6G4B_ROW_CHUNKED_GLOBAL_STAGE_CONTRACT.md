# U1.6G4B Row-Chunked Global Stage Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4B`

**Status:** frozen / implementation ready

## Purpose

U1.6G1 defines the correct original-image coarse grid but its builder accepts a
materialised full field. Halation global nodes are derived streams, so using
that API directly would hide a full-resolution source allocation. U1.6G4B
freezes a bounded row-reader constructor for the exact same G1 stage.

## Frozen algorithm

Given a valid G1 global-resample plan, declared 2-D or HWC float32 field shape,
a positive coarse-row chunk smaller than the coarse height, and a reader for
global source row intervals:

1. process coarse rows in deterministic increasing chunks;
2. map each coarse chunk to the minimum original row span covering its exact
   continuous area cells;
3. request only that full-width row span from the reader;
4. apply the same G1 horizontal area-overlap reduction to those rows;
5. for every global coarse row, apply the same float32 vertical overlap weights
   and `np.tensordot` order as the full builder;
6. apply the same direct coarse-grid Gaussian and return the same immutable
   `StagedGlobalField`;
7. report calls, maximum/total read bytes, maximum row span, coarse bytes and
   plan identity without retaining reader data.

This is exact staging for G1's versioned operator, not legacy Pillow parity.

## Bounded-reader rule

- `coarse_row_chunk < coarse_height` is mandatory, so the builder never asks
  for the entire source height in one call;
- each reader call is full width because horizontal area reduction needs all
  source columns, but its height is derived only from the current coarse chunk;
- the reader must return the exact requested finite float32 shape without cast;
- overlapping fractional boundary rows may be reread; metadata counts actual
  bytes read separately from logical source bytes;
- reader-internal allocation is outside the primitive and must obey its own
  future field-producer contract.

Plans whose coarse shape equals source shape are finite-halo cases and fail
closed rather than using this global-stage builder.

## Scope

Extend `src/filmfx/global_resample.py`, its established exports and focused
tests. Reuse the current area reducer and direct Gaussian; do not create a
parallel resampling module. No effect/G3 integration, renderer, profile, recipe
or CLI switch occurs in this implementation commit.

## DoR

- G1 global-grid pixels pass;
- G4A supplies coordinate-exact derived gradient windows;
- G3 identifies this as the sole remaining integration capability;
- clean pre-contract HEAD is `b649523`;
- no renderer, download or training process is active.

## DoD and gates

1. scalar and multichannel row-reader stages are byte-identical to full G1
   coarse stages for irregular shapes, anisotropic sigma and at least three
   coarse-row chunk sizes;
2. reconstructed full and tiled outputs from both stages are byte-identical;
3. reader calls are ordered, bounded, full-width and never full-height;
4. metadata exactly matches observed call count/read shapes/bytes;
5. repeated execution and metadata are identical;
6. finite-halo plans, invalid field shape/channels/chunk, reader shape/dtype/
   finiteness and reader failures fail closed;
7. injected mid-stage failure retains no reader window or partial public stage;
8. focused tests and complete CPU suite pass;
9. after evidence, G3 with both capabilities may become statically
   `integration_ready=true`, but no effect executor or product integration is
   implied.

## Branches

- **Pass:** remove the final G3 capability gap and open a separately frozen
  simplest staged density-family executor before colour-family expansion.
- **Area/coarse parity failure:** repair operation order; do not loosen byte
  parity or create a new operator version inside this leaf.
- **Unbounded reader request:** reject the implementation.
- **Reader failure:** fail without returning a partial stage.

## Claim ceiling

A pass proves exact construction of a G1 coarse field from bounded source-row
windows. It does not prove effect execution/parity, physical realism, renderer
integration, total-memory bounds, streaming decode, stock/calibrated response
or 100MP performance.
