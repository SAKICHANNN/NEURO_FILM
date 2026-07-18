# U1.6G4C Chunk-Invariant Global-Resample Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4C`

**Status:** frozen / implementation ready

**Config SHA-256:** `157d586b9b1e32c57f791171bdca2f81d2ced0a56a402488cfb473ed6944e8a3`

## Purpose

G4B proves that `shape-stable-global-resample-v1` cannot be reconstructed from
arbitrary source-row batches byte-for-byte: its `np.tensordot` reduction order
depends on batch height. A development audit also finds that v1 can overrun the
last source cell on a `900x1600` real raster because a floating endpoint rounds
above the source extent.

G4C defines a new operator version. It does not repair, relabel or replace the
frozen G1-v1 evidence.

## Frozen operator

The version string is:

`shape-stable-global-resample-v2-explicit-f32`

The grid-selection, coarse Gaussian and original-coordinate bilinear
reconstruction geometry remain conceptually the same as G1. The area reducer
changes as follows:

1. each destination interval is computed in float64 from the global source and
   destination lengths;
2. the first boundary is exactly zero and the final boundary is exactly the
   source length; intermediate bounds are clamped to that closed extent;
3. overlap weights are normalized and cast to float32;
4. each destination cell starts from float32 zero;
5. source cells are visited in strictly increasing global index order;
6. every contribution uses explicit `np.multiply(..., dtype=np.float32)` and
   `np.add(..., out=...)` elementwise operations;
7. no `sum`, `dot`, `tensordot`, matrix multiply, BLAS or parallel reduction is
   allowed on the reduced axis;
8. width is reduced before height, matching the v1 separable direction;
9. full-array and bounded-row builders call the same reduction kernel and use
   the same global interval geometry;
10. the completed coarse grid is blurred once with the existing direct
    float32 Gaussian and made contiguous/read-only.

Batch/channel dimensions may be vectorized only through elementwise ufuncs;
they may not influence reduction order.

## API and structure

Extend `src/filmfx/global_resample.py` and its established package exports. Reuse
the existing plan/stage dataclasses with strict version-specific validation.
New public functions must have explicit `chunk_invariant` names; existing v1
public functions and bytes remain available and unchanged.

The bounded builder accepts a declared 2-D/HWC float32 shape, a v2 plan, a
coarse-row chunk smaller than coarse height, and `reader(y0, y1)`. Every call is
full width and must return exactly the requested finite float32 row window.
Metadata records ordered bounds, call count, maximum/total bytes, maximum row
span, logical source bytes and coarse bytes without retaining reader arrays.

Plans with no spatial downsample fail closed in the bounded builder; those are
finite-halo work, not global staging.

## Development choice

On the three G4B development cases, explicit float32 sequential reduction stays
within `5.59e-9` of the blurred v1 coarse fields. Float64 accumulation is not
consistently closer on RGB and would define a less faithful precision change.
Explicit float32 is therefore frozen before confirmatory runs.

On the fixed `1204x1600` real luma field, development maximum/mean reconstructed
drift versus v1 is `2.38e-7` / `8.65e-9`; three of 1,926,400 rounded uint8
values move by one code. The `900x1600` v1 comparison raises at its last cell;
that historical failure is recorded rather than repaired in place.

## DoR

- G1-v1 numerical result remains committed and reproducible;
- G4B is closed with repeat-identical failure report SHA-256 `e24e9908...`;
- G4A coordinate-exact gradients pass;
- clean pre-contract HEAD is `7789aee`;
- no effect, renderer, download or training process is active.

### Pre-run clerical correction

The first audit invocation stopped at the second real-file hash check before
writing a report. Its SHA had been transcribed incorrectly as `...c9cdcb4d...`.
The unchanged 2026-05-26 file and four pre-existing FilmStyleSafe inventories
all record `...c9cdcdbb...`. The corrected config hash above was frozen before
the confirmatory audit was rerun; no gate or algorithm changed.

## DoD and frozen gates

1. new-seed confirmatory scalar/RGB, irregular and anisotropic cases have
   byte-identical full-builder and row-builder coarse stages for every frozen
   chunk size;
2. full and tiled reconstruction are byte-identical across frozen tile sizes;
3. repeated stages, metadata and audit reports are byte-identical;
4. no reader call covers full source height; bounds/calls/bytes match exactly;
5. `1204x1600` real-luma v2-v1 drift is at most `5e-7` max and `5e-8` mean;
6. its rounded uint8 compatibility difference is at most `1e-5` of pixels and
   one code value; the frozen v1 failure on `900x1600` is preserved as a
   compatibility finding while v2 must complete;
7. fixed real-field v2 full/row/tiled bytes are identical and contact-sheet
   review finds no seam, band, block or other severe field artifact;
8. v1 regression hashes and focused tests remain unchanged;
9. invalid versions, shapes, channels, chunks, reader dtype/shape/finiteness,
   out-of-range geometry and injected reader failure fail closed;
10. focused tests and the complete CPU suite pass.

Runtime is reported but is not a promotion gate in this leaf. Total-memory and
100MP readiness remain open.

## Branches

- **Pass:** open a separate G3 capability binding plus the simplest staged
  density-family executor contract; do not integrate an effect in this leaf.
- **Cross-chunk byte failure:** close the candidate; never choose lucky chunks.
- **Compatibility drift failure:** close or redesign v2; do not loosen the
  frozen gate after viewing confirmatory results.
- **Real-field severe artifact:** reject v2 regardless of determinism.
- **Performance concern with numerical pass:** retain research evidence but
  keep integration closed and open an optimization leaf with identical bytes.

## Claim ceiling

A pass proves one chunk-invariant, bounded-reader numerical primitive plus a
limited v1 compatibility check. It does not prove physical halation, complete
effect parity, renderer integration, streaming decode, bounded total memory,
stock/calibrated response or 100MP readiness.
