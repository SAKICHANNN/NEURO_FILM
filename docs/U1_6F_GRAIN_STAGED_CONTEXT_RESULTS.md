# U1.6F Grain Staged Global-Context Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6F`

**Decision:** **numerical/resource-lifecycle pass** — retain the experimental
staged legacy-grain adapter; do not promote stress strength, physical realism,
renderer defaults or 100MP readiness.

## Delivered implementation

`src/filmfx/tiled_grain.py` adds a separate resource-owning path for the legacy
grain algorithm:

- row-chunked PCG64 normal generation into a private float32 raw memmap;
- U1.6A window planning with exact radius-4 Gaussian high-pass tiles;
- a second full-layout high-pass memmap for legacy-order channel mean, scalar
  standard deviation, luminance envelope and final channel mean;
- raw-file close/delete before global normalisation;
- caller-declared scratch budget and existing scratch-root validation;
- handle closure and temporary-directory removal on success and injected
  filter failure;
- path-independent metadata for topology, row chunk and scratch bytes.

The legacy `grain_residual_layer`, compositor, renderer, CLI, profiles and
recipes are unchanged.

## Exactness and scratch evidence

Four frozen variants cover colour and monochrome, zero/default/high/max
strength, multiple seeds, irregular shapes, non-divisible tiles and row chunks.
Every final residual, composited float image and sRGB8 image is byte-identical
to the unchanged legacy path.

Budget failure occurs before scratch creation. Injected filter failure removes
the private directory and both files. Repeated runs produce identical pixels
and path-independent metadata. The adapter reports peak scratch as exactly two
full-size float32 RGB fields; it does not count the caller's input, returned
full residual or transient tile/chunk allocations as disk scratch.

## Formal committed real-raster smoke

The smoke ran from implementation commit `cbb6c6c` on the same quarantined
mechanics raster used by earlier U1.6 leaves:

- source: `data/film_domain/velvia_50/fl_76ef574fbe793892.jpg`;
- source SHA-256: `11c546ede342a4e80cf75c71f7d9fcb31fa81950cd36627069076c747d6e5a29`;
- crop `(17,274,23,412)`, shape `257 x 389 x 3`;
- bases: heuristic safe-rich `velvia_50` colour and `hp5` neutral B&W;
- accepted smoke strength `0.018`, seed `7`, tile `64`, row chunk `37`;
- topology: 35 tiles, halo 4, maximum expanded tile `72 x 72 x 3`;
- raw/high-pass scratch: `1,199,676` bytes each;
- peak scratch: `2,399,352` bytes;
- remaining scratch files: zero;
- colour residual/float/sRGB8: byte-identical, sRGB8 SHA-256
  `726c7f3adf86bcbd652313b0e73f52e4d57e227700f427860547e8e1c6429c40`;
- B&W residual/float/sRGB8: byte-identical, sRGB8 SHA-256
  `dadd6ecf03c073be3f643f93d915d09cf6b8d53c91fe7422c18b3a33a6292aae`;
- active-effect changes: 270,301 colour and 269,610 B&W sRGB8 channel
  values versus their bases.

The raster supplies no stock, grain, rights, preference or calibration truth.

## Visual adjudication and severe veto

At `0.018`, autonomous visual inspection finds fine visible grain without a
new seam, block, geometry or colour-corruption failure on the mechanics crop.
This is not population preference or physical-grain validation.

The same committed adapter was stress-probed at strength `0.35`. Both colour
and B&W outputs become dominated by severe high-frequency noise, with maximum
base changes near `0.96` and `0.89`. That policy is rejected under the severe
artifact veto even though staged and legacy bytes match. Numerical parity never
overrides visual safety.

## Verification and claim boundary

- dedicated staged-grain tests: **21 passed**;
- adjacent tiler/effect/renderer set: **127 passed**;
- complete CPU suite: **349 passed**;
- compile and diff checks pass.

U1.6F closes as an exact legacy-execution and cleanup pass. It proves bounded
RAM for grain working state only by trading for O(image) temporary disk and
still returning a full residual. It does not prove total memory, streaming,
latency, SSD suitability, 24MP/100MP readiness, physical grain, stock
authenticity or production integration. U1.6 and Ultimate remain active.
