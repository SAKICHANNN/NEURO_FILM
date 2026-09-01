# U7.17A Desktop Bounded Finishing Effects Contract

## Purpose

`U7.17A` closes a real product-workflow gap in the private native desktop. The
existing product CLI and strict recipe already support deterministic grain,
simple halation, and dust, but the desktop can export only colour. Users must
leave the product workflow and manually reconstruct command-line arguments to
use the already-promoted finishing layers. This leaf exposes those existing
layers without changing their algorithms, the three colour Look
Approximations, or any calibrated-stock claim.

The three preview cards remain explicitly **colour-only**. Finishing effects
are resolution-dependent and therefore must not be simulated on the bounded
`300x260` cards. Their exact result is available through the existing
full-resolution `1:1` detail action and final single/batch export.

## Parent and ownership

- Frozen parent HEAD: `fe83854c9`.
- Parent product desktop SHA-256:
  `d9243200254947f9838da026d7e2edd9fa9415840ea92dd1ebae2a582ceb2aec`.
- Parent desktop UI SHA-256:
  `d19d308255cc0fdc1a42a40cc21ee00ecbc79ef8fd2a9c2e0cb1c13ed1de56d2`.
- Parent exact-detail helper SHA-256:
  `001d2b4a35f514e90b906eca54c010510b883dea10751a56ec51c6380e4a373b`.
- Existing product renderer SHA-256:
  `2013e7c2f95c2f6a0b84ccff1d93db9bec2b189aead84b1ca6b5ab477447f786`.
- U7.2R effect-preflight evidence SHA-256:
  `8ce65a299cc3bea82e751e4465fb734a713c7e486f83692d2f45a96cbf804670`.
- U7.16A detail-inspection evidence SHA-256:
  `3ac57e17191e18a704eb23c4fbbcd7356db20f2f2b46e160c040ac5d56481a68`.
- Root is the sole writer and integration owner. Consumer has explicitly
  released all declared U7.17A paths. Existing `.codex/` and `tmp/` are foreign.

## Frozen behavior

1. Add one immutable `DesktopFinishingEffects` value with exact strengths,
   fixed seed `7`, validation, and a canonical receipt representation.
2. The desktop caps are fixed before implementation at grain `0.05`, simple
   halation `0.15`, and dust `0.02`. Each native control displays an integer
   `0..100%` of its cap and maps exactly to `percent * cap / 100`. Values are
   never clamped, inferred, or silently repaired.
3. Defaults are exactly zero. At the all-zero default, export commands, output
   bytes, strict recipes, aggregate batch receipts, previews, and detail crops
   retain their parent identities and schemas.
4. For any nonzero effect, the desktop appends only existing CLI arguments:
   `--grain`, `--halation`, `--dust`, and `--seed 7`. It never selects physical
   or staged-density halation, writes layers/metrics, or changes colour.
5. The strict child recipe must bind the exact three strengths, seed `7`, dust
   seed `24`, `halation.model == "simple"`, and unchanged Look/amount/input/
   output/claim identities. Drift fails closed after preserving returned files.
6. Single-photo exports and U7.16A exact `1:1` temporary exports receive the
   same immutable effects value. The detail crop must match an independent
   crop of the corresponding final export exactly.
7. Batch children receive the same immutable effects value. Nonzero batches
   use aggregate receipt schema `kmcfm.desktop-single-look-batch.v3` and bind a
   top-level `finishing_effects` object. Zero-effect v1/v2 receipts remain exact.
8. The three bounded preview cards and input-basis preview do not rerender when
   effects change. The UI visibly states that cards are colour-only and exact
   effects appear only in `1:1` inspection/export. An effect change closes any
   old detail window but preserves the valid colour preview and Look selection.
9. Busy/close/cancel/error behavior, create-only publication, scratch ownership,
   recipe replay, format selection, representative binding, and batch ordering
   remain unchanged.

## Frozen formal roles and controls

The primary role remains the already-consumed U7.12C portrait
`a4593-kme_0276.jpeg` (4032x6048, SHA-256
`7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6`),
Ektar 100 Look Approximation, amount `0.65`, PNG16, tile size `256`, one worker.
The primary arm uses the maximum desktop bundle (`0.05`, `0.15`, `0.02`, seed
`7`). The identity control uses all-zero effects. Small deterministic fixtures
cover each effect independently, integer-percent mapping, invalid types/ranges,
all three formats, detail parity, single/batch command/recipe semantics,
v1/v2 receipt preservation, v3 receipt identity, cancellation, and foreign
destination preservation.

The formal photographic cohort uses existing product fixtures only. It checks
the maximum bundle across all three Looks and representative content roles for
finite output, range/boundary behavior, deterministic replay, exact recipe
semantics, no new gross clipping, and severe-artifact diagnostics already used
by the product line. It is product safety evidence, not film truth.

## Success gates

1. Parent desktop has no finishing-effect control while the product CLI and
   strict recipe demonstrably support the frozen bundle.
2. All-zero behavior is byte-identical to the parent for single export, detail,
   batch v1/v2 receipts, commands, recipes, and output media.
3. Percent-to-strength mapping is exact at 0, intermediate, and 100; bool,
   nonfinite, wrong-seed, negative, and over-cap values reject before rendering.
4. Nonzero single, detail, and batch commands contain the exact existing CLI
   arguments once; child recipes bind exact strength/model/seed semantics.
5. Exact detail output/crop equals the corresponding final effect export; temp
   output/recipe residue is zero on success and foreign/tampered members survive.
6. Nonzero batch v3 receipt and all children bind the same canonical effects;
   replay is exact and create-only/cancel/late-destination controls pass.
7. The UI is truthful and operable: cards say colour-only, three bounded controls
   disable while busy, changes close stale detail authority, and export/detail
   remain usable without rerendering colour previews.
8. The frozen maximum bundle passes finite, boundary, clipping, deterministic,
   replay, and severe-artifact gates on the formal existing product cohort.
9. U7.10A, U7.11A, U7.12A/B/C/G, U7.14A/B, U7.15A/B/C, and U7.16A behavioral
   suites pass. Historical evidence files remain immutable.
10. Forward/reverse committed-head reports are byte-identical; Ruff, format,
    compile, JSON, diff, source immutability, tracked-clean, and residue gates pass.

## Stop rule and claim ceiling

Any gate failure closes U7.17A. Do not rescue by changing caps, seed, effect
algorithms/order/model, preview resolution, cohort, output format, Look,
strength, thresholds, receipt semantics, or severe gates. Do not open physical
halation, learned grain, stock-specific effects, automatic effect routing,
effect presets, history/cache, or another adjacent wrapper.

A pass establishes only private deterministic desktop control of the existing
bounded finishing layers for `film-inspired / Look Approximation` exports. It
does not establish calibrated stock response, physical-film reproduction,
stock distinguishability, film-grain authenticity, arbitrary-media safety,
public release, population preference, or cross-platform GUI evidence. AO6
remains only a Velvia 50 display-proxy Look Approximation baseline.

## Execution

1. Commit this contract/config before implementation or formal rendering.
2. Commit the bounded core/UI/detail integration and focused tests separately.
3. Commit a formal controller, then run forward/reverse from committed HEAD.
4. Commit evidence/binding test and minimally propagate tracker, README if the
   actual user workflow changes, and AGENT_LOG. Do not push.
