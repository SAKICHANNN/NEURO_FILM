# U7.12C Desktop Display-Native Preview Contract

## Role

`U7.12C` removes a concrete product latency defect from the private native
Look Approximation desktop. The current workflow renders three previews at up
to 1,000,000 pixels each and the UI then resamples every result to the only
visible 300x260 card. This leaf makes the desktop request the final
aspect-preserving card geometry from the existing deterministic preview
renderer. It does not change any look, strength, decoder, export, recipe,
batch, cache, colour, or claim semantics.

## Parent state and ownership

- Frozen parent HEAD: `4f292c1e7b20731dcb38c6be7c2a31dfda54db7b`.
- U7.12B is formally complete at report SHA-256
  `281ff125c9f019f83753c0a98ab2d7047232c0d61c7a656b8e1064b704d81488`.
- Root is the sole writer and integration owner. Producer and consumer tasks
  released the exact U7.12C files and remain read-only reviewers.
- Existing `.codex/` and `tmp/` trees are foreign/untracked and remain
  untouched. Formal scratch uses only an owned repo-relative `tmp/` subtree
  and must remove it.

## Trigger, invariant, and collateral scope

The exact trigger is any source whose current 1MP preview exceeds the UI card:
the renderer performs colour transforms and PNG publication on pixels that the
UI immediately discards. The broken invariant is that product preview work is
not bounded by the actual visible display request.

The producer-side repair belongs in `three_stock_preview.py`, not in a UI-only
shortcut. The preview geometry function and renderer gain optional positive
width/height bounds; omitting both preserves the historical direct-render API
and exact results. The product core owns one shared `300x260` display box and
passes both bounds. The UI imports that same constant and retains its defensive
thumbnail operation, which must be a no-op for candidate output.

Equivalent triggers include portrait, landscape, square, panoramic, JPEG,
RAW, and ordinary raster sources. Geometry unit tests cover representative
aspect ratios and no-upsample behavior. Existing U7.3G/U4.5B/U4.5F direct
renderer tests prove unbounded callers retain historical behavior; U7.10A
through U7.12B prove desktop input, batch, close-safety, format, recipe, and
publication behavior remains available.

## Frozen formal roles

Two previously consumed product fixtures are locked before candidate pixels:

1. Exact JPEG portrait `a4593-kme_0276.jpeg`, 4032x6048, SHA-256
   `7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6`.
2. Exact Blackmagic Pocket Cinema Camera 4K DNG, 4128x2176, SHA-256
   `1491a5e5d580a9be151747bddce89faca4d0b9d352503270f516bbf8e8cdf0d9`.

Both are pre-existing internal product/runtime fixtures, not film truth. The
baseline is the unchanged 1MP preview renderer followed by Pillow LANCZOS into
the 300x260 visible card. The candidate is the same renderer, profile, three
looks, amount `1.0`, seed `7`, tile size `256`, one worker and PNG compression
`6`, but it receives the 300x260 geometry before decode/downsample and look
execution. Expected candidate dimensions are exactly 173x260 for the JPEG and
300x158 for the DNG. JPEG keeps the existing scaled decoder and RAW keeps the
existing half-size decoder.

The formal controller executes both sources in forward and reverse order from
committed clean HEAD. Each source/variant receives two timed fresh worker
processes. Raw timings and peak process-tree RSS remain evidence but are
excluded from the stable scientific identity; source, geometry, pixels,
metrics, output hashes, order-independent decisions, code/config identities,
cleanup, and gate outcomes must be exact.

## Frozen success gates

1. Candidate decoded/display RGB versus the baseline visible RGB passes, per
   source and per look: RMSE <= `0.03`, p95 absolute error <= `0.08`, and new
   exact-boundary fraction <= `0.001`.
2. JPEG candidate median wall time <= `1.0` second and median candidate/baseline
   wall ratio <= `0.25`.
3. DNG candidate median wall time <= `8.0` seconds and median
   candidate/baseline wall ratio <= `0.75`.
4. Candidate peak process-tree RSS is <= the corresponding baseline maximum
   for both sources.
5. Both candidate repeats produce exact decoded RGB hashes and exact three PNG
   hashes; the three look outputs remain distinct.
6. Candidate dimensions, manifest bounds, source hashes, source immutability,
   processing order identity, finite pixels, display-box containment, and zero
   owned residue are exact.
7. Omitting display bounds preserves historical preview geometry and manifest
   behavior. Supplying one bound without the other, booleans, zero, negative,
   non-integers, or contradictory manifest dimensions rejects deterministically.
8. U7.3G, U4.5B, U4.5F, U7.10A, U7.11A, U7.12A and U7.12B behavioral parent
   suites pass after implementation. Historical evidence hashes remain
   immutable and are not rewritten to bind current shared files.

## Stop rule and claim ceiling

Any formal gate failure closes this exact 300x260 display-native candidate.
Do not rescue it by changing the display box, source, threshold, timing
aggregation, decoder mode, interpolation, cache, look, amount, seed, worker,
tile, compression, output format, test role, or baseline. A failure does not
reopen an adjacent resize/cache/model experiment.

A pass establishes only private Windows/Python desktop preview latency and
display-fidelity mechanics for two exact inputs and the three existing
deterministic `film-inspired / Look Approximation` choices. It is not broad
device performance, arbitrary JPEG/RAW quality, calibrated stock response,
physical-film reproduction, stock distinguishability, population preference,
HDR/wide-gamut publication, public release, installer, or cross-platform GUI
evidence. AO6 remains only the Velvia 50 display-proxy Look Approximation
baseline.

## Commits, verification, and rollback

1. Commit this contract/config before implementation or candidate pixels.
2. Commit additive geometry/core/UI/tests separately. Each omitted-bound direct
   API remains backward compatible and independently revertible.
3. Commit the formal controller before running from clean committed HEAD.
4. Preserve raw forward/reverse reports outside Git only until the exact
   evidence record and binding test are committed.
5. Run targeted and adjacent behavioral regressions, Ruff, compile, JSON, diff,
   source immutability, and owned-residue checks.
6. Propagate tracker and agent log only after committed evidence; no push.
