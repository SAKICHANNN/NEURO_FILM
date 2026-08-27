# P300 — canonical-partition PQ PNG create-only publication

## Problem and lineage

P299 repaired the base streaming PNG writer, but the live P91/P238/P264
canonical-partition writer overrides `finish()` and still publishes its sibling
stage with `os.replace`. Its wrappers reject a destination that exists before
writer construction, yet a destination created after that check can still be
overwritten. Direct writer users can overwrite an existing path immediately.

P300 is a prospective sibling transaction-safety leaf. It does not rewrite
P299 evidence and does not change canonical compression, pixels, metadata,
ACES transforms, defaults, or scientific claims.

## Frozen implementation

- Replace only the final `os.replace(stage, destination)` in
  `src/preprocess/canonical_pq_png.py` with the already verified
  `src.film_physics.create_only_file.publish_create_only` primitive.
- Preserve sibling same-volume staging, `fsync`, canonical feed partitioning,
  deterministic PNG bytes, and abort cleanup.
- Preserve the existing early wrapper checks; final no-replace publication is
  authoritative against a late destination race.

## Frozen gates

1. A pre-existing destination is retained byte-exact and the stage is removed.
2. A destination injected after writer construction but before `finish()` is
   retained byte-exact and the stage is removed.
3. A successful real P-backed publication produces the exact frozen P238
   canonical P3-D65/PQ PNG bytes, cICP, and RGB16 samples in forward/reverse
   caller partitions.
4. The existing P91 canonical partition-invariance test remains exact.
5. No stage or media residue remains after success or failure.

Any overwrite, byte drift, residue, non-P-backed formal output, or replay
mismatch closes the leaf without fallback copy, retry, destination deletion,
or filesystem-specific bypass.

## Claim ceiling

Private Windows canonical-P/exFAT create-only transaction safety for the
existing canonical-partition Rec.2100-PQ PNG writer only. This is not a new
colour transform, HDR/display-quality result, arbitrary-media result,
package/schema/capability admission, stock evidence, or product-default change.
