# P299 — HDR PNG create-only publication on canonical exFAT

## Problem and parent

The retained Rec.2020 SDR and Rec.2100 PQ streaming PNG rail stages sibling
bytes but publishes them with `os.replace`. Its higher-level ACES wrappers
check destination absence before construction, yet a destination created after
that check can still be overwritten. Direct users of the streaming writer can
overwrite an existing destination immediately. Both violate the documented
create-only contract.

P299 is a transaction-safety child of P91/P238 and reuses the already verified
U6.P8CS1 cross-filesystem primitive. It does not alter pixels, compression,
metadata, ACES transforms, defaults or scientific claims.

## Frozen implementation

- Replace only the final `os.replace(stage, destination)` in
  `src/preprocess/png_stream.py` with
  `src.film_physics.create_only_file.publish_create_only`.
- Preserve sibling same-volume staging, `fsync`, deterministic PNG bytes and
  abort cleanup.
- Do not change the existing early wrapper checks; the final no-replace
  operation remains authoritative against races.

## Frozen gates

1. A pre-existing destination is retained byte-exact and the stage is removed.
2. A destination injected after writer construction but before `finish()` is
   retained byte-exact and the stage is removed.
3. A successful real P-backed publication produces exact deterministic
   Rec.2100 PQ PNG bytes, cICP and RGB16 samples in forward/reverse row
   partitions.
4. No stage residue remains after success or failure.
5. Existing P91/P238/P252 output identities and adjacent tests remain exact.

Any output drift, overwrite, residue, non-exFAT-safe publication or replay
mismatch closes the leaf without a fallback copy, retry, destination deletion
or filesystem-specific bypass.

