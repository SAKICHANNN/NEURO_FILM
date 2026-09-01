# U7.12G Desktop Batch Output Format Contract

## Product question

The current native desktop exposes the existing PNG16, TIFF16 and JPEG8
encoders for one photo, but disables that same selector for two or more photos
and hard-codes the atomic batch to PNG16. U7.12G asks one product question:
can one explicit format choice apply to the whole atomic batch without adding
an encoder or weakening the existing transaction, recipe and replay rules?

This is a product feature over existing deterministic encoders, not a new film
model, format-support claim, runtime wrapper, recovery rewrite or research
rescue.

## Parent and compatibility boundary

- Source parent is commit
  `a6211b7934df9dcfd0b998b3666b0abdd82aba7e`.
- U7.11A remains the authoritative PNG16 batch contract.
- U7.12B remains the authoritative single-photo format contract.
- U7.12F remains the authoritative post-error selection reset.
- The default `export_batch(...)` call and a UI session that never changes the
  format must still emit the exact U7.11A PNG members, recipes and v1
  `batch.json` bytes.
- TIFF16 and JPEG8 use only the existing `scripts/render_film.py` command and
  strict recipe verifier. No codec quality, compression, colour, look, amount,
  seed, tiling or decoder option is added.

## Versioned batch semantics

Exactly one format applies to every child of a batch. PNG16 retains
`kmcfm.desktop-single-look-batch.v1` byte-for-byte. An explicit TIFF16 or JPEG8
selection uses `kmcfm.desktop-single-look-batch.v2`, records `output_format_id`,
recipe format and bit depth, and uses the canonical extension from the existing
`ProductOutputFormat` table. A mixed-format batch is forbidden.

The UI keeps PNG16 as the default, exposes all three keyboard-reachable choices
for both one-photo and multi-photo sessions, disables them during foreground or
batch work, and updates the export label truthfully. Changing the format does
not invalidate the hash-bound preview because it changes publication encoding
only.

U7.12D recovery remains explicitly PNG16-only. This leaf does not expose or
modify the withdrawn U7.12E recovery UI route.

## Safety and verification gates

1. Default PNG16 batch members, recipes, receipt and strict replay remain exact
   to U7.11A.
2. TIFF16 and JPEG8 batch children match direct CLI bytes at the same final
   paths and each strict recipe replays byte-exactly.
3. Every child recipe binds the selected format, bit depth, suffix, source,
   Look Approximation, amount, output hash and software commit.
4. The v2 aggregate receipt binds the selected format and exact child hashes.
5. Unknown format IDs reject before stage creation and before `command_runner`.
6. Cancellation, source/session/HEAD drift, existing destination and a
   late-created foreign destination preserve the existing atomic behavior.
7. Format controls are disabled while busy and remain usable only when the
   current preview authority is valid.
8. Focused tests plus U7.11A, U7.12B and U7.12F regressions pass in fresh
   processes; no owned scratch residue remains.

Any gate failure closes this exact composition without encoder tuning,
threshold changes, byte normalization, receipt weakening or adjacent recovery
and packaging work.

## Claim ceiling

A pass establishes only private Windows desktop batch publication mechanics
for the existing deterministic `film-inspired / Look Approximation` choices.
It does not establish calibrated stock response, physical-film reproduction,
stock distinguishability, arbitrary input or profile support, HDR/wide-gamut
publication, cross-platform GUI behavior, public release or user preference.

## Change and rollback

Freeze this contract/config before implementation. Reuse the existing output
format table and batch transaction; do not introduce a second encoder or
parallel batch module. Commit implementation/tests separately from formal
evidence. The change must remain a small revertible addition, preserve foreign
untracked `.codex/` and `tmp/`, and must not stage concurrent U4.2A artifacts.
