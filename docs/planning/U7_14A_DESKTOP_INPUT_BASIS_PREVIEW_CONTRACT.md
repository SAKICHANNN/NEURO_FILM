# U7.14A Desktop Input-Basis Preview Contract

## Role

`U7.14A` fixes a concrete native-desktop comparison defect. The current
workflow shows three processed Look Approximation cards but no view of the
pixels that entered the shared look renderer. A user therefore cannot compare
the selected strength against its own input basis inside the product window.

The repair adds one explicitly labelled `Input basis` thumbnail. It is encoded
from the already decoded, resized display-sRGB array used by all three looks.
It is not a second decode, an embedded-camera JPEG shortcut, a new tone map, a
camera-accurate RAW rendering, a neutral target, or a fourth selectable look.
The three existing look previews, final exports, recipes, batch contracts and
default direct-render API remain exact.

## Parent state and ownership

- Frozen parent HEAD: `059b5be4e5fdd5ba88d372d298bacef1d7a2336e`.
- U7.12G is exact at evidence SHA-256
  `59e62b1ce628f4a8999ebddb387d6a73c6bb9cf984cb33dc6a5f9898aed9aa38`.
- U7.12C remains the display-native geometry and fidelity authority at evidence
  SHA-256 `a68f6d0ebba6d74341798a8828fcc10da9c9ed164078ae6f63bf92ec6161c2ef`.
- Root is the sole writer and integration owner for the declared U7.14A paths.
  Producer and consumer tasks are read-only reviewers and reported no overlap.
- Existing `.codex/` and `tmp/` trees are foreign/untracked. Formal scratch may
  use only one owned repo-relative `tmp/u7_14a_*` subtree and must remove it.

## Trigger and invariant

The trigger is any successful desktop preview session. The UI currently gives
the user three processed choices without displaying the common source basis.
The failed invariant is that an explicit strength/look decision should be made
with a visible, same-session input reference.

The source preview belongs in `three_stock_preview.py`, immediately after the
existing one-decode, geometry-bounded `working_image_to_srgb_float` adapter and
before shared look execution. The direct renderer receives a new opt-in
`include_input_preview=False`; omission and explicit false must preserve the
historical manifest and member tree exactly. The product workflow alone passes
true, binds the extra file into the same immutable session, and exposes its
bytes through a separate method rather than adding a fake style ID.

The native UI displays the file in a bounded `160x120` label above the look
cards with the truthful copy:

> Input basis · generic display adapter, not a calibrated camera rendering

The input basis never receives a radio button, never changes selection, and
never becomes an export/recipe/catalog row. Input, amount, operation error and
workspace close clear the thumbnail with the existing preview authority.

## Frozen fixtures and comparisons

The exact U7.12C JPEG portrait and Blackmagic DNG are reused only as existing
product/runtime fixtures:

1. JPEG `a4593-kme_0276.jpeg`, 17,798,433 bytes, SHA-256
   `7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6`.
2. DNG `blackmagic_pocket_cinema_camera_4k.dng`, 4,497,152 bytes, SHA-256
   `1491a5e5d580a9be151747bddce89faca4d0b9d352503270f516bbf8e8cdf0d9`.

Both use the unchanged U7.12C 300x260 request, amount `1.0`, seed `7`, tile
size `256`, one worker and PNG compression `6`. JPEG keeps scaled libjpeg
decode; DNG keeps LibRaw half-size decode. The independent oracle applies the
same selected decoder, same linear-light INTER_AREA resize, the existing
`working_image_to_srgb_float`, and the existing deterministic `save_srgb8`.
No target, external network or new media is used.

## Frozen success gates

1. Opt-in false and omission reproduce the exact historical manifest object,
   member names and three look PNG bytes.
2. Opt-in true performs exactly one selected source decode and its three look
   PNG bytes equal the historical call byte-for-byte.
3. The input-basis PNG decoded RGB and encoded bytes equal the independent
   same-decoder/same-resize/same-adapter oracle for both fixtures.
4. Input-basis geometry equals the three look geometry, stays within 300x260,
   contains finite RGB8 pixels, and is deterministic across two fresh processes
   and forward/reverse source order.
5. The candidate manifest records a separate strict `input_preview` object,
   exact path/hash, source kind, generic adapter statement and the non-calibrated
   claim. It does not alter `rows`, look IDs or the renderer claim ceiling.
6. Product workflow session binding rejects missing, replaced, tampered or
   extra input-preview members; source/session drift still rejects. Cleanup
   removes only still-owned members and leaves zero owned residue.
7. The UI shows the labelled thumbnail only after a successful preview, never
   gives it a selectable look ID, and clears it on input/amount/error/close.
   Existing explicit-selection and export enablement rules remain unchanged.
8. U7.12C, U7.12F, U7.12G, U7.11A and U7.10A behavioral suites pass in fresh
   processes. Historical evidence files remain immutable.
9. Source identities are unchanged; formal forward/reverse scientific payloads
   are exact; tracked diff is clean at formal execution; owned residue is zero.

## Stop rule and claim ceiling

Any gate failure closes this exact input-basis preview without changing source,
decoder, resize, transfer adapter, label, dimensions, format, look, amount,
seed, worker, tile, compression, thresholds or UI role. Do not rescue by using
an embedded preview, inventing a RAW tone map, calling the input camera-accurate,
adding a fourth look, or opening adjacent preview widgets.

A pass establishes only a private Windows/Python desktop comparison aid on two
exact existing fixtures. The displayed source is the renderer's generic input
basis. It is not a calibrated camera rendering, RAW development claim, neutral
reference, calibrated stock response, physical-film reproduction, stock
distinguishability, population preference, HDR/wide-gamut support, installer,
public release or cross-platform GUI evidence. Every named output remains
`film-inspired / Look Approximation`; AO6 remains only the Velvia 50
display-proxy Look Approximation baseline.

## Commits and verification

1. Commit this contract/config before implementation or candidate pixel reads.
2. Commit the additive renderer/core/UI/tests as one reversible product leaf.
3. Commit the formal controller before clean committed-head execution.
4. Run forward and reverse formal processes, targeted/fresh-process parent
   regressions, Ruff, format, compile, JSON, diff and owned-residue checks.
5. Commit evidence/binding test, then minimally propagate tracker and agent log.
6. Do not push.
