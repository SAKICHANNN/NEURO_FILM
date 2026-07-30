# Reference Color Match Local Delivery Evidence

Date: 2026-07-28

Status: **atomic local transaction complete; real candidate use remains
closed**.

## Decision

P39 turns an exact P38 capability into a local file transaction. It rebuilds
the entire P38 authorization immediately before mutation, copies every P36
verified output byte-for-byte, and commits all N files plus one delivery
report atomically.

The state is `committed-local-delivery` and claim ceiling is
`local-files-delivered-reference-look`. This means only that local files were
committed. It is not an app-level `applied` state, a stock identity, a public
share or evidence that an algorithm is promoted.

## Contract

`ExternalLocalDeliveryV1` binds:

- exact P38 authorization and P37 verification IDs;
- ordered source count;
- every P36 staging path and exact file SHA-256;
- every delivered path and exact file SHA-256;
- file format and bit depth;
- canonical report path and delivery ID.

The schema SHA-256 is
`b8b41a3023e63f1c1c59d7962567fae881cde801c9bb69e2d8f9609944143f0c`.

Each delivered hash must equal its staging hash. Destination extensions must
match the staged file format, and the output set must obey the existing SDR
bit-depth extension policy. P33 and P36 staging files and reports are
protected from overwrite.

## Atomicity and adversarial evidence

Eleven dedicated tests cover:

- exact two-file plus report commit and strict schema/JSON roundtrip;
- byte equality between staged and delivered files;
- live output tamper before destination creation;
- valid foreign P38 authorization;
- staging-path overwrite attempt;
- wrong destination count;
- destination extension/format mismatch;
- injected report-commit failure restoring every old destination;
- applied-state, stock-claim, delivered-hash and canonical-ID mutation.

Temporary files are fsynced before commit. The shared batch commit primitive
backs up all existing targets, replaces them as one transaction and restores
them if any replacement fails. No staging debris remains.

## Verification

- dedicated P39: 11 passed;
- combined color-match and FilmFX: 468 passed;
- compileall and diff check: passed;
- complete CPU suite: 1308 passed, one skipped and the unchanged 36
  isolated-worktree failures caused by missing ignored outputs or historical
  Windows checkout asset hashes;
- no color-match or FilmFX test failed.

Implementation commit: `8d60fc8`.

## Latest-main and producer propagation

- common base: `c03c321`;
- main snapshot: `33493ac`;
- consumer implementation: `8d60fc8`;
- consumer changed paths: 184;
- main changed paths: 124;
- exact changed-path overlap: zero;
- conflict-free merge tree:
  `e23ecbfd7c78b330007955616de99a3b5d752233`;
- a fresh detached synthetic merge passed 138 P30-P39 and FilmFX tests and
  was removed.

Main's `.codex/tmp` and D-PCT's uncommitted Volga2K script were not modified.
No producer schema, ABI, receipt, algorithm or HDR rail changed.

## Remaining boundary

P39 completes local transaction mechanics, not real reference-match product
admission. The synthetic conformance chain exists only to prove behavior.
Real output remains blocked until a fixed producer invocation emits genuine
source-bound receipts and that candidate passes A1/A4/A5 plus P27-P30.
Reviewed integration into the main Neuro-Film branch and target-platform
runtime evidence also remain open.
