# Reference Color Match Local Delivery Verification Evidence

Date: 2026-07-28

Status: **restart verification complete; real candidate use remains closed**.

## Decision

P40 makes a P39 local export trustworthy after process restart. The caller
must hold the exact P39 report SHA-256 and delivery ID. P40 strictly rereads
that report, then rehashes both the P36 staging source and P39 delivered file
for every ordered source.

The state is `verified-local-delivery` with ceiling
`verified-local-files-reference-look`. Verification is read-only and does not
create an app-level applied state, stock identity or public-share claim.

## Contract

`ExternalLocalDeliveryVerificationV1` binds:

- exact P39 report path, file SHA-256 and canonical delivery ID;
- exact P38 authorization and P37 verification IDs;
- ordered source count;
- every staging path/hash and delivered path/hash;
- output format and bit depth;
- one canonical verification ID.

The schema SHA-256 is
`abe4270def5a0c60a1ff70674d188fbfa411871b116b59120eac2adbab23bf43`.

Reports are strict UTF-8 JSON and bounded to 16 MiB. Staging, delivered and
report paths must remain collision-free, and staging/delivered hashes must
remain byte-identical.

## Adversarial evidence

Eleven dedicated tests cover:

- full restart rebinding and strict schema/JSON roundtrip;
- report-byte mutation;
- wrong or malformed expected delivery ID;
- changed and missing staging sources;
- changed and missing delivered files;
- report relocation even with a new expected file hash;
- state, claim, source order, byte identity and verification-ID mutation.

## Verification

- dedicated P40: 11 passed;
- combined color-match and FilmFX: 479 passed;
- compileall and diff check: passed;
- complete CPU suite: 1319 passed, one skipped and the unchanged 36
  isolated-worktree failures caused by missing ignored outputs or historical
  Windows checkout asset hashes;
- no color-match or FilmFX test failed.

Implementation commit: `f9ec8c8`.

## Latest-main and producer propagation

- common base: `c03c321`;
- main snapshot: `9fea35b`;
- consumer implementation: `f9ec8c8`;
- consumer changed paths: 188;
- main changed paths: 128;
- exact changed-path overlap: zero;
- conflict-free merge tree:
  `8c6134f2cd99c39fff10aabaf9618fd409012031`;
- a fresh detached synthetic merge passed 149 P30-P40 and FilmFX tests and
  was removed.

Main's `.codex/tmp` and D-PCT's uncommitted Volga2K script were not modified.
No producer schema, ABI, receipt or HDR rail changed.

## Remaining boundary

P40 closes local delivery restart integrity. The consumer mechanics from
authorized staging through verified local export are complete. Real use still
requires a fixed producer invocation, a genuine source-bound candidate that
passes A1/A4/A5 plus P27-P30, reviewed merge into the main Neuro-Film branch,
and target-platform runtime evidence.
