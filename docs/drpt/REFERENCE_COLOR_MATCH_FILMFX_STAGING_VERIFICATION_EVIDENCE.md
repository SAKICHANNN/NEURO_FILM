# Reference Color Match FilmFX Staging Verification Evidence

Date: 2026-07-28

Status: **restart verification complete; final delivery remains closed**.

## Decision

P37 makes a P36 FilmFX staging run trustworthy after process restart. It does
not trust an in-memory result or file existence. The caller must retain the
exact P36 report SHA-256 and run ID; P37 rereads and validates the report,
then rehashes both sides of every recorded input/output lineage edge.

The result state is `verified-filmfx-staging` with claim ceiling
`verified-filmfx-staging-not-delivered`. It is read-only and cannot represent
delivery or application.

## Contract

`ExternalFilmFxStagingVerificationV1` binds:

- exact P36 report path, file SHA-256 and canonical run ID;
- exact P35 composition plan and P34 staging verification IDs;
- ordered source count and signed-int32 base/per-source seeds;
- every P33 input path and file SHA-256;
- every P36 output path, file SHA-256, format and bit depth;
- one canonical verification ID over the complete record.

The schema SHA-256 is
`4e6f883ea3243fc88fa121d065bc08ea08459f49e7625e99d44f1ce1ad1facfa`.

Reports are strict UTF-8 JSON and bounded to 16 MiB. Input paths, output paths
and the report path must be collision-free. All files must exist and match
their recorded hashes at verification time.

## Adversarial evidence

Eleven dedicated tests cover:

- complete report/input/output rebinding and strict schema/JSON roundtrip;
- report-byte mutation;
- wrong or malformed expected run ID;
- changed and missing P33 inputs;
- changed and missing P36 outputs;
- report relocation even when the caller supplies its new file hash;
- state, source order, dust seed, claim ceiling and verification-ID mutation.

## Verification

- dedicated P37: 11 passed;
- combined color-match and FilmFX: 447 passed;
- compileall and diff check: passed;
- complete CPU suite: 1287 passed, one skipped and the unchanged 36
  isolated-worktree failures caused by missing ignored outputs or historical
  Windows checkout asset hashes;
- no color-match or FilmFX test failed.

Implementation commit: `c1f4195`.

## Latest-main and producer propagation

- common base: `c03c321`;
- main snapshot: `a6895ca`;
- consumer implementation: `c1f4195`;
- consumer changed paths: 176;
- main changed paths: 113;
- exact changed-path overlap: zero;
- conflict-free merge tree:
  `9117db57e6562aa271b68445020614d9f8614532`;
- a fresh detached synthetic merge passed 113 P33-P37 and FilmFX tests and
  was removed.

Main's concurrent DoRF work was not modified. D-PCT remained at `c085bb1`
while its BMKL/ColorTransferLib research leaf was dirty; those files were
read-only and no producer schema, ABI, receipt or HDR rail changed.

## Remaining boundary

P37 closes restart integrity only. A final user-visible publish must be a
separate product decision and must not be opened using synthetic candidates.
The shared critical path still requires a fixed real producer invocation and
a candidate that passes A1/A4/A5 plus P27-P30. Physical halation remains
closed until resolved controls receive their own versioned binding.
