# Reference Color Match External FilmFX Execution Evidence

Date: 2026-07-28

Status: **procedural FilmFX staging complete; final delivery remains closed**.

## Decision

P36 executes only the procedural FilmFX branch of an exact P35 external
reference composition. Before rendering, it reruns P34 against the
caller-held P33 report hash and run ID. The verified external reference look
continues to own colour; P36 adds no film colour transform or stock identity.

The output state is `filmfx-rendered-to-staging` and the claim ceiling is
`filmfx-staging-not-delivered`. These are new staging artifacts, not final
product delivery and not an `applied` claim.

## Contract

`ExternalFilmFxRunV1` binds:

- exact P35 composition plan ID;
- exact P34 staging verification ID;
- ordered source count and source indices;
- absolute input/output paths and exact file SHA-256 identities;
- explicit signed-int32 base seed and per-source grain/dust seeds;
- one uniform SDR output bit depth;
- atomic report path and canonical run ID.

The schema SHA-256 is
`5d6d27b6c993e37e8e285fd4113b3a1433385e999f6fc72bdab8ff8bd2b0f96e`.

## Execution semantics

P36 reuses the product's existing primitives:

- `grain_residual_layer`;
- simple `halation_layer`;
- `dust_scratch_layer`;
- `composite_layers`;
- `WorkingImage` SDR decode;
- the existing sRGB encoder and rollback-safe batch commit.

For ordered source index `i`, grain seed is `base_seed + 1009*i` and dust
seed is `grain_seed + 17`. The entire range is checked as signed int32.
Repeated execution with identical pixels, plan and seed produces identical
output file hashes.

Physical halation fails closed. P35 binds the model name and strengths but
does not bind the resolved physical control set required for faithful replay.
P36 will not substitute hidden CLI defaults.

## Adversarial evidence

Eleven dedicated tests cover:

- atomic two-source rendering and strict schema/JSON roundtrip;
- exact repeatability for equal seed and inputs;
- empty-effect rejection;
- physical-halation rejection;
- live P34 re-verification after input tamper;
- P33 source/report overwrite protection;
- injected final report commit failure with complete rollback;
- state, source order, dust seed, claim ceiling and run-ID mutation rejection.

## Verification

- dedicated P36: 11 passed;
- adjacent transaction/composition/FilmFX: 115 passed;
- combined color-match and FilmFX: 436 passed;
- compileall and diff check: passed;
- complete CPU suite: 1276 passed, one skipped and the unchanged 36
  isolated-worktree failures caused by missing ignored outputs or historical
  Windows checkout asset hashes;
- no color-match or FilmFX test failed.

Implementation commit: `aafa097`.

## Latest-main and producer propagation

- common base: `c03c321`;
- main snapshot: `5f99ae9`;
- consumer implementation: `aafa097`;
- consumer changed paths: 172;
- main changed paths: 102;
- exact changed-path overlap: zero;
- conflict-free merge tree:
  `5733ee0ba9de039cc95d3bde721b364722645a0f`;
- a fresh detached synthetic merge passed 102 P33-P36 and FilmFX tests and
  was removed.

Main's concurrent dirty DORF research files were not modified. D-PCT snapshot
`c085bb1` is clean; its AceTone negative resource smoke changes no producer
schema, ABI, receipt or HDR rail and therefore requires no P36 consumer
change.

## Remaining boundary

P36 closes procedural FilmFX staging mechanics only. Real product use still
requires a fixed producer invocation artifact and a genuine candidate that
passes A1/A4/A5 and the existing P27-P30 guards. Final delivery must be a
separate consumer decision over verified P36 artifacts. Physical halation
requires a separately versioned resolved-control binding.
