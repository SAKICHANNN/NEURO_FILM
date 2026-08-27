# P263 R1EJ DNG ProfileGainTableMap callable consumer-intake contract

## Role

P263 is a private, mechanical consumer handoff audit. It does not reopen the
P257/P258 no-silent-drop guard family, alter the P98 DNG loader, or promote a
full DNG renderer. The producer implementation stays in the producer Git
repository. The consumer may read only the frozen Git objects listed below,
materialize them in an isolated temporary package, execute the canonical
fixture, and remove the temporary package after each run.

This contract and its matching configuration are committed before the
consumer deserializes the fixture or imports the callable.

## Exact producer lineage

- producer repository clean head:
  `5f96c1b99bd53d715643782423ec72dd66feec5e`;
- contract commit:
  `311e22e551ce12dc2db35c7d3f59cdf7c6004538`;
- callable/schema/fixture implementation commit:
  `1473f0caabf34f0c1c649517fdd49e907050624e`;
- audit commit:
  `d92659349f9eb1e369b7da6710e9e7ac4f30f4c4`;
- execution/evidence lock commit:
  `af839ca46fec368516559441966fb097124b9621`.

| Role | Path | Git blob | Bytes | SHA-256 |
|---|---|---|---:|---|
| contract | `docs/research/R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_V1_CONTRACT.json` | `571f3295db62f67befb3fcdf125dc6c63bf8ee11` | 3,757 | `17c2c8bf3c118f2f89aa05cc67c899fbb04a3b69a20bd9475f38387c60d39239` |
| payload schema | `schemas/zhuise_dng_profile_gain_table_callable_v1.schema.json` | `314b6708302e61f9df5789877c2e8485946429ef` | 1,644 | `435d04417de9af9dab218f23452c04f8696ccf2b6bf815bed0da54ad9e57142b` |
| arithmetic core | `src/zhuise/dng_profile_gain_table.py` | `7814dd301ebf08e992d4c82b06f9dcdf17ff387c` | 8,829 | `f608a5e4631e88d145d32379fa3b55480f8e38d7a13be9945ff564d146416b94` |
| callable | `src/zhuise/dng_profile_gain_table_callable.py` | `057f84c30478175cd7ee3bf55def978f68830733` | 6,291 | `3025af18cb8b6f9540815d0971ce3228ede45288148607d224487414be200439` |
| canonical fixture | `tests/fixtures/r1dy_dng_profile_gain_table_callable_v1_fixture.json` | `96962f3b492221bfdca505e5874fabf91c55fcc5` | 1,291 | `37bda94a036b84ae125c84c32f4e20b101b3bec468e6907ea13b2987bddee724` |
| execution lock | `docs/research/R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_EXECUTION_LOCK.json` | `4b14e04b9ca9e9b25af3bfc41cbc3039f41a0f54` | 2,217 | `8a150b4d9bb13241be44c8c64bc6400b75c2d5d8219b5b9f194053d815fdbfc6` |
| evidence | `docs/evidence/R1EJ_DNG_PROFILE_GAIN_TABLE_CALLABLE_HANDOFF_RESULT.json` | `d901c5d20ebbf126a3c4801c12a4a2b4e76eb91a` | 3,640 | `7a978197fc9617bb0e674e838bfd96ad12a425e5b93f82de189691fdcad246d1` |

The consumer must extract these exact objects with `git show`. Importing the
producer worktree or copying the callable into consumer `src/` is forbidden.

Additive prescore source-lock correction: the first committed contract bound
the wrapper but omitted its separate relative-import arithmetic core. Reading
the already-bound producer execution lock exposed that dependency before any
fixture deserialization or callable import. This amendment adds only the exact
core Git object; roles, fixture, expected output, gates, and claim ceiling are
unchanged.

## Frozen callable boundary

- callable ID: `zhuise.dng-profile-gain-table-callable.v1`;
- function:
  `zhuise.dng_profile_gain_table_callable.apply_dng_profile_gain_table_callable_v1`;
- input: finite `HxWx3` linear ProPhoto/ROMM RGB;
- output: owned, writable, C-contiguous float32 with unchanged shape;
- stage order: after ProfileHueSatMap and before exposure, ProfileLookTable,
  and ProfileToneCurve;
- payload fixes geometry, spacing, origin, five weights, row-major
  vertical/horizontal/intensity gains, exposure-weight gain, exact image area,
  and the boolean overrange policy;
- failures raise `ValueError` before returning output and perform no file or
  network I/O.

## Formal execution

Two fresh consumer processes execute forward and reverse control order. They
must independently verify every Git object, validate the payload with the
frozen Draft 2020-12 schema, import only from an isolated temporary package,
and reproduce the fixture's exact float32 output bytes. The audit also freezes
input/payload immutability, output ownership, wrong-shape/nonfinite/malformed
payload rejection, default-SDR direct-call parity, source-lock integrity, zero
network/new RAW/pixel/target reads, and zero temporary residue.

The producer reports exact fixture output f32le SHA-256
`9b74c55d...5b9b0`, formal report SHA-256
`6d3b381a08674eb3b78935f9d6c6f4ed6be0d6a902059f260c06bc3e3d9b0f3a`,
and stable identity `fa61d6dc...323be`. Full values must be read from the
source-locked execution object only after this contract is committed.

## Stop rule and claim ceiling

Any object, schema, domain, stage-order, output, ownership, mutation,
invalid-control, replay, or cleanup mismatch closes this exact handoff. No
implementation copy, worktree substitution, profile/table tuning, tolerance
relaxation, old-cohort rerun, or loader integration is allowed.

A pass proves only private consumer reproducibility of one exact R1EJ
ProfileGainTableMap v1 callable and canonical fixture. It does not establish
ProfileGainTableMap2, arbitrary cameras/profiles, a full DNG renderer,
real-pixel quality, default-loader support, public package/schema/capability,
film-stock evidence, automatic single-reference matching, or product
admission. Candidate 3 remains closed at `2/3`.
