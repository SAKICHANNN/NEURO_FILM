# P309 — R1FS DNG ImageSequenceInfo no-copy intake contract

Status: `FROZEN_BEFORE_PRODUCER_OBJECT_IMPORT_OR_SOURCE_METADATA_READ`

## Question

Can this consumer independently verify and execute the exact source-locked R1FS
`zhuise.dng-image-sequence-info-group.v1` boundary from producer Git objects on
the exact three official Adobe DNG 1.7 files, without copying its implementation
into consumer source or decoding image samples?

## Frozen execution

- Verify the producer implementation, test and evidence by exact commit, Git
  blob, byte count and SHA-256.
- Materialize only the bound module in a fresh temporary package. Never import
  the producer worktree and never add a consumer sequence core or API.
- Verify all three source files by byte count and SHA-256 before metadata read.
- Load the same files in forward and reverse order and require the same immutable
  canonical sequence: one exact sequence ID/type, indices `1,2,3`, count `3`,
  final flags `false,false,true`, exact frame-info strings and payload hashes.
- Require input-file immutability, frozen-slot dataclasses, canonical ordering,
  zero image-sample decoder imports/calls and zero persistent temporary residue.
- Independently exercise empty, missing, duplicate, non-contiguous,
  cross-sequence, wrong-final and four malformed-payload controls. Every failure
  must be `ValueError` before a sequence is returned.
- Run two fresh processes with forward/reverse control order and require
  byte-exact reports and scientific identity.

## Stop rule and claim ceiling

Any producer object, source identity, field semantic, ordering, immutability,
failure, replay or cleanup mismatch closes P309. Do not copy or patch the
producer implementation, infer filenames, accept partial groups, decode pixels,
estimate focus, register/fuse frames, or rescue through a wrapper.

Passing proves only private no-copy consumer replay of one exact official
three-file metadata/group boundary. It is not arbitrary-DNG support, focus or
depth truth, registration, fusion, all-in-focus rendering, a public package,
schema/capability, automatic matching, stock evidence or product admission.
Candidate 3 remains closed at `2/3`.
