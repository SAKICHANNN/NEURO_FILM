# Reference-match stored-recipe replay evidence

Date: 2026-07-27

Status: P16 complete for the current display-linear SDR file boundary.

## Closed gap

Before P16, a `ReferenceLookRecipe` could be replayed only through the
in-memory `WorkingImage` API. The file API and product CLI always required the
original reference image and fitted a new recipe. That did not prove a saved
recipe was a durable product artifact.

P16 adds:

- `load_reference_look_recipe_bound`: one bounded byte read supplies both the
  validated recipe and SHA-256 of the exact bytes that were parsed;
- `replay_reference_files`: one verified recipe is applied to an ordered,
  non-empty N-file batch;
- `FileReferenceReplayResult`: replay provenance without a fabricated current
  reference path;
- `reference_match_replay_report_v1.schema.json`: strict language-neutral
  provenance with `operation=recipe-replay`;
- CLI `--recipe-input` mode, mutually exclusive with `--reference`.

## Transaction and safety contract

1. Recipe, source and output path contracts are validated before decoding.
2. The recipe is size-bounded, read once, hashed, UTF-8 decoded and strictly
   parsed; canonical recipe-ID drift rejects before output staging.
3. The same recipe and guard policy are used for all N sources.
4. Every output is staged before any destination is replaced.
5. A late source/decode/render/encode failure removes every stage and commits
   no output.
6. Existing destinations are backed up and restored if batch commit fails.
7. The input recipe and source files are protected from output overwrite.
8. Default replay retains `algorithm-not-promoted` identity fallback; research
   rendering still requires explicit `--allow-research-baseline`.

The report binds loaded recipe bytes, embedded reference-pixel identity,
source hashes, output hashes, candidate diagnostics and each safety decision.
It intentionally does not claim the historical reference file is still
present.

## Evidence

- Removing the original reference after fitting does not affect recipe replay.
- Two replays of the same recipe/source produce byte-identical encoded output.
- A second invalid source after a valid first source leaves zero committed
  outputs and zero staging files.
- Recipe-ID tampering fails before creating an output.
- Output attempts targeting the recipe or source fail as protected-path
  violations.
- CLI recipe mode processes two ordered sources and produces a schema-valid
  replay report without the original reference.
- Dedicated P16 tests: 9 passed.
- P16 plus adjacent file/replay/report/schema tests: 40 passed.
- Focused colour-match/preprocess tests: 156 passed.
- Complete CPU collection: 1028 passed, one skipped and the same 36 known
  ignored-output/CRLF-hash failures.

## Remaining boundary

This closes durable recipe replay, not final algorithm promotion. The current
statistical algorithm still fails A1/A4/A5 and defaults to identity. RAW/HDR
ingress remains behind A3, and Android/iOS/macOS/Windows implementations must
still pass the P15 portable conformance bundle.
