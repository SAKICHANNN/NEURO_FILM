# P161 — encoded render lifetime evidence

## Result

The exact interleaved order `baseline,candidate,candidate,baseline` passes all
frozen gates on the P160 three-source 24 MP workload.

- baseline median peak: `2,166,706,176 B`;
- candidate median peak: `1,886,382,080 B`;
- reduction: `280,324,096 B`;
- candidate/baseline RSS ratio: `0.870622`;
- candidate/baseline median wall ratio: `0.984366`.

This exceeds the frozen 200 MiB reduction gate and stays below the 0.92 RSS
and 1.05 wall-ratio ceilings.

## Exactness and lifecycle

All four runs reproduce the same ordered output hashes, recipe and normalized
report. All twelve source executions take identity fallback, and every worker,
staging path and detached worktree is cleaned.

The weak-reference regression separately proves each rendered image is alive
at encoder entry and collectible before the next source load. Phase sampling
agrees: source-load peaks fall from about `2.172 GB` to `1.881--1.883 GB`.

The frozen config, runner and ignored report SHA-256 identities are
`996f2261...dda3a`, `1febdbfe...1c2bd` and `3903253b...d1a3`.

## Verification boundary

Focused file tests pass `31 passed, 1 skipped`; audit tests pass `2 passed`.
A broader non-manifest repository attempt completed `1876 passed, 6 skipped`
and 36 failures in pre-existing main-project film-style fixtures, asset hashes
and absent generated outputs. None names `src.color_match` or P161; the broad
run is recorded as mixed rather than claimed green.

P161 changes no output, report, schema, ordering, guard, transaction, colour
or producer semantics. It proves only one local Windows/Python lifetime
improvement on the frozen SDR PNG batch.
