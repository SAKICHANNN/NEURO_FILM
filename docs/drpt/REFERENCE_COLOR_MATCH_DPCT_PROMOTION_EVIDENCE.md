# Exact D-PCT Invocation Promotion Evidence

Date: 2026-07-28
Node: P44A-D
Decision: **rejected**

## Result

The exact P43 `zhuise.dpct-chroma.cpu-reference.v1` wheel fails every
automated product evidence family on the frozen six-image known-look matrix.
No blind aesthetic review opens and no P30 authorization is possible.

This is a source-plus-reference fitted algorithm. Every row was independently
fit as declared by the producer; it is not represented as one reusable
reference-only operator. Known targets entered only after each candidate had
been rendered.

## Frozen A1/A4 matrix

Thirty cross-content rows use six neutral SDR sources and six corresponding
deterministic Velvia-look targets:

| Metric | Result | Frozen gate |
|---|---:|---:|
| improved rows | 0/30 | at least 75% |
| regressed rows | 30/30 | diagnostic |
| median improvement | -201.2941% | at least +10% |
| worst improvement | -569.3720% | at least -10% |
| maximum new-boundary fraction | 21.1987% | at most 5% |

The candidate therefore fails improvement rate, median, worst tail and new
boundary gates by large margins. This is stronger negative evidence than a
small underpowered difference.

## Frozen photographic tail

All six independently fitted photographic probes fail:

- worst neutral-axis chroma p95: `50.1215` against maximum `18`;
- worst new-boundary fraction: `28.6965%` against maximum `5%`;
- worst semantic hue-rotation p95: `89.0774` degrees against maximum `75`.

Tone reversal and plateau are not the only relevant failure modes; neutral,
boundary and semantic colour tails independently reject the candidate.

## Frozen A5 shared-colour context

All six references fail when the same exact chart pixels are embedded in
dark/cool and bright/warm surroundings and each source context is independently
fit to the same reference:

- worst median drift: `72.6391` Delta E76 against maximum `0.5`;
- worst p95 drift: `96.9062` against maximum `1.0`;
- worst maximum drift: `119.6310` against maximum `3.0`.

Deterministic execution and source ordering do not rescue this result. The
source-bound fit changes the same colours as unrelated surroundings change,
so the current capability cannot support strong album consistency.

## Reproducibility

Two full 48-invocation runs produce identical metrics, decisions, request IDs,
producer bundle IDs and consumer transform IDs. Their stable evidence ID is
exactly:

`90d0022c8c0d16070f38f96364e9f055ca86a3f5571a6eeafc780deea3f82d2a`.

Run IDs correctly differ because factual producer `timing_ms` changes
Diagnostics, ApplyResult, response and consumer receipt identities:

- run A: `1f3742c799e06d47ab656bae13f43238b9fb9bda62e7dd0cec2b9f43f38a088e`;
- run B: `41e437b87eae778b8726d262110f41b5df71490ee989433a3f8af4e533da7b0f`.

The evaluator preserves both identities. It never erases timing facts merely
to force byte-identical run reports. Stable evidence excludes only those
timing-dependent receipt identities and binds the contract, all metrics,
requests, bundles/transforms and decision.

Ignored report file hashes are `2c33dbc0...41250` and
`a80acc90...dee44`. Evaluator source SHA-256 is
`c66e3c902766ef8de665912ed500b0742a91dc4038d3c16420f2198e8a4c5c3e`.

## Verification and propagation

- 21 P43/P44/promotion/consistency tests pass.
- 441 combined color-match/FilmFX tests pass.
- Full suite: 1336 passed, one skipped and the unchanged 36 isolated-worktree
  output/asset failures; no P44/color-match/FilmFX failure.
- Latest main: `209da15b87f8ed52221923b35b366077f803dd54`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main paths: 202/147 with zero exact overlap.
- Merge tree: `31e9b9272e9e414feee77935a3a20109d4440d35`.
- Fresh detached merge passes 21 focused tests and was removed.

## Decision and next branch

The exact `dpct-chroma` capability remains callable research evidence but is
not product-promotable. Threshold relaxation, a research override or visual
review cannot rescue failures of this magnitude. P33-P40 correctly remain
unreachable for real use.

Future producer candidates may reuse P43 invocation plumbing only after a new
versioned capability/package lock. They must rerun P44 unchanged. Research
results such as BMKL/ROGR are not automatically substitutable because they do
not currently publish this invocation contract or pass these product gates.
