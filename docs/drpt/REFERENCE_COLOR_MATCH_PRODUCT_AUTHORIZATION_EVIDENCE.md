# Reference Color Match Product Staging Authorization Evidence

Date: 2026-07-28
Node: P30A-D
Status: stable staging authorization contract; no commit or delivery

## Purpose

P29 establishes numeric eligibility, not product authority. The upstream
CoreAcceptance contract intentionally allows a research baseline to enter
guards when `research_baseline_override=true`. P30 closes that boundary
without duplicating the promotion evaluator.

## Contract

`CoreProductStagingAuthorizationV1` binds:

- the exact P28 ordered batch ID and source rows;
- the exact P29 numeric guard batch and per-source decision IDs;
- each original CoreAcceptance decision ID and transform;
- each source, admission and receipt lineage already fixed upstream.

Every evaluated row must have:

- `accepted_for_product_guard=true`;
- `core_status=ok`;
- `promotion_status=promoted`;
- `research_baseline_override=false`;
- a passing P29 numeric decision.

Every row passing yields only `authorized-for-staging`. One failure makes the
whole authorization `identity-fallback`. An upstream numeric fallback is
not evaluated and cannot accept a supplied acceptance list.

The claim ceiling is `staging-only-not-committed`; there is no pixel payload,
file path, commit, delivery, partial-output or applied state.

Schema SHA-256:
`804477ba9fda53885c90a4a158e60ca2f1cc50ec2019cd21a050b1863b69af1e`.

## Discriminating evidence

The key negative control deliberately builds an upstream batch where one
candidate is accepted through `research_baseline_override=true`. It passes
P28 admission and a relaxed P29 numeric policy, then P30 rejects the row with:

- `algorithm-not-promoted`;
- `research-baseline-override`.

The complete batch becomes identity fallback. A matched batch whose two
acceptances are genuinely promoted and non-research receives staging
authorization.

Other tests cover acceptance reordering, count mismatch, cross-batch binding,
numeric fallback short-circuit, source index mutation, research flag mutation,
claim/state/canonical-ID mutation, strict JSON and JSON Schema roundtrip.

## Verification

- 11 dedicated P30 tests pass.
- 76 combined P27-P30 tests pass.
- Full suite: 1229 passed, one skipped, 36 unchanged environment failures.
- No colour-match or P30 test fails.
- Latest main remains `a33526e`; common base is `c03c321`.
- Consumer/main changed-path overlap is zero.
- Clean merge tree: `5fbb6ca217884820dc23c896cca0af918f6ee0a6`.
- Detached synthetic merge `b06748b5...` passes all 76 selected tests.
- The temporary merge worktree was removed.

Main's concurrent untracked Z1 result files were observed read-only and were
not included, edited, moved or deleted.

## Propagation

P30 changes no D-PCT algorithm, producer schema, diagnostic definition,
threshold, media/RAW/HDR path, transaction writer, FilmFX path or main
worktree. The separate absolute BT.2020 HDR rail remains explicitly unmapped.
