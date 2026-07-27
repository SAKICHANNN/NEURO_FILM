# Reference Color Match Delivery Evidence

Date: 2026-07-27

Status: **module delivery ready; algorithm promotion remains correctly
closed**.

This distinction is intentional. The branch delivers the complete local
one-reference/N-source execution, replay, safety, provenance and film-effects
composition shell. It does not mislabel a rejected reference estimator as a
photographic champion.

## Delivered product contract

```text
fit:
  one display-linear SDR reference + N display-linear SDR sources
    -> immutable ReferenceLookRecipe
    -> N ordered outputs
    -> one transaction-bound provenance report

replay:
  stored verified ReferenceLookRecipe + N new sources
    -> N ordered byte-replayable outputs
    -> one transaction-bound replay report

composition:
  verified delivered-run report
    -> all applied: reference colour + optional film effects
    -> all fallback: identity, no film effects attributed to matcher
    -> mixed: reject one batch-level composition plan
```

The public Python surface is `src/color_match`; the local CLI is
`scripts/match_reference_color.py`. Strict JSON Schema and a
language-neutral canonical wire format are supplied for non-Python clients.

## Safety and claim state

- default delivery is identity because no algorithm passed A1/A4/A5;
- `--allow-research-baseline` is explicit and still subject to gamut and new
  boundary guards;
- candidate diagnostics and rejection reasons remain in the report even when
  identity is delivered;
- recipe/report/output writes are one rollback-safe transaction;
- stored replay does not require retaining the original reference;
- claim ceiling is `reference-look`;
- reference matching never identifies a stock or upgrades film authenticity;
- RAW, HDR/gain-map and video remain at the separately owned D-PCT ingress
  boundary until a trusted scene/display-to-MatchView bridge exists.

## End-to-end CLI evidence

The final smoke uses one real styled reference and two real neutral source
images, writes 16-bit PNG, then replays the stored recipe without the
reference.

Default fit:

- recipe ID:
  `12c084adb0916e0fcaadc0885a443a10070ac232d768f8b84073125b30e7cacd`;
- 0 applied / 2 identity fallback;
- fallback reasons include `algorithm-not-promoted` and the applicable
  photographic-tail guards.

Replay:

- same recipe ID;
- 0 applied / 2 identity fallback;
- output hashes exactly equal their fit counterparts:
  - source 02:
    `2710790453fb1d6ba0be4a4b32f5ca85bff39581a0b5c6c170aaadbb916b52b5`;
  - source 03:
    `f415c4c9ac76697badb5094d101962aacb8c0e111fc4b6b5a7111692577625dd`.

The stored recipe file is
`c88926c8...f7b9c`; fit and replay reports are independently bound at
`3f00908f...52d04` and `6481ad82...792a`.

The explicit research path is also executable: the same-content positive
control is accepted/applied after passing guards and produces
`23dc9347...c2d42`. This is execution evidence only, not algorithm promotion.

Generated smoke files remain ignored under
`outputs/reference_color_match_delivery_smoke`.

## Test evidence

### Branch collection

Final branch collection:

- 1081 passed;
- one skipped;
- 36 failed for the already classified worktree environment conditions;
- zero reference-colour-match test failures.

The 36 are not hidden:

1. historical tests bind ignored `outputs/` artifacts that exist in the main
   checkout but are intentionally not duplicated into this worktree;
2. frozen byte hashes for unrelated legacy configs assume LF Git blobs while
   this Windows checkout materializes CRLF.

The module-focused suites passed throughout every leaf. P23's adjacent suite
is 20/20.

### Latest-main synthetic merge

Read-only preflight target:
`350b68730da26002605f16db8db0cd24cae7527a`.

- merge base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`;
- this branch changes 95 paths;
- latest main changes 81 paths;
- exact path intersection: zero;
- `git merge-tree --write-tree` succeeds at synthetic tree
  `97263dc4db6be5e38748ce05981ad938d991f61e`;
- a temporary detached worktree merged both histories without conflicts;
- all 196 current reference-colour-match and adjacent ingress tests pass on
  that merged tree.

The temporary worktree was validated as the exact created path and removed
after the check. No main checkout file, branch or ignored output was changed.

## Concurrent-task boundary

- Main task `019f4b76-e70a-75c0-b7ea-b473ab38c200` is clean except its own
  `.codex/` and stable at `350b687`; its stock-first and film-algorithm work is
  not duplicated here.
- Standalone D-PCT task
  `019f9f3b-d0c2-7f21-b486-1dd902148739` is clean at `d2c7eff` and actively
  owns RAW/DNG/HDR/video/cross-platform media work.
- This branch reads only stable artifacts/commits, sends no coordinating
  mutation and writes only its separate worktree.

## Integration handoff

The branch is safe to review/merge as a self-contained module. Integration
does not make the statistical or neural research candidates default.

After a normal repository-owner review:

1. merge `codex/reference-color-match`;
2. rerun the 196 focused merged-tree tests;
3. run the existing full suite in the main checkout where ignored evidence
   artifacts are present;
4. keep the default identity certification until a future challenger passes
   every frozen promotion gate;
5. connect product UI only after choosing how to label
   `reference aesthetic approximation` versus paired `exact look copy`.

No push, merge into main, public release or third-party asset vendoring was
performed by this task.
