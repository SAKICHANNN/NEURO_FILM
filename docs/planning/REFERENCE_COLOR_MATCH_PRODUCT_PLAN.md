# Neuro-Film Reference Color Match Product Plan

## Document role

This is the executable product implementation plan for neuro-film's uploaded
reference-photo matching capability. It is not a research result, a stock
calibration plan, or a replacement for the standalone D-PCT project.

## Goal

Provide an image-first product module with this contract:

```text
one uploaded reference WorkingImage + N source WorkingImages
    -> one immutable LookRecipe
    -> N deterministic, batch-consistent rendered WorkingImages
```

The module is a peer of user-selected film simulation. It can later be composed
with a selected stock, but reference matching alone is labeled
`reference-look`, never as recovered stock truth.

## Current state

- Branch/worktree: `codex/reference-color-match` /
  `C:\Users\hhvrf\Documents\neuro_film_color_match`
- Base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`
- Reusable product ingress: `src/preprocess/types.py::WorkingImage`
- Reusable deterministic math: `src/color_engine/lab.py`,
  `src/color_engine/safe_lab.py`, `src/color_engine/gamut.py`
- Main-chat W1/W2 research and standalone D-PCT media work are concurrent and
  explicitly out of this branch's write scope.

## Non-goals for the first product slice

- no RAW/HDR/gain-map/video codec implementation;
- no neural model, training, external weights or GPU job;
- no direct RGB generation;
- no semantic/local masks;
- no stock inference, authenticity or calibration claim;
- no modification of the existing renderer default or CLI;
- no integration of uncommitted W1/W2 research code.

## Product invariants

1. `WorkingImage` is the only image ingress.
2. The first slice accepts `display_linear` SDR in `linear_srgb` or
   `linear_rec2020`; scene-linear RAW and unsupported dynamic range fail
   closed until an explicit render bridge exists.
3. Fitting consumes only the reference image and policy configuration.
4. One recipe is reused for all N sources.
5. Every input and output is finite float32 HxWx3; inputs are not mutated.
6. Reference target statistics, policy, schema version and deterministic
   fingerprint are replayable.
7. Output gamut handling is explicit. No hidden clipping is allowed.
8. The claim ceiling is `reference-look`.

## Implementation tree

| Step | Status | Scope | Required verification | Commit point | Rollback |
|---|---|---|---|---|---|
| P0 | DONE | Coordination, contract, plan | Git/other-chat snapshot and diff check | `ec001b2`, `acb0c82` | revert commits |
| P1 | DONE | `LookRecipe` types, fitting and validation | 30 focused/existing colour tests pass | `1820af1` | revert commit |
| P2 | DONE | deterministic single/batch render | 44 focused/existing colour tests pass | `cabe6fc` | revert commit |
| P3 | DONE | JSON roundtrip and replay | 51 focused/existing colour tests pass | `8c7d45e` | revert commit |
| P4 | DONE | file-level SDR image adapter | 67 focused/preprocess tests pass | `23221fa` | revert commit |
| P5 | DONE | regression/integration evidence | 84 focused tests; full-suite result classified | `8d3f60c` | release claim |

At most one row may be `IN_PROGRESS`. A row becomes `DONE` only when its
verification evidence and commit are recorded in the branch log.

## First-slice algorithm

The initial baseline deliberately reuses the established safe-Lab kernel:

1. validate the reference colour state and working space;
2. convert reference pixels to D65 Lab without clipping;
3. freeze reference Lab mean/std and guarded policy parameters into a recipe;
4. for each source, compute its own full-image Lab normalization context;
5. apply the same recipe target through `apply_safe_lab_transform`;
6. apply the selected existing destination-gamut compression policy;
7. convert back to the source working space and return a new `WorkingImage`.

This is a deterministic statistical baseline, not machine learning. It provides
the product API, replay and safety shell needed for future W1-validated
descriptors or parameter predictors without claiming that mean/std statistics
are the final photographic algorithm.

## Success criteria

- one reference and N>=1 sources produce N outputs in source order;
- output shapes and metadata remain source-specific;
- the fitted reference recipe is identical regardless of source batch;
- repeated fit/serialize/load/render produces identical float32 bytes;
- unsupported transfer state, working space, empty batch, non-finite data,
  invalid recipe or out-of-gamut source fails closed;
- relevant existing colour/preprocess tests remain green;
- no forbidden/concurrent file is changed.

## Failure and branch rules

- If safe-Lab cannot preserve exact identity under a neutral recipe, record the
  discrepancy and do not hide it with output clipping.
- If gamut compression cannot start because a source is outside its declared
  destination gamut, reject the source rather than silently clamp.
- If W1 later shows the reference representation is unidentified, the product
  may retain this deterministic baseline but must expose uncertainty/strength
  rather than claim recovered look.
- If D-PCT later publishes a stable shared media/recipe contract, integrate
  through an adapter leaf instead of copying its implementation.

## Execution evidence

- P0: coordination and rollback contract committed as `ec001b2` and
  whitespace normalization as `acb0c82`.
- P1: `30 passed` across `test_color_match_contracts.py`,
  `test_color_engine_lab.py` and `test_color_engine_gamut.py` using the
  project's existing Python 3.12 virtual environment.
- P2: `44 passed` after adding deterministic single/batch rendering, including
  repeated-byte identity, batch-order independence, input non-mutation,
  cross-working-space execution and fail-closed source boundaries.
- P3: `51 passed` after atomic recipe persistence and file replay. Loaded
  recipes reproduce in-memory batch output bytes and diagnostics exactly.
- P4: `67 passed` after adding transactional PNG/JPEG/TIFF SDR file matching,
  8/16-bit outputs and real preprocessing regressions. Repeated output files
  and recipes are byte-identical; a late invalid source leaves no staged or
  committed partial batch.

### Pre-existing worktree-line-ending failure

The unrelated `tests/test_render_contract.py` has five failures in this
worktree because `configs/color_rendering_profiles.yaml` is checked out with
48 CRLF line endings. Its worktree SHA-256 is `a29663b2...`, while the Git blob
and tracked profile manifest both use the LF-byte SHA-256 `d919402a...`. No
reference-match commit modifies either file. This branch records the failure
but does not rewrite protected legacy profile hashes or shared renderer assets.

The complete CPU collection later reached `939 passed, 1 skipped, 36 failed`.
All failures were either the same checked-out-byte hash class or tests whose
ignored `outputs/` evidence is not copied into a new Git worktree. No
`src/color_match` test failed and no new failure family appeared.

## Phase-two gates

The product shell is ready, but the statistical recipe is not frozen as the
final photographic/aesthetic algorithm.

| Gate | Status | Dependency | Allowed next action |
|---|---|---|---|
| A1 reference identifiability | BASELINE FAILED | local known-operator cross-content falsification; main-chat W1 single/multi/paired evidence remains active | replace or augment the recipe descriptor only if hidden-operator/source-use gates pass |
| A2 film-business composition | WAITING | stable stock-expert product interface | add an adapter that keeps `reference-look` and `film_stock_id` claims separate |
| A3 media portability | WAITING | standalone D-PCT stable recipe/media contract | map recipe fields through an adapter; do not copy RAW/HDR/video code |
| A4 photographic preference | BASELINE REJECTED | six-image known-operator slice completed; broader frozen suite and blind review remain open | compare identified challengers under severe-artifact veto and blind aesthetic review |

Until A1/A4 pass, this implementation is an operational deterministic baseline,
not the claimed final or strongest colour-matching algorithm.

The first A4 falsification uses one known Velvia-look target as the reference
and applies its recipe to six neutral sources. The same-content positive control
improves median Delta E76 by 59.8%, but every held-out-content image regresses
(-3.8% to -175.4%). See
`docs/drpt/REFERENCE_COLOR_MATCH_QUALITY_BASELINE.md`. This rejects global
reference moments as the final algorithm and prevents parameter-tuning from
being mistaken for content-independent look recovery.

## Verification and rollback

Focused tests run before broad tests. Each verified leaf receives a scoped
commit. No generated images, datasets, weights or output caches are committed.
Rollback is commit-level and never requires resetting or rewriting another
chat's work.
