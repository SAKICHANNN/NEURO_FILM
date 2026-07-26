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
| P1 | DONE | `LookRecipe` types, fitting and validation | 30 focused/existing colour tests pass | contract commit | revert commit |
| P2 | IN_PROGRESS | deterministic single/batch render | identity/reference target tests, N-order independence, no mutation | renderer commit | revert commit |
| P3 | NOT_STARTED | JSON roundtrip and replay | byte-identical recipe JSON and output replay | replay commit | revert commit |
| P4 | NOT_STARTED | file-level SDR image adapter | focused PNG/JPEG/TIFF smoke, output/profile verification | adapter commit | revert commit |
| P5 | NOT_STARTED | regression/integration evidence | focused plus relevant existing tests, full CPU suite if feasible | evidence commit | release claim |

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

## Verification and rollback

Focused tests run before broad tests. Each verified leaf receives a scoped
commit. No generated images, datasets, weights or output caches are committed.
Rollback is commit-level and never requires resetting or rewriting another
chat's work.
