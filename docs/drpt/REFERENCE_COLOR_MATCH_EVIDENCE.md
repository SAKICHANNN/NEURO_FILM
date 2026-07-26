# NFCM-P1 Reference Color Match Evidence Bundle

## Leaf and parent

- Leaf: NFCM-P1 product reference-look engine
- Parent: neuro-film uploaded reference-photo matching
- Branch: `codex/reference-color-match`
- Worktree: `C:\Users\hhvrf\Documents\neuro_film_color_match`
- Base commit: `c03c321b9fc642e2e092d59e20dd1b145b96192d`

## Delivered interface

```text
fit_reference_look(reference WorkingImage)
    -> immutable ReferenceLookRecipe v1

render_reference_batch(recipe, N source WorkingImages)
    -> N ordered ReferenceMatchResult values

match_reference_files(reference path, N source paths, N output paths)
    -> one recipe + N committed SDR files + hashes/diagnostics

build_reference_composition(recipe, optional verified film profile)
    -> one colour owner + optional film-effects-only provenance
```

The recipe claim ceiling is `reference-look` and its evidence grade is
`deterministic-statistical-baseline`. It never identifies a film stock or
permits `calibrated-reference`.

## Files added

- `src/color_match/contracts.py`
- `src/color_match/fit.py`
- `src/color_match/render.py`
- `src/color_match/replay.py`
- `src/color_match/files.py`
- `src/color_match/composition.py`
- `src/color_match/__init__.py`
- `tests/test_color_match_contracts.py`
- `tests/test_color_match_render.py`
- `tests/test_color_match_replay.py`
- `tests/test_color_match_files.py`
- `tests/test_color_match_composition.py`
- branch-specific coordination, plan, log and this evidence record

No forbidden W1/W2/S4, global tracker, renderer-default or standalone D-PCT
file was changed.

## Commits

- `ec001b2` - claim reference color match product leaf
- `acb0c82` - normalize plan whitespace
- `1820af1` - add replayable reference look recipe
- `cabe6fc` - render one reference look across image batches
- `8c7d45e` - persist and replay reference look recipes
- `23221fa` - match uploaded reference across SDR image files
- `8d3f60c` - certify integration boundary and rollback safety

## Verification

- P1: 30 contract/Lab/gamut tests passed.
- P2: 44 contract/render/Lab/gamut tests passed.
- P3: 51 contract/render/replay/Lab/gamut tests passed.
- P4: 67 reference-match/preprocessing tests passed.
- P5: 84 focused/preprocessing/Lab/gamut tests passed.
- Python bytecode compilation passed for `src/color_match`.
- `git diff --check` passed.
- Injected final-commit failure restored old output A, old output B and old
  recipe, with no stage/backup residue.

The full CPU collection produced 939 passed, 1 skipped and 36 failures. The
failures are pre-existing worktree conditions:

1. historical tests require ignored `outputs/` artifacts that are available in
   the main checkout but not copied into this worktree;
2. Windows CRLF checkout changes byte hashes frozen against LF Git blobs.

For example, `configs/color_rendering_profiles.yaml` has Git-blob SHA-256
`d919402a...`, matching its manifest, but worktree SHA-256 `a29663b2...` after
48 LF endings become CRLF. This branch did not change the asset or manifest.

## Assumptions and interpretation limit

- First-slice file input is SDR display-linear after existing preprocessing.
- The file adapter currently exports linear-sRGB sources only.
- Mean/std safe-Lab is a deterministic baseline and product integration shell,
  not evidence of strongest photographic or aesthetic matching.
- One recipe is fixed across a batch; per-source normalization is the existing
  safe-Lab content adaptation.
- RAW/HDR/video and portable media execution remain owned by the standalone
  D-PCT project.
- Single-reference identifiability and stronger content-invariant descriptors
  remain owned by the neuro-film main W1/W2 research.

## Phase-two quality finding

`docs/drpt/REFERENCE_COLOR_MATCH_QUALITY_BASELINE.md` records the first
known-operator cross-content test. The v1 recipe improves its same-content
positive control but regresses all five held-out-content images. It is
therefore certified only as a safe operational fallback and contract baseline,
not as the final photographic matcher.

## Film composition finding

The A2 composition contract prevents reference colour and film-profile colour
from being silently stacked. A verified film profile can bind grain, halation
and dust defaults as effect provenance, but the output remains
`reference-look`, does not claim `film_stock_id`, and executes reference colour
before effects. The current leaf validates the contract; it does not duplicate
or bypass the existing FilmFX renderer.

## Change propagation

- Upward: neuro-film now has an implementable peer capability to stock
  selection, labeled `reference-look`.
- Downward: future descriptor/parameter heads can replace fitting internals
  while retaining recipe, replay, batch and file contracts.
- Sideways: stock simulation and D-PCT integration must use adapters and retain
  separate claim/evidence fields.
- Validation: any algorithm promotion must retain all current deterministic,
  gamut, replay, batch-order and rollback tests, then add aesthetic/severe-tail
  evidence.

## Rollback and integration recommendation

Every implementation leaf is a separate commit and can be reverted in reverse
order. Integrate only after the main development owner reconciles its current
dirty W1/W2/S4 state. Do not copy the standalone D-PCT repository into this
module; connect it later through a versioned recipe/media adapter.
