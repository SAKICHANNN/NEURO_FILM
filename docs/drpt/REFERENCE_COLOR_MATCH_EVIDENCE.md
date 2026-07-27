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

match_reference_color.py + build_file_match_report(result)
    -> executable batch + deterministic hash/diagnostic/safety report
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
- `src/color_match/evaluation.py`
- `src/color_match/safety.py`
- `src/color_match/reporting.py`
- `scripts/match_reference_color.py`
- `configs/schemas/reference_look_recipe_v1.schema.json`
- `configs/schemas/reference_composition_v1.schema.json`
- `configs/schemas/reference_match_report_v1.schema.json`
- `src/color_match/__init__.py`
- `tests/test_color_match_contracts.py`
- `tests/test_color_match_render.py`
- `tests/test_color_match_replay.py`
- `tests/test_color_match_files.py`
- `tests/test_color_match_composition.py`
- `tests/test_color_match_evaluation.py`
- `tests/test_color_match_safety.py`
- `tests/test_color_match_reporting.py`
- `tests/test_color_match_schemas.py`
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

The latest full CPU collection produced 952 passed, 1 skipped and 36 failures.
The failures are pre-existing worktree conditions:

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
- One recipe is fixed across a batch, but the existing safe-Lab per-source
  normalization fails shared-colour context invariance. The baseline is
  deterministic/order-stable, not strongly album-consistent.
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

## Promotion metric boundary

`evaluate_known_operator_batch` is the frozen A4 automated comparison surface.
It fails closed on alignment, display-linear state, working space, finite range
or duplicate-ID violations and reports both centre and tail Delta E76 plus new
boundary pixels. Its result is evidence for candidate rejection or progression
to visual review, never a standalone photographic-aesthetic promotion.

## Product tail guard

The file adapter and guarded replay now use fixed
`reference-render-guard.v1`. A candidate exceeding 25% gamut-adjusted pixels or
5% newly introduced boundary pixels is not written as the delivered look;
instead, an identity copy is encoded and the candidate diagnostics plus
rejection reasons remain in the result. This policy catches all five
cross-content failures in the frozen baseline and accepts its positive control.

## D-PCT integration boundary

At committed standalone D-PCT state `413d713...`, its canonical scene rail
(linear ACEScg/D60 scene-relative) and display rail (linear absolute XYZ/D65
with reference-white nits) are semantically different from this module's
relative linear-sRGB SDR rail. A3 is therefore contract-mapped but intentionally
closed at the pixel bridge. This branch contains no D-PCT RAW, HDR, DNG, video
or platform decoder implementation.

## Executable delivery evidence

The CLI exercises the real file adapter in a subprocess with one reference and
two ordered sources. It writes two 16-bit outputs, one replayable recipe and one
atomic report. A cardinality mismatch returns code 2 with no partial artifact.
The report is byte-repeatable and covers all run inputs, outputs, candidate
diagnostics and product safety decisions.

A full-resolution smoke on the existing Velvia-look baseline also passes:
`applied_count=1`, `identity_fallback_count=1`, `output_count=2`, with report
SHA-256 `13560e40...014c1` after portable identity finalization. The fallback
row records both frozen guard reasons.

## Rejected stronger statistical comparator

Full-covariance Gaussian/MKL transport was run on the frozen 6x6 known-look
matrix and rejected. It improves the same-content median but worsens
cross-content median recovery to -109.5%, with 27/30 regressions. This evidence
closes stronger unpaired global moments/distribution fitting as the next
product algorithm; it does not close canonicalized, paired-supervised or
otherwise identified explicit-operator prediction.

## Executable photographic promotion gate

The promotion gate now streams known-operator rows, aggregates centre/tail
metrics, evaluates every independently fitted recipe on a frozen neutral/tone/
skin/sky/foliage probe, and requires a separate blinded aesthetic record before
promotion. Automated evidence can reject or return
`eligible-for-visual-review`; it cannot return `promoted` without at least 12
blinded rows, preference above 50% and zero severe artifacts.

The six-reference baseline formal report is `rejected`: 5/30 cross-content
improvements, -92.19% median, -344.87% worst and 21.86% maximum new-boundary
fraction. Four of six recipe probes also fail the 5% new-boundary tail, with a
12.69% worst case. All six fail shared-colour context invariance; worst
median/p95/maximum drift is `60.86/77.33/80.15` Delta E76. Two full-resolution
pre-portability runs were byte-exact; the current content-bound report passes
its strict schema with ID `2f7b8b2c...736411`. Relocating identical fixtures
changes recorded paths but not report identity.

## Cross-platform payload boundary

Recipe, composition and report now have strict JSON Schema 2020-12 definitions.
The schemas are meta-validated and checked against real generated payloads.
They forbid additional properties and bind hashes, ranges, claims, composition
order and safety state. Python additionally verifies canonical IDs and
relational numeric constraints.

Recipe and composition IDs now hash a documented typed canonical byte stream,
not serialized JSON text. A frozen cross-language vector covers Unicode,
null/bool/int/binary64/list/object behavior and rejects NaN, infinity, byte
strings and non-string keys.

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
