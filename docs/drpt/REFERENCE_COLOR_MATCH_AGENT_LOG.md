# Reference Color Match Agent Log

## 2026-07-27 - Start product reference-look Goal

- Node/parent goal: NFCM-P1 / neuro-film uploaded reference-photo matching.
- Trigger: user explicitly requested implementation in this chat using Goal.
- Skills used: `dev-research-reliability` primary; router, AI/ML, DRPT,
  plan-tracker, project-log and structure stewardship as read-only governance.
- Decisions: use isolated branch/worktree; implement image-first
  `WorkingImage -> LookRecipe -> batch WorkingImage`; do not duplicate D-PCT
  media/video work or main-chat W1/W2 research.
- Files changed: coordination record, product plan and this branch log only.
- Verification evidence: both active task states and three Git worktrees were
  read before the claim; this worktree began clean at `c03c321...`.
- Risks or unknowns: safe-Lab mean/std is only a deterministic baseline, not
  evidence of final photographic/aesthetic superiority.
- Handoff state: P0 active; implementation remains gated on committing the
  coordination contract.

## 2026-07-27 - Freeze coordination contract

- Changed: P0 coordination, plan and dedicated branch log committed as
  `ec001b2`; EOF whitespace normalized in `acb0c82`.
- Evidence: `git diff --check` passes after the normalization commit.
- Next: P1 implements only the immutable recipe/fitting contract and focused
  tests under the claimed file scope.

## 2026-07-27 - Complete P1 recipe contract

- Changed: added immutable v1 reference-look policy/recipe, canonical SHA-256
  identity, strict JSON roundtrip and a `WorkingImage`-only fitting entrypoint.
- Evidence: 30 tests pass across the new contract tests and existing Lab/gamut
  regressions. System Python lacked pytest; tests used the project's existing
  Python 3.12 `.venv` without installing dependencies.
- Risks: the recipe currently freezes mean/std Lab statistics; it is explicitly
  labeled a deterministic statistical baseline.
- Next: P2 adds deterministic single/batch application without changing this
  recipe contract.

## 2026-07-27 - Complete P2 batch renderer

- Changed: added deterministic safe-Lab application, explicit source/chroma
  gamut policy, ordered N-source batch API and per-source diagnostics.
- Evidence: 44 tests pass across reference-match contracts/rendering and
  existing Lab/gamut tests. Repeated rendering is byte-identical; batch order
  does not affect any source result; inputs and metadata containers are not
  mutated.
- Boundaries: display-linear SDR only; scene-linear RAW and out-of-gamut source
  pixels fail closed.
- Next: P3 persists recipe files atomically and proves load/replay identity.

## 2026-07-27 - Complete P3 atomic replay

- Changed: added bounded UTF-8 recipe loading, atomic JSON save and recipe-file
  batch replay using the existing inference atomic writer.
- Evidence: 51 focused/existing colour tests pass. Repeated recipe writes are
  byte-identical; loaded-recipe render bytes and diagnostics equal in-memory
  execution.
- Existing failure isolated: the unrelated render-contract suite reports five
  asset-hash failures because this worktree converts the tracked LF YAML blob
  (`d919402a...`) to CRLF bytes (`a29663b2...`). Neither asset nor manifest was
  changed by this branch.
- Next: P4 adds an SDR file adapter through existing preprocessing/output
  boundaries without absorbing RAW/HDR/media work.

## 2026-07-27 - Complete P4 SDR file adapter

- Changed: added one-reference/N-source PNG/JPEG/TIFF adapter, deterministic
  8-bit and 16-bit SDR output, per-file hashes and hidden staging cleanup.
- Evidence: 67 tests pass across all reference-match modules and existing
  preprocessing. Repeated outputs and recipe files are byte-identical. A
  deliberately invalid late source leaves no output, recipe or staging file.
- Boundaries: file adapter accepts only display-linear linear-sRGB sources;
  16-bit JPEG is rejected; RAW/HDR/video/media remain owned by the standalone
  D-PCT work.
- Next: P5 refreshes concurrent task claims, runs broader CPU regression and
  publishes the integration evidence bundle.

## 2026-07-27 - Complete P5 integration evidence

- Changed: added rollback-safe multi-file commit. A final replacement failure
  restores every pre-existing output/recipe and removes all stage/backup files.
- Evidence: 84 focused/preprocess/Lab/gamut tests pass, including injected
  second-output commit failure. `compileall` and `git diff --check` pass.
- Broad regression: complete CPU collection produced 939 passed, 1 skipped and
  36 failures. Failures are confined to missing ignored `outputs/` evidence and
  CRLF-changed tracked-byte hashes in the worktree; no reference-match test or
  new failure family failed.
- Concurrent state: neuro-film main remains on W1/W2/S4 files; standalone
  D-PCT is clean at `89a1953`; neither overlaps this branch's files.
- Handoff state: first product vertical slice is ready for integration review.
  Phase-two algorithm promotion remains gated on W1 identifiability and an
  aesthetic/preference evaluation rather than being declared final.

## 2026-07-27 - Reject v1 as the final photographic matcher

- Node: NFCM-A4 known-operator cross-content falsification.
- Coordination refresh: the main task remains active on W1 reference
  identifiability; standalone D-PCT is active on corrected RAW/DNG evidence.
  This leaf reads their state but does not write either task's files.
- Method: fit one recipe from Velvia-look target `01`, then reuse it for six
  neutral real-photo sources with corresponding same-look targets available
  only for evaluation.
- Evidence: same-content `01` improves median Delta E76 by 59.8%; all five
  cross-content rows regress, from -3.8% to -175.4%. Visual inspection confirms
  exposure drift, sky/neutral contamination and content-palette leakage without
  spatial corruption.
- Decision: v1 remains the deterministic fallback/API shell but is rejected as
  the final or strongest matcher. Do not spend the next leaf tuning global
  moments.
- Change propagation: A1 is now `BASELINE FAILED`; A4 is
  `BASELINE REJECTED`. A promotable algorithm must use a canonical or otherwise
  identified content-independent grade and retain the explicit bounded
  operator/replay contract.
- Next: consume the main task's committed W1 evidence to select or reject a
  stronger descriptor/head; meanwhile define only non-overlapping product
  composition and adapter boundaries.

## 2026-07-27 - Complete A2 composition contract

- Changed: added a versioned `ReferenceCompositionPlan` and immutable film
  effect provenance binding.
- Product decision: uploaded reference matching and film-profile selection are
  peer colour sources, not two implicit colour stages. Reference mode may add
  film-derived procedural effects only; this never claims a film-stock
  identity.
- Evidence: 66 focused colour-match/Lab/gamut tests pass. Tests prove stable
  plan identity, exact stage order, profile/hash requirements and fail-closed
  rejection of hidden film colour stacking or stock-claim escalation.
- Structure: the adapter lives in `src/color_match`; it consumes the existing
  render-profile validator without changing `src/inference`, the renderer
  default, stock profiles or FilmFX.
- Next: keep W1 descriptor/head consumption active and defer runtime FilmFX
  execution until the existing product renderer exposes a stable
  `WorkingImage` effect-only entrypoint.

## 2026-07-27 - Freeze A4 known-operator evaluation API

- Changed: added a pure `WorkingImage` evaluator for neutral source, known
  same-content target and candidate triplets.
- Metrics: median/p95 Delta E76, relative median improvement and newly
  introduced output-boundary fraction per sample; batch summaries preserve
  improved/regressed counts and worst tail.
- Evidence: 69 focused colour-match/Lab/gamut tests pass. The committed harness
  exactly reproduces the six-image baseline: one improved, five regressed,
  aggregate median -45.6%, worst -175.4%, maximum new boundary 10.9%.
- Claim boundary: targets are evaluation-only and automated metrics cannot
  substitute for severe-artifact and photographic-preference review.
- Next: use this fixed product gate for any W1-derived challenger after its
  source task commits and passes its own hidden-operator/source-use gates.

## 2026-07-27 - Add default identity fallback for tail failures

- Changed: added `reference-render-guard.v1`, guarded single/batch replay and
  default file-adapter enforcement.
- Frozen thresholds: gamut-adjusted fraction <=25% and newly introduced
  encoding-boundary fraction <=5%. Threshold failure returns a copied source
  image, never the rejected candidate, and records deterministic reasons.
- Real-slice evidence: the guard accepts reference-content positive control
  `01` and rejects every held-out-content regression (`02`, `03`, `05`, `07`,
  `09`).
- Verification: 96 reference-match/preprocess/Lab/gamut tests pass, including
  exact guarded recipe replay, ordered batch decisions, identity non-mutation
  and file-result safety evidence.
- Interpretation: the guard prevents delivery of known severe colour failures;
  it does not improve v1 look identification and therefore does not close A1
  or the photographic-preference gate.

## 2026-07-27 - Calibrate guard on a 6x6 reference/source matrix

- Method: rotate all six known Velvia-look targets through the reference role
  and render all six neutral sources, yielding six same-content controls and 30
  cross-content comparisons.
- Algorithm result: every same-content control improves (median +60.0%); only
  5/30 cross-content combinations improve, with cross-content median -92.2%
  and worst -344.9%.
- Guard result: zero of 25 cross-content regressions is accepted. The guard
  rejects one same-content improvement and two cross-content improvements, so
  it is correctly classified as a conservative veto rather than an aesthetic
  ranker.
- Threshold decision: do not raise the 5% new-boundary gate to rescue the one
  positive-control rejection; the required 11.8% allowance would also admit
  known regressions.
- Next: a W1-derived content-independent descriptor/head must improve the
  cross-content distribution itself; safety fallback cannot substitute for
  algorithm identification.
