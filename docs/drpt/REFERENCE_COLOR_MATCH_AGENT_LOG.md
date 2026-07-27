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

## 2026-07-27 - Map D-PCT media contract without copying execution

- Source refresh: standalone D-PCT commit `413d713...`; its committed
  `contracts.py` and media-frame schema are unchanged while the task develops
  a separate lossless-JPEG DNG subset.
- Mapping: D-PCT scene rail is linear ACEScg/D60 scene-relative; display rail
  is linear absolute XYZ/D65 with explicit reference-white nits. Current NFRM
  is relative linear-sRGB SDR.
- Decision: A3 is `CONTRACT MAPPED / PIXEL BRIDGE CLOSED`. Re-labeling a frame
  is forbidden; a future adapter needs a versioned pixel/render bridge,
  luminance scale and full provenance.
- Ownership: RAW/HDR/video decoding, DNG entropy work and cross-platform media
  remain entirely in D-PCT. No decoder code or mutable D-PCT file was copied.

## 2026-07-27 - Refresh complete CPU regression

- Result: `952 passed, 1 skipped, 36 failed` in 79.40 seconds.
- Delta: 13 additional tests pass relative to the prior branch snapshot; the
  36 failures are the same missing ignored-output and CRLF frozen-byte-hash
  families already classified at P5.
- No `src/color_match` test and no new failure family appears.
- Decision: preserve historical assets and hashes; do not rewrite unrelated
  evidence to make a secondary worktree's full suite artificially green.

## 2026-07-27 - Add executable CLI and provenance report

- Changed: added `scripts/match_reference_color.py` and versioned
  `neuro-film.reference-match-report.v1`.
- Contract: one reference, repeated source/output pairs, one recipe and one
  report. The report binds reference file/pixel identity, every source/output
  hash, candidate diagnostics, guard policy and fallback reasons.
- Safety: source/output cardinality mismatch returns exit code 2 and leaves no
  output, recipe or report. Reports are written atomically and cannot overwrite
  a reference, source, output or recipe.
- Verification: 103 focused reference-match/preprocess/Lab/gamut tests pass,
  including subprocess CLI execution with two source images and exact repeated
  report bytes.
- Structure: CLI is deliberately thin; no renderer default, codec, W1/S4 or
  D-PCT file changed.

## 2026-07-27 - Run full-resolution CLI delivery smoke

- Inputs: known Velvia-look reference `01`, same-content neutral source `01`
  and held-out-content neutral source `02` from the existing ignored baseline
  artifacts.
- Result: two 16-bit PNGs, one recipe and one report were created through the
  public CLI in 6.8 seconds.
- Safety: source `01` applied; source `02` identity-fell back for both
  `gamut-adjusted-fraction` and `new-boundary-fraction`.
- Report: schema `neuro-film.reference-match-report.v1`, two outputs, report
  SHA-256 `13560e40...014c1` after portable identity finalization; every
  input/output/recipe hash and decision is
  present.
- Artifacts remain ignored under
  `outputs/reference_color_match_cli_smoke`; nothing generated was committed.

## 2026-07-27 - Reject full-covariance Gaussian/MKL challenger

- Challenger: per reference/source D65 Lab symmetric positive-definite
  Gaussian optimal transport, 85% luma strength, existing source-relative
  gamut compression and frozen guard.
- Same-content result: 6/6 improve, median +65.2%, but only 4/6 pass the guard.
- Cross-content result: 3/30 improve and 27/30 regress; median -109.5%, worst
  -332.2%, only two candidates pass the guard.
- Comparison: v1 mean/std is still poor but reaches 5/30 improvements and
  -92.2% median. Covariance fitting therefore worsens the actual target slice.
- Decision: do not implement Gaussian/MKL as a product recipe and do not rescue
  global distribution matching with more histogram/moment capacity. Await an
  identified content-independent grade representation from W1.

## 2026-07-27 - Add language-neutral cross-platform schemas

- Changed: added strict JSON Schema 2020-12 contracts for reference recipe,
  composition plan and run report; added composition JSON encode/decode.
- Cross-field gates: reference colour remains the sole colour owner; film
  effects and execution order must agree; safety acceptance, action and reasons
  must agree.
- Validation: schemas pass Draft 2020-12 meta-validation. Real Python payloads
  validate; unknown fields, invalid claims and inconsistent state fail.
- Verification: 105 focused reference-match/preprocess/Lab/gamut tests pass;
  compile and diff checks pass.
- Product impact: Android/iOS/macOS/Windows bindings can implement the payload
  contract without importing Python dataclasses. Pixel execution remains a
  later native/runtime concern and is not claimed by schema compatibility.

## 2026-07-27 - Finalize portable canonical identity

- Defect found: recipe/plan IDs previously hashed Python's JSON float text,
  which is deterministic in Python but not a language-neutral wire contract.
- Changed: added typed canonical bytes with explicit null/bool/int/string/list/
  sorted-object tags and big-endian IEEE-754 binary64 hex for finite floats.
- Frozen vector: multilingual/multitype payload SHA-256
  `283b5aeb...188a5`; integer/float and positive/negative zero remain distinct.
- Verification: 112 focused tests pass. Full-resolution CLI repeats at one
  applied/one fallback, recipe ID `12c084ad...7cacd`, report SHA-256
  `13560e40...014c1`.
- Compatibility: no recipe artifact is committed or released from this branch,
  so finalizing v1 now does not invalidate an external consumer. Pixel output
  and guard decisions are unchanged.
- Commit: `e1f03f6` (`fix: make reference identities language neutral`).

## 2026-07-27 - Make photographic promotion fail closed

- Parent: P9 / A4 photographic preference and severe-tail evidence.
- Changed: added a frozen structural-colour probe, multi-recipe tail
  aggregation, known-operator streaming aggregation, blind-review evidence
  contract and a decision function that cannot promote from automated metrics
  alone.
- Reliability correction: both known-operator and probe validation now allow
  only `1e-6` float32 gamut roundoff while raw values still participate in
  boundary diagnostics. This admits the renderer's observed
  `1.000000119...` output without clipping and rejects excursions above the
  tolerance.
- Runner: `scripts/evaluate_reference_match_promotion.py` fits one recipe per
  reference, streams cross-content pairs, writes atomically, binds file hashes
  and excludes local paths from canonical report identity.
- Formal evidence: 30 cross-content rows reproduce 5 improved/25 regressed,
  median `-0.921932`, worst `-3.448730`, maximum new boundary `0.218555`.
  Two of six recipe probes pass; worst probe new boundary is `0.126946`.
- Decision: baseline `rejected`; no blind review opens. Report ID
  `7e22df00...5c996`, report SHA-256 `02483ad2...9534`.
- Verification: 125 focused colour-match/preprocess/colour-engine tests pass;
  the CLI report repeats exactly on identical inputs, remains identity-stable
  after directory relocation, and validates against strict JSON Schema
  2020-12. The full collection is 981 passed, one skipped and the same 36
  known failures from absent ignored outputs and CRLF-sensitive frozen hashes;
  no colour-match or adjacent validation test fails.
- Commit: `8e03ed7` (`feat: gate reference matcher promotion`).

## 2026-07-27 - Correct the meaning of batch consistency

- Parent: P10 / A5 album and batch consistency.
- Defect: prior tests proved deterministic replay, one immutable recipe and
  source-order independence, but not consistent mapping of the same colour
  across different source-image contexts.
- Changed: added two frozen 32x48 probes with one exactly shared colour chart
  inside unrelated dark/cool and bright/warm surroundings. Added single- and
  multi-recipe median/p95/max Delta E76 gates at `0.5/1.0/3.0`, a
  future-renderer output adapter, promotion-decision integration and strict
  report-schema coverage.
- Formal result: all six Velvia-reference baseline recipes fail. Worst
  shared-colour median/p95/maximum drift is
  `60.8633/77.3292/80.1493` Delta E76.
- Decision: v1 remains deterministic and order-stable but is not strongly
  album-consistent. Reusing a recipe ID cannot substitute for a fixed explicit
  operator or a bounded adaptation that passes the context gate.
- Current report: ID `2f7b8b2c...736411`, SHA-256
  `40d52826...213a1`; promotion remains `rejected`.
- Verification: 131 focused colour-match/preprocess/colour-engine tests pass.
  The full collection is 987 passed, one skipped and the unchanged 36 known
  ignored-output/CRLF-hash failures; no new failure family appears.
- Commit: `4bfd5da` (`feat: enforce reference batch consistency`).

## 2026-07-27 - Fail closed on the rejected product algorithm

- Parent: P11 / product delivery certification boundary.
- Problem: A1, A4 and A5 all reject the only implemented algorithm, but the
  default file adapter could still apply it when per-image pixel guards passed.
- Changed: `reference-render-guard.v2` defaults to
  `algorithm-not-promoted` and identity delivery. File API, guarded replay,
  CLI, report payload and strict schema propagate the decision.
- Research boundary: `--allow-research-baseline` is an explicit opt-in. Every
  output records `research_baseline_override=true`; gamut/new-boundary vetoes
  remain active.
- Full-resolution evidence: default mode is 0 applied / 2 identity fallback,
  report SHA-256 `1740d166...0f25a`; research override is 1 applied / 1
  fallback, report SHA-256 `47e38275...c7c62`.
- Decision: the module remains executable and replayable for research but can
  no longer present rejected safe-Lab output as the default deliverable look.
- Verification: 134 focused tests pass; the full collection is 990 passed,
  one skipped and the unchanged 36 known ignored-output/CRLF-hash failures.
- Commit: `adae6cb` (`fix: fail closed on unpromoted matcher`).

## 2026-07-27 - Propagate certification into film composition

- Parent: P12 / A2 delivery-aware composition boundary.
- Defect: after guard v2 defaulted the renderer to identity, the composition
  payload could still label the same recipe `reference-look+film-effects`.
- Changed: composition v1 now binds `reference_color_status` and
  `research_baseline_override`. Default is owner/claim/label `identity`,
  order `identity_color`, with no film effects.
- Fail-close: requesting film effects without available reference colour
  raises and routes the product to its separate film-simulation mode.
- Research boundary: explicit override restores `reference_color` and may bind
  verified procedural effects, while film colour and stock claims remain
  forbidden.
- Validation: Python roundtrip and strict JSON Schema enforce the same
  cross-field state matrix.
- Verification: 136 focused tests pass; the full collection is 992 passed,
  one skipped and the unchanged 36 known ignored-output/CRLF-hash failures.
- Commit: `ba6f2c1` (`fix: propagate matcher certification to composition`).
