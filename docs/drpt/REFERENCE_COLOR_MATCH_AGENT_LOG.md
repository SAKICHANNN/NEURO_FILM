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

## 2026-07-27 - CFSM explicit-operator challenger

- Parent: P13 / A1+A4+A5 algorithm frontier.
- Added: isolated CFSM v0 fit/render/replay candidate, canonical candidate
  identity, tamper validation, fixed batch operator, boundary-pinned residual
  and deterministic numerical projection.
- Synthetic gates: deterministic non-identity fit, exact JSON replay,
  constrained LUT, four-of-four known-operator improvement and zero
  shared-colour context drift.
- Frozen real matrix: 19/30 improve, median +3.851%, worst -9.649%, zero new
  boundary; 6/6 photographic and 6/6 context probes pass.
- Repeat: report ID `1ead5590...574c7f` and SHA-256
  `ccf34c09...acf243` are byte-identical across two runs.
- Rejected ablations: neutral-axis limits 0.08/0.12 worsen the tail; using the
  uploaded N-source batch as the prior falls to 16/30, +0.571% median and
  -11.524% worst.
- Decision: v0 and batch-v1 remain research-only/rejected. Default delivery
  remains identity. P14 requires independent canonicalizer evidence, not
  stronger distribution fitting or post-hoc strength tuning.
- Verification before full collection: 6 CFSM tests and 126 focused
  colour-match/preprocess tests pass; compileall and diff check pass.
- Full collection: 998 passed, one skipped and the unchanged 36
  ignored-output/CRLF-hash failures; no new failure family and no
  `src/color_match` failure.
- Commit: `ec437e1` (`feat: add projected CFSM matcher challenger`).

## 2026-07-27 - Refresh stable external boundaries after P13

- Read-only main snapshot: `ed1dbb5` closes S4 as conditional-distribution
  success but hidden-operator failure. No W1 canonicalizer is published.
- Read-only D-PCT snapshot: `e1f67d3` executes 67/67 local compression-7 DNG
  mosaics and observed profile mechanics to ACEScg, but retains
  `real_raw_paths=FAIL`.
- Propagation: P14 cannot consume S4 as identified operator evidence; A3 cannot
  consume the DNG decoder as a trusted MatchView bridge.
- Files imported or copied: none. External dirty work remains untouched.

## 2026-07-27 - Close data-free analytic canonical priors

- Parent: P14A / canonical-prior identification.
- Contract: three analytic priors plus uniform control; disjoint generated
  development/validation/confirmation operators; no real-matrix access before
  synthetic confirmation.
- Development: uniform 43/48 and +9.70% median; best analytic wide-chroma
  25/48 and +0.47%.
- Validation: uniform 19/24 and +6.92%; best analytic wide-chroma 13/24 and
  +1.02%. Decision is `no-analytic-prior-selected`.
- Reproducibility: validation report ID `d08ff8b...53fe2`, SHA-256
  `816ba59e...f6549`, byte-identical twice.
- Propagation: synthetic confirmation and real 30-pair confirmation remain
  closed. P14B requires independent learned/published canonicalizer evidence.
- Verification: 10 dedicated tests pass; compileall and diff check pass. The
  complete CPU collection is 1002 passed, one skipped and the unchanged 36
  known ignored-output/CRLF-hash failures; no color-match test failed.
- Commit: `3debf13` (`feat: test analytic CFSM priors`).

## 2026-07-27 - Add fail-closed W1 development evidence intake

- Parent: P14B0 / independent canonicalizer evidence boundary.
- External stable source: main-task W1 implementation `77df1b9`, currently
  reachable under main head `ed1dbb5`; formal W1 execution remains queued.
- Implementation: pin five critical external source hashes; require two
  byte-identical reports; validate the recorded external commit and source;
  validate gate inventory and recompute the decision branch without importing
  main-task code.
- Product boundary: development evidence never opens integration. A valid
  single-reference unseen-look pass opens only untouched confirmation;
  multi-reference and seen-bank branches do not satisfy arbitrary one-
  reference upload. Every missing/invalid branch delivers identity.
- Current execution: no reports exist, so status is `not-ready`, decision ID
  `26800a73...619f5`, delivery identity.
- Verification: 8 dedicated and 138 focused tests pass; compileall and diff
  check pass. Full CPU collection is 1010 passed, one skipped and the same 36
  ignored-output/CRLF-hash failures; no color-match test failed.
- Commit: `8e602f5` (`feat: gate external W1 matcher evidence`).
- Propagation: P14B1 awaits stable repeated W1 reports. Standalone RAW work is
  still uncommitted and A3 remains closed.

## 2026-07-27 - Refresh stable standalone RAW dependency boundary

- Parent: A3 media portability / same-project multi-chat coordination.
- Read-only source: standalone D-PCT stable commits `12d5767`, `820497e` and
  `77e64e1`; its later 67-file LibRaw agreement expansion is active and
  uncommitted, so it is excluded.
- Stable evidence: 67/67 local compression-7 DNG mosaics and 52/52 observed
  profile-look paths execute; all 17 CR2 entropy paths execute; pinned LibRaw
  unpacks 32/39 vendor RAW files; three DNG storage/bit-depth strata agree on
  28,682,816 post-linearization sensor codes with zero probe/decode errors.
- Remaining boundary: seven Nikon HE/HE* inputs are unsupported and
  `real_raw_paths=FAIL`. Sensor-code agreement does not establish crop, black
  normalization, demosaic, camera/profile colour, a trusted scene renderer,
  perceptual agreement or a versioned scene/display-to-MatchView bridge.
- Propagation: A3 remains `CONTRACT MAPPED / PIXEL BRIDGE CLOSED`; no decoder,
  probe, report or mutable file was copied from the standalone task.

## 2026-07-27 - Freeze portable matcher conformance vectors

- Parent: P15 / cross-platform algorithm delivery evidence.
- Gap closed: prior byte-exact replay evidence covered only the Python
  implementation and could not adjudicate independent mobile/desktop ports.
- Added: exact binary32 reference/source/expected-output vectors, strict
  language-neutral bundle and result schemas, typed bundle identity, a
  product-facing verifier and a deterministic CLI.
- Coverage: linear-sRGB source-segment compression and linear-Rec.2020
  chroma-only compression; both execute full-image reductions and the detail
  path, with non-zero gamut adjustment.
- Gates: exact recipe/reference identity; linear RGB max absolute `2e-5`;
  Delta E76 p95 `0.003`, max `0.01`; diagnostic max absolute `2e-5`.
- Claim boundary: the reference Python implementation passes, but no Android,
  iOS, macOS or independent Windows/macOS port has been tested. The baseline
  remains rejected for final photographic delivery and defaults to identity.
- Verification: 9 dedicated tests and 147 focused colour-match/preprocess
  tests pass; complete CPU collection is 1019 passed, one skipped and the same
  36 ignored-output/CRLF-hash failures.
- External isolation: main remains `ed1dbb5`; D-PCT stable `d0d4e6c` only
  improves probe build reproducibility. No external mutable file or code was
  consumed.
- Commit: `8b505ca` (`feat: add portable matcher conformance vectors`).

## 2026-07-27 - Add transactional stored-recipe file replay

- Parent: P16 / durable LookRecipe delivery.
- Gap: memory replay existed, but the product file API and CLI could not
  process a later N-image batch without reloading the original reference and
  fitting again.
- Added: exact-byte bound recipe loading, `FileReferenceReplayResult`,
  transactional `replay_reference_files`, strict replay-report schema and
  mutually exclusive CLI `--reference`/`--recipe-input` modes.
- Provenance: replay reports bind operation, recipe-file hash, stored
  reference-pixel identity, every source/output hash, candidate diagnostics
  and safety decision; no current reference path is fabricated.
- Safety: recipe/source overwrite attempts reject; tampered recipe rejects
  before staging; a late invalid second source leaves no outputs or stages;
  default replay remains identity unless research override is explicit.
- Verification: 9 dedicated, 40 adjacent and 156 focused tests pass. Full CPU
  collection is 1028 passed, one skipped and the same 36 known
  ignored-output/CRLF-hash failures.
- External isolation: no main W1 file/report or standalone D-PCT code was
  consumed. Their current mutable work remains untouched.
- Commit: `378c846` (`feat: replay matcher recipes across file batches`).

## 2026-07-27 - Consume completed D-PCT corpus evidence read-only

- Parent: A3 / same-project multi-chat dependency closure.
- Stable source: standalone D-PCT `bd3ff70`, clean worktree and completed task.
- Accepted evidence: all 67 local compression-7 DNG files and 686,122,932
  post-linearization sensor codes agree with pinned LibRaw; 14 files execute
  explicit LinearizationTable and 53 use identity; zero failures.
- Boundary retained: this independent comparison ends before crop, black
  normalization, demosaic, profile colour and trusted scene/display rendering.
  Seven Nikon HE/HE* files remain unsupported and `real_raw_paths=FAIL`.
- Propagation: A3 does not open and the reference-match file/replay code does
  not import D-PCT. Only current dependency documentation changes.

## 2026-07-27 - Make complete run artifacts transactional

- Parent: P17 / end-to-end failure closure.
- Gap: image outputs and recipe committed before the CLI attempted the
  provenance report, so report failure could leave a partial successful run.
- Change: optional report paths enter the same stage/hash/backup/replace/
  rollback primitive as N outputs and the fitted recipe; replay uses the same
  path.
- Fault evidence: path collision rejects pre-stage; injected report-builder
  failure commits nothing; injected final report-replace failures restore all
  old fit/replay artifacts; committed report/output hashes match result state;
  no transaction debris remains.
- Verification: 5 dedicated, 32 adjacent and 161 focused tests pass. Full CPU
  collection is 1033 passed, one skipped and the same 36 known
  ignored-output/CRLF-hash failures.
- External isolation: no main-task or D-PCT file was read for implementation
  and no external mutable state was consumed.
- Commit: `2f30826` (`fix: commit matcher run artifacts atomically`).

## 2026-07-27 - Bind composition to actual delivered runs

- Parent: P18 / A2 actual-delivery composition closure.
- Defect: the static plan described requested research mode, but a source
  could still be rejected by gamut/new-boundary safety and delivered as
  identity. The old plan alone could therefore overstate the colour stage.
- Change: added strict run-level composition binding, JSON schema and portable
  canonical identity. The builder verifies the transactional report, recipe,
  output hashes and actual safety actions.
- Batch rule: all-applied may bind reference colour plus optional procedural
  film effects; all-fallback binds only identity/no effects; mixed delivery
  rejects one batch-level plan.
- Replay: both original fit reports and stored-recipe replay reports are
  supported. Original source/reference files may be removed after their
  committed report identities exist, but delivered output tamper fails.
- Verification: 13 dedicated and 174 focused tests pass; compileall and diff
  check pass. Full CPU collection is 1046 passed, one skipped and the same 36
  ignored-output/CRLF-hash failures; no color-match test failed.
- External coordination: main remains active at stable `ed1dbb5` with U1-B
  running and no stable repeated W1 report. D-PCT is idle/clean at `bd3ff70`.
  No external code, report or uncommitted file was consumed.
- Commit: `b44fc96` (`feat: bind composition to delivered matcher runs`).
- Handoff: P18 is complete. P14B1 remains gated on stable repeated W1 evidence;
  A3 remains gated on an independently trusted scene/display-to-MatchView
  bridge.

## 2026-07-27 - Preflight stable-main integration without merging

- Parent: integration evidence / same-project multi-chat isolation.
- Compared both branches from base `c03c321...`: this branch changes 66 paths,
  stable main `ed1dbb5` changes 29, and their exact path intersection is zero.
- A read-only `git merge-tree` contains no conflict marker or Git
  both-modified/both-added conflict record.
- No checkout, merge, rebase, navigation or write occurred in the main task.
  Its U1-B run remains active and unconsumed.
- Limitation: this result applies only to committed `ed1dbb5`; integration must
  rerun after the main task publishes its eventual stable commit.

## 2026-07-27 - Consume and close repeated W1 evidence

- Node/parent goal: P14B1 / A1 reference identifiability.
- Skills: `dev-research-reliability` primary; AI/ML, DRPT-BI, tracker,
  agent-log and structure disciplines read-only.
- Trigger: main stable commit `86b484b` published two byte-identical W1 reports
  after the earlier P14B0 receiver had remained not-ready.
- Change: freeze strict decision JSON/schema; add bounded parsing, semantic
  validation, canonical-ID verification and hostile-type regression tests.
- Evidence: report SHA `9b42e9a8...8dec68`, external software
  `b72b594...`, decision `f8661315...f60e`,
  `paired_upper_bound_only_passes`.
- Interpretation: paired information proves operator/optimizer headroom;
  output-only single/four references fail. No descriptor/capacity rescue,
  confirmation, visual candidate or integration opens.
- Verification: 11 dedicated and 177 focused tests pass; full suite is 1049
  passed, one skipped and the same 36 known failures. Compileall and diff check
  pass.
- Commit: `9cf7d6c` (`feat: freeze repeated W1 matcher decision`).
- Handoff: A1 fixed output-only operator recovery is closed. The next distinct
  leaf must change information/objective, while identity remains the default.

## 2026-07-27 - Audit and close empirical neutral-photo prior

- Node/parent goal: P19 / A1 independent-prior Look Approximation.
- Skills: `dev-research-reliability` primary; AI/ML, DRPT-BI, tracker,
  agent-log and structure disciplines read-only.
- Rights: official FiveK Adobe and Adobe+MIT terms are research-only and
  prohibit commercial advantage. All 128 sources map exactly once (59/69);
  product and commercial integration remain executable false fields.
- Data audit: only `raw_default_srgb16` is read; 128 unique orientation-1
  uint16 RGB files and 201,547,776 pixels pass. Expert/target pixels are not
  read. Three artifact builds repeat at SHA `d590f75a...895d`, prior
  `bb823874...b0ae`.
- Method: equal-image linear-RGB mean/covariance drives the same bounded,
  positive-Jacobian, fixed-LUT CFSM projection. It remains research-isolated.
- Confirmation: empirical reaches 15/48 and median `-6.04%` versus uniform
  31/48 and `+4.63%`; median gain is `-10.67` points and worst loss `5.22`
  points. Two reports repeat byte-exactly at SHA `2a995096...f387`.
- Decision: `empirical-prior-route-closed`. Safety constraints pass, but
  identification fails. More data cannot rescue the same moments.
- Verification: 16 dedicated and 167 focused tests pass; compileall and diff
  check pass.
- Commits: `9d82eda` (`feat: freeze empirical neutral-photo prior`) and
  `e7317f1` (`research: close empirical CFSM prior route`).
- Handoff: continue only with a new objective/information regime; default
  delivery remains identity and no FiveK-derived product asset opens.

## 2026-07-27 - Test and close bounded quantile challenger

- Node/parent goal: P20 / A1 non-moment explicit-operator challenger.
- Hypothesis: alternating affine transport with fixed monotone marginal
  quantiles may recover tone/colour structure missed by global moments.
- Isolation: data-free CPU experiment; no FiveK, W1, D-PCT, training, neural
  weight or mutable external task state consumed.
- Development: 44/48 improve, median `+9.49%`, versus uniform 43/48 and
  `+9.70%`; no tuning follows.
- Stress confirmation: 38/48 improve and median `+12.96%`, versus uniform
  37/48 and `+11.70%`. Gain `+1.26` points misses the frozen `+3` floor;
  worst loss `1.15` points and every structural gate pass.
- Repeat: report ID `785d0d14...a478`, SHA `cf39f27c...1c99` is byte-exact
  across two runs.
- Verification: 14 dedicated and 171 focused tests, compileall and diff check
  pass.
- Commit: `f73e3ef` (`research: test bounded quantile matcher`).
- Decision/handoff: `quantile-route-closed`; no real-photo review or product
  integration. Next work must change objective/information rather than tune
  this estimator.

## 2026-07-27 - Test and close source-batch quantile extension

- Node/parent goal: P21 / batch-conditioned Look Approximation.
- New information: fit may inspect all three synthetic source palettes, but
  one fixed candidate serves the batch and never claims future-album replay.
- Frozen confirmation: unused stress operator indices 8--15, two observations
  each; no development tuning.
- Result: Gaussian and quantile each improve 40/48. Quantile median `+9.13%`
  trails Gaussian `+9.32%` by `0.19` points; its worst row is `1.51` points
  better. Structural gates pass but median-gain gate fails.
- Repeat: ID `e5a2d670...c14a`, SHA `c9d261ad...05b1` byte-exact twice.
- Verification: 12 dedicated and 175 focused tests, compileall and diff check
  pass.
- Commit: `bdffb94` (`research: test batch-conditioned quantile matcher`).
- Handoff: quantile extension closes. Synthetic Gaussian mechanism evidence
  cannot reverse P13's real-matrix rejection or become a reference-only
  reusable product recipe.

## 2026-07-27 - Implement Lab extractor and close linear residual mapper

- Node/parent goal: P22 / learned-parameter Look Approximation intake.
- Research source: July 2026 StatLUT preprint; spatially invariant Lab
  statistics and explicit smooth LUT architecture. No code/weight was found
  or imported.
- Change: implement validated soft L, sqrt-ab and conditional-L|ab features;
  test a generated-data ridge predicting nine bounded operator parameters.
- Confirmation: reserved stress indices 16--31. Median/p90 grid RMSE
  `0.1267/0.1814`, captured style `-104.94%`, identity max `0.1637`,
  same-look `0.1222`; all gates fail.
- Repeat: report ID `f1e2bdb3...41eb`, SHA `0b4cf6c9...8716` byte-exact.
- Verification: 8 dedicated and 183 focused tests, compileall and diff check.
- Commits: `104d7b3` (`feat: add spatially invariant Lab statistics`) and
  `0e8235a` (`research: test Lab statistics residual mapper`).
- Handoff: keep the extractor, close the linear mapper. Do not add capacity on
  the same failed identity/replicate contract without new assets and evidence.

## 2026-07-27 - Reproduce and close obtainable external matchers

- Node/parent goal: P23 / current published reference-grading frontier.
- Sources: official StatLUT paper; official CanonCGT, SA-LUT, NLUT and Neural
  Preset repositories; all external files remain ignored and pinned.
- Rights: CanonCGT Apache-2.0 including weights; NLUT MIT; SA-LUT
  non-commercial S-Lab 1.0; Neural Preset CC BY-NC-SA 4.0. StatLUT has no
  located official implementation/weight.
- CanonCGT: exact 5,056,383-parameter SSL weight load. Published mode improves
  1/30, median `-61.71%`, worst `-157.12%`, max new boundary `8.532%`,
  max preclip OOG `9.447%`; target LUT varies with source up to `.11725` RMSE.
- Innovation: derive one fixed target LUT from the reference's own canonical
  pivot, then apply it after per-source canonicalization. It improves 0/30,
  median `-57.52%`, worst `-118.81%`, max new boundary `1.083%`; closed.
- Repeat: two CanonCGT reports have ID `ec825926...e3efe` and byte-exact SHA
  `9b954501...e8a3f`.
- SA-LUT: bundled checkpoint is only the image-to-image pseudo-VLog teacher;
  the actual 4D-LUT inference network state is absent. Spatial context and
  non-commercial rights independently block global product promotion.
- NLUT: official 236,254,785-byte weight loads. Shared no-tune 5/30, median
  `-59.50%`; representative official-style 40-step batch tuning worsens to
  median `-100.13%` and 51.07% new boundary, so the tuning grid stops.
- Verification: 3 dedicated and 20 focused tests pass; evaluator repeats
  exactly. No external RGB output is retained.
- Commit: `208dfa8` (`research: audit CanonCGT reference matcher`).
- Handoff: exact arbitrary-photo look recovery remains underidentified.
  Preserve the completed guarded product shell and identity default. Keep
  aesthetic approximation and exact paired look-copy claims distinct; no
  capacity rescue or third-party promotion opens.

## 2026-07-27 - Close delivery and latest-main integration evidence

- Node/parent goal: P24 / complete reference-colour-match module delivery.
- Concurrent snapshot: main clean except `.codex/` at `350b687`; D-PCT clean
  at `f8ab191` after its 67/67 DNG/LibRaw agreement and 273-test boundary
  closure; this branch remains the only writer to its worktree.
- Full branch collection: 1081 pass, one skip and the same 36 classified
  missing-ignored-output/CRLF-hash failures; no module test fails.
- CLI: two-source default fit and stored replay use recipe
  `12c084ad...e7cacd`; both output pairs are hash-identical. Explicit
  research positive control applies successfully under the unchanged guard.
- Merge evidence: base `c03c321`, ours 95 paths, main 81, intersection zero;
  synthetic merge tree `97263dc4...f61e`; temporary latest-main merge passes
  all 196 selected module/ingress tests.
- Cleanup: the exact temporary integration worktree was removed and Git
  worktree metadata pruned; no user or unrelated file was removed.
- Decision: delivery is ready without algorithm promotion. Default identity,
  SDR-only ingress, D-PCT media ownership and film-claim separation remain
  mandatory.
- Handoff: repository owner may review and merge the branch. No push or merge
  into main was authorized or performed.

## 2026-07-28 - Start equal-task D-PCT consumer integration

- Node/parent goal: P25 / versioned D-PCT producer-consumer boundary for the
  Neuro-Film uploaded-reference module.
- Trigger: the user directed both tasks to continue autonomously as equal
  peers with efficient cross-task communication and independent long-term
  Goals.
- Skills used: `dev-research-reliability` primary; `drpt-bi-governance`,
  `plan-tracker-discipline`, `project-agent-log-discipline` and
  `project-structure-steward` as read-only governance/review layers.
- Live state: this branch started clean at `82baef6`; D-PCT published clean
  stable HEAD `59b72e0` for its CPU-only PST50 cross-scene professional
  development evaluator. That commit changes no shared producer interface.
- Decision: D-PCT is the sole algorithm/media/native authority; Neuro-Film is
  the sole product/transaction/replay/safety/A1-A4-A5/FilmFX authority.
  Neither task is subordinate.
- Interface correction: preserve one shared `ReferenceIntent`, but allow
  source-bound `TransformBundle` objects for source-plus-reference fitting.
  Do not claim a shared cross-content operator until A1/A4/A5 pass.
- Files claimed: `src/color_match/**`, matching tests and
  `reference_core_*.schema.json`, plus the named product plan, coordination,
  evidence and agent-log documents. D-PCT source and main-task W1/W2 files are
  forbidden.
- Handoff state: P25A documents the Mode C claim. P25B may implement only an
  additive consumer acceptance contract; a fixed D-PCT producer schema remains
  a future compatibility dependency, not an inferred current API.

## 2026-07-28 - Implement strict external-core consumer contracts

- Node/parent goal: P25B / D-PCT producer-consumer boundary.
- Change: add immutable `MatchViewV1`, source-bound `TransformBundleV1`,
  `DiagnosticsV1` and `CapabilitiesV1` records, strict JSON parsers/serializers,
  canonical identities and cross-record binding validation.
- Schemas: add four Draft 2020-12
  `configs/schemas/reference_core_*.schema.json` contracts.
- Key decision: v1 cannot express a shared operator. A transform is bound to
  one source and reference; cross-content sharing remains a future
  A1/A4/A5-gated contract.
- Verification: 20 dedicated tests; 54 new plus adjacent
  contract/schema/canonical/replay tests; compileall passes.
- Failure-path evidence: unknown fields, hash drift, profile semantics,
  absolute-white omission, source/reference mismatch, producer/build mismatch,
  unadvertised algorithm/profile/capability and diagnostics mismatch all fail
  closed.
- Peer state: D-PCT clean `4b71a8a`; no shared schema change. PST50 sRGB
  confirms the frozen candidate wins only 23/50 against identity and therefore
  does not weaken Neuro-Film's identity default.
- Structure: one additive module in the existing `src/color_match` package,
  matching schemas/tests/docs; no parallel package, decoder or D-PCT parameter
  representation was created.
- Handoff: P25C may adapt current `WorkingImage` only into its exact
  Neuro-Film relative-display profiles. A producer must explicitly advertise
  those profiles before invocation.

## 2026-07-28 - Adapt WorkingImage into exact consumer views

- Node/parent goal: P25C / external-core product adapter.
- Change: add `core_adapter.py` with an isolated read-only dense float32 buffer,
  exact relative-display sRGB/Rec.2020 profiles, network-order pixel hash,
  path-independent provenance and pinned capability checks.
- Fail-closed scope: scene-linear, encoded/unknown transfer states, unsupported
  working space, alpha, unapplied orientation, unadvertised profile, algorithm,
  schema or feature all reject before an external call.
- Verification: 16 adapter tests; 36 combined core tests; 123 adjacent
  product/preprocess tests. Non-contiguous input becomes dense without sharing
  or mutating source storage.
- Peer state: D-PCT clean `59ccb2f`; its Goal now explicitly includes active
  peer communication. Strict SA-LUT and PST50 BT.709 work changes no shared
  schema and remains non-commercial/development-only.
- Structural impact: one adapter beside the consumer contracts; no preprocess
  type change, duplicate decoder, media bridge, algorithm parameters or main
  renderer modification.
- Handoff: P25D may add a frozen consumer conformance envelope and bind its
  acceptance to existing A1/A4/A5 state. It must not invent a D-PCT producer
  artifact.

## 2026-07-28 - Bind external core intake to A1/A4/A5

- Node/parent goal: P25D1 / external-core product acceptance.
- Change: add `CoreAcceptanceDecisionV1`, canonical decision identity, strict
  schema/parser and `adjudicate_core_acceptance`.
- Product invariant: an external core can produce only identity fallback or a
  candidate for the existing product guard. It cannot directly produce an
  applied delivery state.
- Gates: the decision binds exact `A1/A4/A5`; unpromoted output fails closed
  unless a recorded research override exists. Non-ok core output fails closed
  even under that override.
- Verification: 12 dedicated and 48 combined core tests cover promoted,
  rejected, override, unsupported/invalid/fallback and inconsistent states.
- Handoff: P25D2 may freeze consumer-owned synthetic wire vectors. Fixture
  producer identity must be visibly synthetic and must not be labeled D-PCT.

## 2026-07-28 - Freeze synthetic core consumer conformance

- Node/parent goal: P25D2 / external-core product consumer boundary.
- Change: add a strict consumer conformance loader/verifier, exact-bit
  two-profile fixture, bundle/result Draft 2020-12 schemas and public exports.
- Fixture boundary: the producer role and identity are explicitly synthetic;
  no D-PCT/Zhuise algorithm, schema, weight, pixel result or mutable checkout
  is represented.
- Profiles: relative display-linear sRGB and Rec.2020 use extended float32
  samples with negative and above-one values, network-order pixel hashes,
  exact descriptor identities and path-independent provenance.
- Failure evidence: fixture hash drift, unknown fields, wrong producer role,
  insufficient cases, descriptor mismatch and missing advertised profile all
  fail closed or produce an explicit failed-case result.
- Verification: 9 dedicated tests and 57 combined core tests pass.
- Concurrent state: D-PCT live repository is clean at `06d57d4`; its latest
  communicated SA-LUT result is promising on average but only 30/50 versus
  identity and remains rights-blocked. No shared producer schema changed.
- Main authority: the user identified task
  `019f4b76-e70a-75c0-b7ea-b473ab38c200`; live main is `60617f9`, active on
  U5.R2Z1, with only its own `.codex/` untracked. No main file is consumed.
- Propagation: upward identity fallback and A1/A4/A5 remain unchanged;
  downward adapter/conformance binding is now executable; sideways
  recipe/replay/report/FilmFX behavior is untouched. P25E owns broad
  regression and latest-main integration preflight.

## 2026-07-28 - Close P25 consumer integration evidence

- Node/parent goal: P25E / external-core product consumer boundary.
- Stable inputs: consumer `729810d`, main `60617f9`, common base `c03c321`.
- Overlap: 113 consumer paths versus 86 main paths, exact intersection zero.
  Merge-tree result is `24ff7653...e179c`.
- Verification: 282/282 adjacent reference-match/ingress tests pass on the
  branch; full collection is 1138 passed, one skipped and the same 36 known
  ignored-output/CRLF failures, with no new failure family.
- Synthetic integration: detached merge `d01e2e4...061cd` passes the same
  282/282 tests. Its exact temporary worktree was validated, removed and
  pruned without touching any unrelated path.
- Change propagation: parent consumer contract is integration-ready;
  adapter/conformance children remain exact and fail-closed; replay/report,
  transaction, A1/A4/A5 and FilmFX siblings are unchanged; main and D-PCT
  retain their respective integration ownership.
- Remaining risks: there is no D-PCT producer schema/package/ABI conformance,
  no trusted relative-display compatibility profile, no RAW/HDR/video bridge,
  no native/mobile parity and no algorithm promotion. Identity remains the
  product default.
- Handoff: publish this stable consumer evidence to both equal peer tasks.
  Future producer integration must pin both commits and a producer-owned
  conformance bundle; it must not import a mutable checkout.

## 2026-07-28 - Start exact external-output receipt

- Node/parent goal: P26 / external-core product consumer integrity.
- Trigger: post-P25 audit found that acceptance v1 binds transform and
  diagnostics but cannot identify the exact output bytes returned by a future
  producer.
- Decision: add a consumer-owned receipt over a copied output buffer, then add
  a new admission version that binds that receipt. Do not mutate or reinterpret
  the frozen v1 acceptance record.
- Contract ceiling: same source profile/shape, finite dense float32 RGB,
  candidate-only. A receipt is not producer `ApplyResultV1`, final delivery,
  media conversion or product safety approval.
- Coordination: D-PCT received intent before writes; no reply is required and
  both tasks continue independently.
- Handoff: P26A freezes semantics; P26B may implement only the additive
  receipt and exact binding tests.

## 2026-07-28 - Implement exact external-output receipt

- Node/parent goal: P26B / external-core product consumer integrity.
- Change: add `CoreApplyReceiptV1`, prepared read-only output storage, strict
  schema/JSON roundtrip, canonical diagnostics/output provenance and complete
  execution binding validation.
- Fail-closed behavior: reject non-ok diagnostics, non-finite pixels, wrong
  shape/profile, output mutation, source/diagnostics swap, unknown fields,
  non-finite nested JSON and any delivery state other than `candidate-only`.
- Verification: 11 dedicated tests and 68 combined external-core tests pass;
  compileall and diff check pass.
- Peer state: D-PCT clean `1d9aa72` announced a separate producer-contract
  leaf with its own names and initial Rec.2020 profile. It explicitly claims
  no Neuro compatibility and does not overlap this consumer receipt.
- Structure: one additive module/schema/test beside existing P25 contracts;
  no ABI, decoder, producer payload, transaction, FilmFX or renderer change.
- Handoff: commit P26B, then P26C must issue a new admission version that binds
  the exact receipt ID without changing frozen acceptance v1.

## 2026-07-28 - Bind exact pixels into candidate admission

- Node/parent goal: P26C / external-core product consumer integrity.
- Change: add `CoreCandidateAdmissionV2`, strict schema/roundtrip and binding
  to both `CoreApplyReceiptV1.receipt_id` and the frozen v1 acceptance
  `decision_id`.
- State ceiling: accepted output is only `pending-product-guard`; rejected
  output is `identity-fallback`. No `applied` state exists.
- Failure evidence: mutated pixels, swapped receipt, swapped acceptance,
  removed A1/A4/A5 gate, contradictory accepted state, unknown field and
  attempted applied state all fail closed.
- Verification: 11 dedicated tests and 79 combined external-core tests pass;
  compileall passes.
- Compatibility: this is downstream of any future producer adapter. It does
  not consume or constrain the D-PCT producer contract currently in progress.
- Handoff: commit P26C, then P26D refreshes both peer heads, runs broad
  regression/propagation and publishes the fixed consumer schema hashes.

## 2026-07-28 - Close exact-output consumer integrity

- Node/parent goal: P26D / external-core product consumer integrity.
- Verification: 304/304 adjacent tests; full suite 1160 passed, one skipped
  and the same 36 known ignored-output/CRLF failures. No colour-match failure.
- Main isolation: stable main remains `60617f9`; 119 consumer paths versus 86
  main paths from `c03c321`, with zero intersection.
- Producer progress: D-PCT `3c2e9fdf...` now supplies fixed schemas,
  exact-bit fixture `c9c8c0ff...e326` and independent C++17 8/8 identity
  conformance. It still makes no compatibility claim.
- Propagation: exact pixels are now bound through receipt and admission;
  A1/A4/A5 and delivered-pixel guard remain authoritative; transactions,
  replay and FilmFX are unchanged.
- Handoff: P26 is complete. A separately claimed P27 may audit the fixed
  producer contract and exact-bit fixture, but compatibility remains closed
  until every field/profile/hash mapping passes.

## 2026-07-28 - Pin producer v1 and close candidate bridge

- Node/parent goal: P27A / explicit D-PCT producer compatibility.
- Source: fixed producer `3c2e9fdf...`; no mutable import or source copy.
- Independent hashes: MatchView `ac422dd8...`, TransformBundle `e1cf6a7e...`,
  Diagnostics `f6c1dec0...`, ApplyResult `887e964d...`, exact fixture
  `c9c8c0ff...e326`, Python contract `98a32053...` and C++ reference
  `dab41ef6...`; all match the communicated evidence.
- Mapping: relative display-linear sRGB and exact f32be pixel hash are
  mappable; producer/consumer canonical record IDs remain separate.
- Decision: `contract-mapped-candidate-pixel-bridge-closed`. DiagnosticsV1
  lacks required factual metrics/backend build fields, and ApplyResultV1 does
  not bind source geometry. No consumer invocation, receipt or admission.
- Peer response: D-PCT accepted both issues and will add v2 schemas/fixture
  without changing v1. Its metric definitions remain producer-owned.
- Verification: four lock/schema tests pass; diff check passes.
- Handoff: commit P27A. P27B may act only on a fixed v2 snapshot.

## 2026-07-28 - Open corrected D-PCT v2 candidate bridge

- Node/parent goal: P27B-D / explicit D-PCT producer compatibility.
- Producer correction: independent audit rejected `281b13f` fixture
  `eec54e...` because it paired unclipped output with a false clipping
  fraction. D-PCT independently fixed it at `11c581e`; the retained fixture
  `60e7466d...` has output range `[0,1]` and factual OOG/clipping `3/12`.
- Fixed boundary: producer code `73fcf290...`, native reference
  `ae06f388...`, DiagnosticsV2 schema `6f12c68d...`, ApplyResultV2 schema
  `5ab3fa9c...`; v1 remains unchanged and closed.
- Implementation: add a no-import `dpct_adapter` that verifies f32be byte
  length, pixel hashes, MatchView IDs, bundle payload/ID, DiagnosticsV2 ID,
  ApplyResultV2 ID, geometry/profile and all cross-envelope bindings.
- Identity policy: retain producer source/reference/bundle/diagnostics/result
  aliases and regenerate Neuro-Film view/transform/capability/receipt IDs.
  Dynamic timing-bound producer IDs are not cache or recipe identities.
- Product ceiling: exact producer pixels receive only a `candidate-only`
  receipt. Unpromoted candidates remain identity fallback; promoted candidates
  stop at `pending-product-guard` with A1/A4/A5 still required.
- Verification: 36 focused adapter/core tests and 279 adjacent colour-match
  tests pass; compile and diff checks pass.
- Structure: additive adapter, v2 lock/schema, one fixed fixture and tests in
  established homes; no D-PCT source, decoder, ABI, RAW/HDR/video, transaction,
  FilmFX or main worktree file is copied or changed.
- Handoff: commit P27B-D, then P27E runs full/latest-main propagation and sends
  a fixed compatibility snapshot to both equal peer tasks.

## 2026-07-28 - Close P27 producer compatibility propagation

- Node/parent goal: P27E / explicit D-PCT producer compatibility.
- Producer pin advanced to `b1b68b6` solely to include exact failed
  DiagnosticsV2 fixture `9f7a3581...`; success schemas/fixture/IDs remain
  unchanged. Failure has no bundle, measurements or result and maps only to
  identity fallback.
- Fresh-checkout correction: first latest-main merge test exposed CRLF
  conversion of the producer fixture. A narrow `.gitattributes` rule now
  forces both exact producer fixtures to LF; the repeated detached merge
  reproduces both published artifact hashes.
- Local verification: 41 focused tests; full suite 1179 passed, one skipped
  and the same 36 known ignored-output/CRLF asset failures. No reference-match
  or P27 failure.
- Main preflight: stable main `a33526e`; common base `c03c321`; 129 consumer
  versus 86 main changed paths, zero intersection; merge tree
  `63863b7b...`; detached synthetic merge passes 284/284 selected tests.
- Producer evidence: current peer adds independent MSVC/LLVM-MinGW exact
  success/failure identity and Android compile evidence. These do not claim
  Android/Apple runtime or consumer invocation compatibility.
- Propagation: success remains receipt-bound candidate-only; failure cannot
  issue a receipt; A1/A4/A5, transaction and delivered-pixel guard remain
  unchanged. RAW/HDR/video and the proposed absolute BT.2020 rail remain
  separate producer work with no implicit consumer mapping.
- Structure: no main or producer file changed; temporary merge worktree was
  verified, merge-aborted and removed. Branch is ready for peer handoff.

## 2026-07-28 - Start P28 atomic producer batch intake

- Node/parent goal: P28 / one uploaded reference plus ordered N source product
  semantics.
- Ownership: consumer-only batch binding over already verified P27 outcomes.
  D-PCT remains sole owner of fit/apply/media/native behavior.
- Contract: contiguous source indices, one exact shared reference, per-source
  producer and consumer identity validation, no hidden global/shot state.
- Atomic rule: any producer failure stops before admission; otherwise every
  candidate requires its own exact receipt plus A1/A4/A5 admission. Any
  admission fallback makes the full batch identity fallback.
- State ceiling: all-success is only `pending-product-guard`; partial output
  and `applied` do not exist.
- Coordination: producer peer received the non-overlapping intent. Proposed
  HDR absolute rail remains outside scope.

## 2026-07-28 - Implement P28 atomic producer batch intake

- Node/parent goal: P28B-C / ordered one-reference/N-source product intake.
- Change: add `DpctBatchResolutionV1`, strict schema/JSON roundtrip and
  source-by-source binding across consumer view, producer view, producer
  result/failure, transform, receipt, acceptance and admission identities.
- Producer failure: canonical failed diagnostics is reconstructed from the
  typed record before use. It has no transform/receipt and short-circuits all
  candidate admission in the same batch.
- Admission rule: all-candidate batches require exactly one receipt-bound
  acceptance/admission pair per source. Any individual fallback makes the
  atomic state identity fallback; all accepted sources stop at
  `pending-product-guard`.
- Ordering: source indices are contiguous user order. Reordering the complete
  source/outcome/adjudication tuples changes the batch identity while
  preserving exact per-source association; swapping outcomes alone fails.
- Failure evidence: empty/mismatched batches, source reassignment, partial
  admission after producer failure, applied state, removed A1/A4/A5,
  contradictory atomic state, unknown JSON and identity mutation fail closed.
- Verification: 38 initial and 65 combined P25-P28 focused tests pass;
  compile and diff checks pass.
- Structure: one additive batch module/schema/test beside P27 contracts; no
  file output, producer invocation, media/native/HDR code or FilmFX change.
- Handoff: commit P28B-C, then P28D runs broad/latest-main propagation.

## 2026-07-28 - Close P28 atomic batch propagation

- Node/parent goal: P28D / one-reference/N-source product intake.
- Full verification: 1191 passed, one skipped and the same 36 known
  ignored-output/CRLF environment failures; no colour-match or batch failure.
- Latest-main preflight: main `a33526e`, common base `c03c321`; 132 consumer
  versus 86 main paths and zero intersection; merge tree `10e3b164...`.
- Detached synthetic merge passes 296/296 selected colour-match tests. The
  temporary merge was aborted and its verified worktree removed.
- Propagation: P27 single-source receipts remain valid. P28 adds only an
  immutable batch resolution; it does not deliver pixels, write files or
  mutate admissions. File transaction, replay/report and FilmFX siblings are
  unchanged.
- Producer coordination: D-PCT confirms P28 has no producer interface impact.
  Its later SDR numeric/backend-fingerprint hardening and absolute HDR rail
  remain producer-owned; HDR stays explicitly unmapped.
- Handoff: P28 is stable and ready for both equal peer tasks. Next product
  work may bind this resolution to a durable batch transaction envelope, but
  may not invent producer invocation/package compatibility.

## 2026-07-28 - Start P29 exact-receipt numeric guard

- Node/parent goal: P29 / post-admission delivered-pixel safety.
- Scope: exact P27 source/output buffers, producer factual DiagnosticsV2
  fractions, receipt/admission identity and P28 atomic resolution.
- Consumer metric: new boundary-pixel fraction relative to the exact source,
  with a frozen epsilon. Producer OOG/clipping/projection facts remain
  authoritative and are thresholded, never consumer-reconstructed.
- State ceiling: pass means only `eligible-for-transaction`; failure means
  identity fallback. No file write, final applied state or partial batch.
- Claim ceiling: numeric guard cannot establish absence of semantic/visual
  severe artifacts or aesthetic quality; those evidence gates remain
  independent.
- Coordination: both equal peer tasks received the non-overlapping intent.

## 2026-07-28 - Close P29 exact-receipt numeric guard

- Node/parent goal: P29A-D / post-admission delivered-pixel safety.
- Single-source change: add an exact receipt-bound decision over producer
  OOG/clipping/projection facts and consumer new-boundary fraction. Default
  limits are 0.25/0.05/0.25/0.05 with epsilon `1/65535`; equality passes.
- Batch change: consume only a P28 `pending-product-guard` resolution, bind
  every ordered source/receipt/admission and require one policy. Any numeric
  failure makes the complete batch identity fallback.
- State/claim ceiling: success is only `eligible-for-transaction`;
  `applied`, pixels, partial delivery and visual/aesthetic claims are absent.
- Adversarial evidence: threshold equality, pixel mutation, source/decision
  swap, policy/reason/action/ID mutation, mixed/missing decisions, upstream
  short-circuit and strict JSON/schema cases pass.
- Verification: 65 combined P27-P29 tests pass. Full suite is 1218 passed,
  one skipped and the same 36 known missing-output/advanced-main asset-hash
  failures; no colour-match/P29 failure.
- Main propagation: main remains `a33526e`, base `c03c321`, zero changed-path
  overlap, clean merge tree `39d578ec...`; detached synthetic merge
  `d8ae8caa...` passes 65/65 selected tests and was removed.
- Producer propagation: HEAD `6c7118c` closes Windows x64 arithmetic parity
  for a separate absolute-HDR Sparks payload, but remains explicitly unmapped
  to this relative-SDR consumer and changes no P29 schema or threshold.
- Structure: two additive modules, schemas and tests in existing contract
  homes plus one evidence record. P27/P28, transactions, media and FilmFX
  remain unchanged.
- Handoff: P29 is stable for equal-peer consumption. A later transaction
  node may consume only the whole-batch eligible state and must still supply
  independent visual/severe-artifact adjudication before delivery.

## 2026-07-28 - Start P30 product transaction authorization

- Node/parent goal: P30 / fail-closed boundary between numeric eligibility and
  durable product staging.
- Audit finding: P29 correctly proves only numeric eligibility. The earlier
  CoreAcceptance contract deliberately permits a research-baseline override,
  so a transaction boundary must not infer product promotion from admission
  or numeric success alone.
- Contract intent: rebind P28 rows, P29 batch identity and the original
  per-source CoreAcceptance decisions. Require `promotion_status=promoted`,
  `research_baseline_override=false`, accepted core status and exact ordered
  acceptance IDs for every source.
- State ceiling: success only `authorized-for-staging`; failure makes the
  whole batch identity fallback. No file write, commit, delivery or applied
  state exists.
- Coordination: both equal peer tasks received the consumer-only intent;
  D-PCT has no schema or implementation action.

## 2026-07-28 - Close P30 product transaction authorization

- Node/parent goal: P30A-D / fail-closed product staging authority.
- Implementation: add an immutable atomic authorization that cross-binds P28,
  P29 and original CoreAcceptance identities. All rows must be core-ok,
  genuinely promoted, non-research and numerically eligible.
- Discriminating control: a candidate admitted through
  `research_baseline_override=true` passes P28/P29 but is rejected by P30 with
  algorithm-not-promoted and research-baseline-override; the full batch falls
  back. The promoted/non-research control is authorized only for staging.
- State ceiling: no pixel, path, partial output, commit, delivery or applied
  state; claim is `staging-only-not-committed`.
- Verification: 11 dedicated and 76 combined P27-P30 tests pass. Full suite
  is 1229 passed, one skipped and the unchanged 36 environment failures.
- Main propagation: stable HEAD `a33526e`, base `c03c321`, zero path overlap,
  merge tree `5fbb6ca2...`; detached merge `b06748b5...` passes 76/76 and was
  removed. Main's concurrent untracked Z1 files were not touched.
- Structure: one additive module/schema/test and one evidence document in
  established homes; no producer, transaction writer, media or FilmFX change.
- Handoff: P30 is stable. A later staging/commit integration must consume the
  exact authorization ID and retain rollback-safe all-or-nothing semantics.

## 2026-07-28 - Implement P31 portable product-chain conformance

- Node/parent goal: P31A-C / cross-platform consumer contract evidence.
- Delivery prerequisite audit: no external candidate has real product
  promotion plus a frozen invocation package. Direct file-delivery wiring
  remains closed rather than using a synthetic promoted fixture.
- Vector: freeze promoted-product and research-override cases across P28,
  P29 and P30, with ten exact canonical payloads and identities.
- Independent implementation: standalone C++17 performs its own lowercase
  hex decoding, SHA-256 and final staging state rule; it contains no D-PCT
  algorithm, media code or Python call.
- Windows evidence: MSVC `/O2 /W4 /WX`, 10/10 identity matches, both state
  matches and noncanonical hex rejection.
- Android evidence: independently pinned NDK r27d/Clang 18 links the verifier
  for arm64-v8a and x86_64 with validated ELF machines. No device/JNI/Apple
  runtime claim.
- Verification so far: five Windows conformance tests, one Android build test
  and 82 combined P27-P31 tests pass; compile and diff checks pass.
- Structure: additive fixture/schema/native verifier/build scripts/tests in
  established homes. Exact artifacts use LF. No producer or main file changed.
- Handoff: P31D still owns final full/latest-main propagation and peer
  snapshot before closure.

## 2026-07-28 - Close P31 portable product-chain propagation

- Node/parent goal: P31D / cross-platform consumer contract evidence.
- Full verification: 1235 passed, one skipped and the unchanged 36
  ignored-output/advanced-main asset failures; no colour-match/P31 failure.
- Latest-main propagation: main `079c7a1`, base `c03c321`; 88 main versus 152
  consumer changed paths with zero intersection; merge tree `f3febcf9...`.
- Fresh-checkout synthetic merge `66d2d6e3...` passes 82/82, including MSVC
  exact execution and pinned Android arm64/x86_64 cross-linking. Its worktree
  was removed.
- Concurrent safety: main's modified/untracked source-research files and
  D-PCT's uncommitted RPSCT files were read-only and never included, moved or
  deleted.
- Claim boundary: Python/MSVC identity parity plus Android build evidence does
  not establish device/Apple execution, producer invocation, image quality or
  applied delivery.
- Handoff: P31 is stable for both equal peer tasks. Product commit remains
  closed until a genuinely promoted candidate and frozen invocation package
  exist.

## 2026-07-28 - Add P31E second-compiler execution

- Node/parent goal: P31E / reduce compiler-specific consumer-contract risk.
- Toolchain: independently hash-pin LLVM-MinGW 20260616, Clang 22.1.8,
  `x86_64-w64-windows-gnu`; the producer-provided local path is only a hint
  and is not accepted as evidence.
- Result: static C++17 build executes all ten identities and both staging
  states exactly. Seven portability tests and 83 combined P27-P31 tests pass.
- Full/latest-main propagation: 1236 passed, one skipped and the unchanged 36
  environment failures; latest main `a8e5372`, zero path overlap, merge tree
  `3117e364...`; detached merge `8c231adb...` passes 83/83 and was removed.
- Claim ceiling: second compiler, same Windows x64 host. Apple and Android
  device execution remain open; no producer, quality or delivery claim.

## 2026-07-28 - Audit long-term reference-match completion

- Node/parent goal: P32 / full Neuro-Film reference-match product capability.
- Skills: `dev-research-reliability` primary; `drpt-bi-governance`,
  `plan-tracker-discipline`, `project-agent-log-discipline` and
  `project-structure-steward` as governance reviewers.
- Live ownership snapshot: consumer `0d23e27` clean; main `e7da085` with its
  own untracked `.codex/` and `tmp/`; D-PCT `ed01054` with its own uncommitted
  RPSCT leaf. Only consumer documentation is changed.
- Decision: P1-P31 complete the fail-closed consumer shell, but do not complete
  a real D-PCT-backed product render. The missing critical inputs are a
  genuinely A1/A4/A5-promoted candidate and a fixed producer invocation
  package/ABI.
- Completion matrix: records one-reference/N-source, replay, transaction,
  safety, promotion, compatibility, FilmFX, SDR/HDR/RAW/video, portable
  runtime and main-integration states with an explicit next owner.
- Change propagation: no producer or main interface changes. Existing P27-P31
  hashes and verdicts remain unchanged. External output commit stays closed;
  synthetic promoted conformance cannot authorize product delivery.
- Structural result: one evidence document in the established DRPT location
  plus plan/log pointers; no duplicate ABI, schema, source or media layer.
- Evidence commit: `f1ea3fb` (`docs: audit reference match completion`);
  corrected document SHA-256 `c91a55de...dac0`.
- Handoff: P32 is complete and both peer tasks received the audit intent. The
  next consumer leaf must not precede a real producer invocation artifact
  unless it is purely fail-closed readiness evidence.

## 2026-07-28 - Freeze P33 external staging transaction intent

- Node/parent goal: P33A / consumer-owned durable staging after P30.
- DoR: P27 owns exact isolated output bytes, P28 owns ordered N-source
  binding, P29 owns numeric eligibility and P30 owns genuine non-research
  promotion authorization. No producer invocation is assumed.
- Contract: require one shared reference intent plus source-bound transforms;
  independently revalidate authorization, batch row, receipt, output view,
  profile and destination bindings before the first write.
- Atomicity: stage every SDR file and one strict report, then commit all or
  restore every previous destination. Any fallback or mismatch writes
  nothing.
- Claim ceiling: `committed-to-staging-not-delivered`; no `applied` state,
  UI exposure, FilmFX attribution or final delivery.
- Scope: additive consumer module/schema/tests/docs only. Producer ABI,
  algorithms, media, HDR/RAW/video and main files are forbidden.
- Coordination: both equal peer tasks received intent and may continue their
  independent leaves without waiting.

## 2026-07-28 - Implement and verify P33 external staging transaction

- Node/parent goal: P33B-D / consumer-owned durable staging after P30.
- Implementation: `0cb94c3` adds one module, strict schema, public exports and
  nine adversarial tests. It reuses the existing rollback-safe batch commit
  primitive rather than creating a parallel transaction mechanism.
- Binding: P28 batch ID, P30 authorization ID, ordered source/transform/apply
  receipt/output-view IDs and one shared reference intent are revalidated
  before staging. Only the pinned relative linear-sRGB profile is accepted.
- State ceiling: success is `committed-to-staging` with claim
  `staging-files-committed-not-delivered`; no applied/delivered/partial state
  exists.
- Fault evidence: research override, permutation, duplicate paths, split
  intent and canonical mutations fail closed. Injected report replacement
  failure restores two old outputs and the old report with no debris.
- Verification: 9 dedicated, 107 combined P27-P33 and 74 adjacent tests pass;
  compileall and diff check pass. Full suite is 1245 passed, one skipped and
  the unchanged 36 environment failures; no colour-match failure.
- Latest-main: `e7da085`, zero overlap across 160 consumer and 94 main paths,
  merge tree `4344cbf1...`; detached synthetic merge passes 107/107 and is
  removed.
- Concurrent safety: main's untracked Kodak AA1 and temporary files were
  untouched. D-PCT `1bcb1d1` is exploratory/non-promoted and therefore
  cannot satisfy P33's real authorization prerequisite.
- Handoff: evidence documentation is ready to commit. Real use remains closed
  on fixed producer invocation plus A1/A4/A5 promotion; final delivery is a
  later consumer decision.
- Evidence commit: `36a1a98` (`docs: record external staging evidence`).
  P33 is complete as transaction mechanics, not as real product delivery.

## 2026-07-28 - Freeze P34 restart-safe staging verification

- Node/parent goal: P34A / durable verification after P33.
- Gap: P33 returns a trustworthy in-memory result at commit time, but later
  delivery or composition must not rely on stale object state or path
  existence.
- Contract: require the expected P33 run ID and report file SHA-256; reread a
  bounded strict UTF-8 report, revalidate canonical run identity, then reread
  and hash every ordered staged output.
- State ceiling: `verified-staging-not-delivered`; no file write, final
  delivery, applied state or FilmFX permission.
- Scope: additive consumer binding/schema/tests only. Producer, main, media
  and existing composition modules remain unchanged.
- Coordination: equal producer and main tasks received intent without a wait
  dependency.

## 2026-07-28 - Implement and verify P34 restart binding

- Node/parent goal: P34B-D / durable verification after P33.
- Implementation: `f1e8d35` adds a bounded read-only verifier, strict schema,
  public exports and nine adversarial tests. P33 v1 paths are hardened to
  absolute transaction-bound paths.
- Binding: caller-held expected report SHA and run ID are mandatory. The
  verifier recomputes report bytes, strict P33 canonical identity, absolute
  path and every ordered output file hash before issuing one verification ID.
- State ceiling: `verified-staging` /
  `verified-staging-not-delivered`; no mutation, delivery, applied state or
  FilmFX authority.
- Adversarial result: report/output mutation, missing output, report
  relocation, wrong run ID, order/state/claim/identity drift all fail closed.
- Verification: 18 P33-P34 and 144 combined P27-P34/transaction/composition
  tests pass; full suite is 1254 passed, one skipped and the unchanged 36
  environment failures.
- Latest-main: `b71fb68`, zero overlap across 164 consumer and 100 main paths,
  merge tree `d9d6fe5b...`; fresh detached merge passes 144/144 and is
  removed.
- Concurrent safety: main's modified Kodak AA1 work and D-PCT `c688c32` were
  read-only; no peer interface changed.
- Handoff: evidence is ready to commit. Final delivery must bind the P34
  verification ID rather than trust files or the P33 in-memory result.
- Evidence commit: `61431b9` (`docs: record external staging verification`).
  P34 is complete as restart verification, not as delivery authorization.

## 2026-07-28 - Freeze P35 external reference/FilmFX composition

- Node/parent goal: P35A / external-reference product composition.
- Gap: P18 correctly models the local rejected baseline, whose only applied
  path is an explicit research override. It must not be relabeled for a future
  genuine non-research external candidate.
- Contract: bind exact P34 verification/run/reference-intent identities;
  assign colour ownership only to the verified external reference look; append
  optional validated grain/halation/dust after colour.
- Prohibitions: no film colour profile, stock identity, research override,
  rendering, file mutation, delivery or applied state.
- Claim ceiling: `composition-ready-not-rendered` with output label
  `reference-look` or `reference-look+film-effects`.
- Scope: additive consumer composition module/schema/tests plus a small
  extraction of the existing FilmEffectBinding validator/builder for reuse.
- Coordination: both equal peers received intent; no producer or main action
  is requested.

## 2026-07-28 - Implement and verify P35 external composition

- Node/parent goal: P35B-D / external-reference product composition.
- Implementation: `249e415` adds a strict external composition
  module/schema/tests and extracts the existing FilmEffectBinding
  builder/validator for shared use. P18 behavior remains unchanged.
- Binding: exact P34 verification, P33 run, P30 authorization, reference
  intent and source count. External reference colour is the sole colour
  owner; optional procedural FilmFX follows it.
- Prohibitions: film colour remains null, stock identity false and state
  `composition-ready-not-rendered`. No renderer, file or delivery mutation.
- Adversarial result: applied state, colour stacking, stock claim, reversed
  order, identity drift, invalid effects and unknown fields all fail closed.
- Verification: 40 composition/P34 adjacent and 147 combined P27-P35 tests
  pass; full suite is 1265 passed, one skipped and the unchanged 36
  environment failures.
- Latest-main: `cfd1271`, zero overlap across 168 consumer and 100 main paths,
  merge tree `ce79f519...`; fresh synthetic merge passes 147/147 and is
  removed.
- Concurrent safety: main's Kodak AA1 files and D-PCT `c688c32` were read-only.
- Handoff: evidence is ready to commit. A future effect renderer must reverify
  P34, consume exact P35 and produce its own atomic report; no real candidate
  is currently eligible.
- Evidence commit: `fcb6278` (`docs: record external FilmFX composition
  evidence`). P35 is complete as composition planning, not effect rendering.

## 2026-07-28 - Freeze P36 atomic FilmFX execution

- Node/parent goal: P36A / execute the procedural branch of P35.
- Reuse audit: `src/filmfx` already exposes deterministic grain, simple
  halation, dust and compositing over display-sRGB arrays; preprocess and
  output encoders provide the required SDR bridge. No duplicate effect engine
  is needed.
- Contract: rerun P34 from its caller-held expected identities, require exact
  P35 binding and FilmFX presence, bind explicit base seed/source order, then
  commit all N effect outputs and one report atomically.
- First profile: simple halation only. Physical halation is rejected because
  the current P35 binding lacks its resolved control set; hidden CLI defaults
  are forbidden.
- State/ceiling: `filmfx-rendered-to-staging` /
  `filmfx-staging-not-delivered`; P33 source files remain immutable and no
  final delivery/applied state exists.
- Scope: additive consumer executor/schema/tests. Existing FilmFX,
  preprocess, transaction and producer code remain unchanged.
- Coordination: peer intent sent; no wait or peer action is required.

## 2026-07-28 - Implement and verify P36 atomic FilmFX execution

- Node/parent goal: P36B-D / execute the procedural branch of P35.
- Implementation: `aafa097` adds a strict FilmFX staging transaction, schema,
  public exports and eleven adversarial tests. It reuses the existing
  deterministic grain, simple-halation, dust, compositor, SDR encoder and
  rollback-safe batch commit primitives.
- Binding: exact P35 plan plus P34 verification; P34 is rerun against the
  caller-held report hash/run ID before pixels are read. P33 source outputs
  and report are protected from overwrite.
- Determinism: the caller supplies one signed int32 base seed. Ordered source
  `i` uses grain seed `base + 1009*i` and dust seed `grain + 17`; repeated
  runs with the same pixels/effects/seeds produce exact output file hashes.
- Fail-closed boundary: empty effects, physical halation without resolved
  controls, staging tamper, P33 destination collision, identity/order/seed/
  claim mutations and injected report-commit failure all reject. The injected
  failure restores every prior destination and leaves no staging debris.
- State ceiling: `filmfx-rendered-to-staging` /
  `filmfx-staging-not-delivered`; no final delivery, applied state, stock
  identity or film-colour transform is created.
- Verification: 115 adjacent, 436 combined color-match/FilmFX and compilation
  checks pass. Full suite is 1276 passed, one skipped and the unchanged 36
  isolated-worktree output/hash failures; no color-match or FilmFX test fails.
- Latest-main: `5f99ae9`, 172 consumer versus 102 main changed paths, zero
  exact overlap, merge tree `5733ee0b...`; a fresh detached synthetic merge
  passes 102 P33-P36/FilmFX tests and is removed.
- Concurrent safety: main's dirty DORF research files were read-only. D-PCT
  `c085bb1` is clean and its AceTone negative resource smoke changes no
  producer schema, ABI, receipt or HDR rail.
- Handoff: evidence documentation is ready to commit. P36 completes
  procedural FilmFX staging mechanics, not real producer invocation or final
  product delivery.
- Evidence commit: `b398e53` (`docs: record external FilmFX execution
  evidence`). P36 is complete as staging execution, not final delivery.

## 2026-07-28 - Freeze P37 restart-safe FilmFX staging verification

- Node/parent goal: P37A / durable integrity after P36 FilmFX staging.
- Gap: P36 proves atomicity and returns a trustworthy in-memory result at
  commit time. A later delivery decision must not trust stale objects, paths
  or mutable files after a process restart.
- Contract: require caller-held P36 report SHA-256 and run ID; bounded strict
  UTF-8 reread and canonical validation; verify the report path; then rehash
  every ordered P33 input and P36 FilmFX output.
- State/ceiling: `verified-filmfx-staging` /
  `verified-filmfx-staging-not-delivered`; no file mutation, delivery or
  applied state.
- Scope: additive consumer verifier/schema/tests/docs only. Producer
  algorithms, ABI, HDR/media and main-project files are forbidden.
- Coordination: both equal peer tasks received intent and can continue
  independently without waiting.

## 2026-07-28 - Implement and verify P37 FilmFX staging verification

- Node/parent goal: P37B-D / durable FilmFX staging integrity.
- Implementation: `c1f4195` adds one read-only verifier, strict schema, public
  exports and eleven adversarial tests.
- Binding: caller-held P36 report SHA-256 and run ID are mandatory. The
  verifier checks bounded exact report bytes, strict canonical P36 identity,
  report path, each ordered P33 input hash and each P36 output hash. Its
  canonical ID also binds P35 plan, P34 verification, seeds, formats and bit
  depths.
- State/ceiling: `verified-filmfx-staging` /
  `verified-filmfx-staging-not-delivered`; no writer, final publish or applied
  state exists.
- Adversarial result: report mutation/relocation, wrong run ID, changed or
  missing input/output, order, state, seed, claim and verification-ID drift
  all fail closed.
- Verification: 11 dedicated and 447 combined color-match/FilmFX tests pass.
  Full suite is 1287 passed, one skipped and the unchanged 36 isolated
  output/hash failures; no color-match or FilmFX failure.
- Latest-main: `a6895ca`, 176 consumer versus 113 main changed paths, zero
  exact overlap, merge tree `9117db57...`; fresh detached synthetic merge
  passes 113 P33-P37/FilmFX tests and is removed.
- Concurrent safety: main's dirty DoRF work and D-PCT's uncommitted
  BMKL/ColorTransferLib research files were read-only. D-PCT producer schema,
  ABI, receipt and HDR rail remain unchanged.
- Handoff: P37 evidence is ready to commit. It closes restart integrity, not
  real producer invocation or final user-visible delivery.
- Evidence commit: `fd1aa1a` (`docs: record FilmFX staging verification`).
  P37 is complete as restart verification, not final delivery.

## 2026-07-28 - Freeze P38 local delivery authorization

- Node/parent goal: P38A / authorization boundary after P37.
- Gap: P37 proves files and lineage are still exact, but it does not authorize
  a product action. A future writer must not infer delivery permission from
  verified staging alone.
- Contract: live-rerun exact P37, then bind the exact P35 composition, P34
  verification and P30 product authorization. Require the P30 atomic state
  and every source action to remain product-authorized.
- Scope/state/ceiling: `local-user-export` /
  `authorized-for-local-delivery` /
  `authorized-local-delivery-not-committed`.
- Prohibitions: no file write/copy, no applied state, no public/cloud share
  and no synthetic-candidate product eligibility claim.
- Scope: additive consumer authorization/schema/tests/docs only. Producer,
  media/HDR, FilmFX arithmetic and main-project files are forbidden.
- Coordination: both equal peer tasks received intent and need not wait.

## 2026-07-28 - Implement and verify P40 local delivery verification

- Node/parent goal: P40B-D / durable verification after P39.
- Implementation: `f9ec8c8` adds one read-only verifier, strict schema,
  public exports and eleven adversarial tests.
- Binding: caller-held P39 report SHA-256 and delivery ID are mandatory. The
  verifier checks bounded exact report bytes, canonical delivery identity,
  report path, every P36 staging source hash and every P39 delivered hash.
- State/ceiling: `verified-local-delivery` /
  `verified-local-files-reference-look`; no mutation or applied state.
- Adversarial result: report mutation/relocation, wrong delivery ID, changed
  or missing staging/delivered files, order, state, byte identity, claim and
  verification-ID drift all fail closed.
- Verification: 11 dedicated and 479 combined color-match/FilmFX tests pass.
  Full suite is 1319 passed, one skipped and the unchanged 36 isolated
  output/hash failures; no color-match or FilmFX failure.
- Latest-main: `9fea35b`, 188 consumer versus 128 main changed paths, zero
  exact overlap, merge tree `8c6134f2...`; fresh detached synthetic merge
  passes 149 P30-P40/FilmFX tests and is removed.
- Concurrent safety: main's `.codex/tmp` and D-PCT's uncommitted Volga2K
  script were read-only. Producer interface state was unchanged.
- Handoff: P40 evidence is ready to commit. Consumer local-delivery mechanics
  and restart integrity are complete; real use remains externally gated.
- Evidence commit: `ff6d632` (`docs: record local delivery verification`).
  P40 is complete as restart integrity, not real candidate admission.

## 2026-07-28 - Freeze P41 main-integration handoff

- Node/parent goal: P41A-B / integrate the completed consumer mechanics
  without modifying the concurrent main worktree.
- Scope: docs and coordination only. Pin the P1-P40 payload head, current main
  and producer snapshots, merge-tree evidence, focused/full checks, ownership
  boundaries and external blockers.
- Prohibitions: no merge, cherry-pick, push or edit in either peer repository;
  no relabeling BMKL or any synthetic candidate as promoted.
- Current snapshots: consumer `6db15c4`, main `9fea35b`, D-PCT `fa592e2`.
  BMKL Volga2K confirms a task-profile-dependent mapping family only and
  changes no producer contract or receipt.
- Handoff: `docs/drpt/REFERENCE_COLOR_MATCH_MAIN_INTEGRATION_HANDOFF.md`
  freezes payload/base/main/producer identities, all P33-P40 schema hashes,
  the conflict-free merge tree, review/test procedure and remaining blockers.
  Main integration remains owned by the equal main task.
- Handoff commit: `db52d6e` (`docs: freeze reference match integration
  handoff`). The stable snapshot was sent to both equal peer tasks; no wait or
  merge was imposed.

## 2026-07-28 - Implement and verify P39 atomic local export

- Node/parent goal: P39B-D / local file transaction after P38.
- Implementation: `8d60fc8` adds a byte-exact local delivery transaction,
  strict schema, public exports and eleven adversarial tests.
- Authorization: the exact P38 object is structurally validated and rebuilt
  live from P37/P35/P34/P30 immediately before any destination is staged.
- Transaction: every P36 output is copied byte-for-byte to an fsynced
  temporary file. N files and one canonical report then use the existing
  rollback-safe batch commit. Existing destinations are restored on an
  injected report-commit failure.
- Protection: P33 inputs/report and P36 outputs/report cannot be destinations;
  target extension must match staged file format and bit-depth policy.
- State/ceiling: `committed-local-delivery` /
  `local-files-delivered-reference-look`; this is a local file result, never
  an app-level applied state, stock identity or public share.
- Verification: 11 dedicated and 468 combined color-match/FilmFX tests pass.
  Full suite is 1308 passed, one skipped and the unchanged 36 isolated
  output/hash failures; no color-match or FilmFX failure.
- Latest-main: `33493ac`, 184 consumer versus 124 main changed paths, zero
  exact overlap, merge tree `e23ecbfd...`; fresh detached synthetic merge
  passes 138 P30-P39/FilmFX tests and is removed.
- Concurrent safety: main's `.codex/tmp` and D-PCT's uncommitted Volga2K
  script were read-only. Producer interface state was not changed.
- Handoff: P39 evidence is ready to commit. Local export mechanics are
  complete, while real use remains closed on producer invocation,
  A1/A4/A5 and main-project integration.
- Evidence commit: `4e5788b` (`docs: record atomic local delivery`). P39 is
  complete as local transaction mechanics, not real candidate admission.

## 2026-07-28 - Freeze P40 restart-safe local delivery verification

- Node/parent goal: P40A / durable verification after P39.
- Contract: require caller-held P39 report SHA-256 and delivery ID; bounded
  strict UTF-8 reread and canonical validation; then rehash every recorded
  P36 staging source and P39 delivered file.
- State/ceiling: `verified-local-delivery` /
  `verified-local-files-reference-look`.
- Prohibitions: read-only; no applied state, public share, stock identity or
  producer promotion inference.
- Scope: additive consumer verifier/schema/tests/docs only. Producer,
  media/HDR and main-project files are forbidden.
- Coordination: both equal peer tasks received intent and need not wait.

## 2026-07-28 - Freeze P42 portable consumer canonical core

- Node/parent goal: P42A / cross-platform conformance boundary after P41.
- Gap: the existing C++ verifier executes exact consumer canonical vectors on
  Windows and cross-links Android executables, but its hashing primitive is
  coupled to the hosted C++ runner and has no honest Apple compile evidence.
- Contract: extract one freestanding, header-free C ABI that computes SHA-256
  over arbitrary canonical bytes. The existing runner must call that exact
  core, so the already frozen P28-P30 vectors remain the semantic authority.
- Platform evidence: MSVC and pinned LLVM-MinGW execute the vectors; pinned
  Android NDK links arm64-v8a and x86_64 executables; pinned LLVM/Clang emits
  macOS arm64 and iOS arm64 Mach-O relocatable objects from the same core.
- Claim boundary: Apple object compilation is not SDK linking, app
  invocation, simulator/device execution, Swift interoperability, image I/O,
  performance or product compatibility. Android remains compile/link only.
- Scope: consumer canonical hashing and conformance scripts/tests/docs only.
  Product states, P33-P40 schemas, D-PCT producer algorithms/contracts/media,
  FilmFX arithmetic and both peer worktrees are forbidden.
- Probe evidence: a minimal no-header C file compiled warning-free into both
  macOS arm64 and iOS arm64 Mach-O objects. A hosted C++ Apple attempt failed
  at incompatible LLVM-MinGW CRT headers, confirming the narrower boundary.
- Coordination: both equal peer tasks will receive the committed intent and
  need not wait. The producer's announced invocation package remains a later
  explicit adapter leaf and is not guessed or preimplemented here.

## 2026-07-28 - Implement and verify P42 portable canonical core

- Node/parent goal: P42B-D / portable consumer conformance after P41.
- Implementation: `618f74e` adds a header-free C ABI for SHA-256 and the P30
  staging predicate, then makes the existing hosted C++ runner call it.
- Exact execution: MSVC and pinned LLVM-MinGW reproduce all ten frozen
  canonical identities, both state outcomes and eight SHA padding/multi-block
  boundaries. Uppercase hex still fails closed.
- Cross-target: pinned NDK r27d links Android arm64-v8a and x86_64
  executables. Pinned Clang 22.1.8 emits macOS/iOS arm64 Mach-O relocatable
  objects from the same core.
- Apple boundary: object compile only; no SDK, platform link/load,
  simulator/device, Swift/Objective-C, app, signing, image-I/O or performance
  evidence. Android remains compile/link only.
- Evidence identity: core `62c35870...a9fe4`; Android report
  `ff43b488...2278a`; Apple report `13bd7a9e...b0090`.
- Verification: 16 dedicated and 433 combined color-match/FilmFX tests pass.
  Full suite is 1328 passed, one skipped and the unchanged 36 isolated
  output/asset failures; no color-match, FilmFX or P42 failure.
- Latest-main: `bebd34f`, common base `c03c321`, 194 consumer versus 135 main
  paths, zero exact overlap, merge tree `f356c146...93b`; fresh detached
  synthetic merge passes all 16 focused tests and is removed.
- Concurrent propagation: main and D-PCT worktrees were read-only. Producer
  `eb4b889` now contains its own invocation package; that is a separate P43
  compatibility audit and no interface mapping is inferred here.
- Handoff: P42 completes portable canonical-core compile evidence, not mobile
  runtime or real product admission.
- Completion propagation: the top-level completion audit now distinguishes
  producer artifact availability at `eb4b889` from consumer compatibility or
  admission. P43 becomes the next ready consumer leaf; no package semantics
  are inferred from commit names alone.

## 2026-07-28 - Freeze P43 fixed producer invocation audit

- Node/parent goal: P43A-D / real package boundary after P42.
- Producer authority: stable clean HEAD `e22725d`; source commit `01ef061`;
  installed fixture commit `eb4b889`; package lock SHA
  `300b95b0...0287`.
- Exact artifact: `zhuise-research==0.2.0`, wheel size 95,994 and SHA
  `fd995ad8...c292`; request/response schemas `bac28668...b37b8` and
  `204da4f6...5473e`; fixture `eab24eae...b295`.
- Audit finding: an older same-name wheel exists locally at 94,548 bytes and
  fails the new hash. Any consumer path must select by exact size/hash and
  reject filename/path/version-only trust.
- Contract: invoke exact wheel bytes, never mutable producer source. Use the
  pinned Python 3.12/NumPy 2.4.4/Pillow 12.1.1 runtime; stage canonical f32be
  request bytes; require a new output directory; independently rehash and
  adapt response/artifacts through the existing P27 v2 verifier.
- Claim boundary: candidate-only local research integration. Repository lacks
  a release grant; A1/A4/A5, product promotion, HDR, native ABI, mobile/Apple
  runtime, applied/delivered state and redistribution remain closed.
- Scope: additive consumer lock/schema/invocation module/tests/docs only. No
  producer/main worktree writes, no FilmFX/media changes and no package source
  copy.

## 2026-07-28 - Implement and verify P43 exact-wheel invocation

- Implementation: `3d80e0d` adds a strict compatibility lock/schema and local
  invocation adapter. It verifies wheel bytes and exact runtime before
  creating a temporary transaction, then independently verifies the producer
  response and routes candidate facts through P27.
- Audit negative: stale same-name 94,548-byte wheels fail before mutation.
  Exact authority is 95,994 bytes / `fd995ad8...c292`.
- Packaging correction: ZIP import fails because producer source
  fingerprinting opens module files. The adapter safely expands only the
  hash-verified wheel into a temporary content tree; it never imports mutable
  producer source or trusts a persistent installation.
- Result: one real small synthetic exact-wheel invocation returns a verified
  `zhuise.dpct-chroma.cpu-reference.v1` candidate and leaves scratch empty.
  Wrong runtime, source change and stale wheel fail closed.
- Verification: 24 dedicated/adjacent, 187 D-PCT/core, and 438 combined
  color-match/FilmFX tests pass. Full suite is 1333 passed, one skipped and
  the unchanged 36 isolated output/asset failures.
- Latest main `2afa6c0`: 199 consumer versus 141 main changed paths, zero
  overlap, merge tree `1ce7695f...e2e7`; fresh detached merge passes 38
  P42/P43 tests and is removed. Concurrent main dirty research was read-only.
- Producer advanced to `e73ad14` for ROGR development only; no invocation
  package/schema change is consumed.
- Claim ceiling remains
  `candidate-only-not-promoted-not-applied-not-delivered`; A1/A4/A5,
  redistribution, native/mobile/Apple runtime and main merge remain open.

## 2026-07-28 - Freeze P44 exact-invocation A1/A4/A5 evaluation

- Node/parent goal: P44A-D / product evidence immediately after P43.
- Frozen data: the existing six neutral RAW-derived SDR sources and six
  same-content deterministic Velvia targets (`01/02/03/05/07/09`) in main's
  ignored evidence tree. Targets are read only after candidate rendering.
- Producer semantics: every source/reference pair is independently fit, as
  declared by the producer. The report must not call this a shared
  reference-only operator; source binding is explicit.
- A1/A4 automated matrix: all 30 cross-content rows, unchanged
  `evaluate_known_operator_batch` metrics and promotion thresholds.
- A4 tail: for each of six references, fit/render the frozen photographic
  probe and evaluate unchanged neutral/tone/boundary/semantic-hue limits.
- A5: for each reference, independently fit the two frozen images containing
  identical chart pixels in dark/cool versus bright/warm surroundings, then
  apply the unchanged median/p95/max shared-colour limits.
- Decision: existing `adjudicate_promotion` only. Automated pass can at most
  become `eligible-for-visual-review`; any failed gate is rejection. No
  threshold tuning, target use at inference, research override or owner vote.
- Scope: additive evaluator/tests/docs/ignored report only. Exact P43 wheel is
  invoked; no producer/main/FilmFX/media schema changes.

## 2026-07-28 - Execute and reject P44 exact D-PCT candidate

- Implementation: `d71aa01` adds a resumable 48-invocation evaluator and
  focused aggregation/progress tests.
- A1/A4: 0/30 cross-content rows improve; median/worst improvement
  `-201.2941%/-569.3720%`; maximum new boundary `21.1987%`.
- Photographic tail: 0/6 pass; worst neutral chroma `50.1215`, boundary
  `28.6965%`, semantic hue rotation `89.0774` degrees.
- A5: 0/6 pass; worst median/p95/max shared-colour drift
  `72.6391/96.9062/119.6310` Delta E76.
- Decision: rejected for ten independent reason labels. Blind review, P30 and
  real P33-P40 remain closed; no threshold tuning or research override.
- Reproducibility: two complete runs have different timing-bound run IDs but
  identical metrics/bundles/transforms and stable evidence ID
  `90d0022c...2d2a`. The evaluator preserves factual timing identities.
- Verification: 21 focused, 441 combined and full 1336 pass/1 skip/36
  unchanged. Latest main `209da15`: 202 consumer versus 147 main paths, zero
  overlap, merge tree `31e9b927...0d35`; fresh merge 21 pass and removed.
- Propagation: P43 remains useful invocation plumbing. The current capability
  is not the product algorithm; future versioned candidates rerun P44.

## 2026-07-28 - Freeze P45 successor-candidate admission

- Node/parent goal: P45A-D / non-overlapping consumer work after P44.
- Trigger: the first real callable D-PCT capability is reproducibly rejected,
  while stronger producer research families are not callable contracts.
- Decision: define a consumer-owned intake declaration and two-stage
  preflight. Evaluation readiness requires a genuinely new exact producer,
  package, wheel and capability identity, explicit colour profile and
  source/reference/batch semantics, and producer conformance. Product
  readiness additionally requires unchanged P44 A1/A4/A5 evidence, rights and
  target runtime evidence.
- Boundary: this leaf does not publish an algorithm, generalize the existing
  hard-coded invocation adapter, create an HDR bridge, or admit BMKL/ROGR by
  name. It only makes substitution and promotion fail closed.
- Files allowed: additive consumer config/schema/module/tests plus planning,
  audit and handoff docs. Producer and main repositories remain read-only.
- DoD: mutation tests reject identity reuse, semantic ambiguity, implicit
  rail conversion and missing product evidence; focused/full/latest-main
  propagation is recorded and both equal peers receive the stable snapshot.

## 2026-07-28 - Implement and verify P45 successor admission

- Implementation: `7b0ec11` adds one strict canonical declaration, policy,
  schema, public validator and eleven focused tests.
- Evaluation readiness: new exact producer/package/wheel/capability,
  conformance, existing mapped rail, consistent fit/batch semantics,
  deterministic stateless execution and evaluation rights are mandatory.
- Product readiness: unchanged P44 A1/A4/A5 plus blind review, commercial and
  redistribution rights, and Windows/macOS/iOS/Android runtime evidence are
  independently mandatory.
- Negative binding: P44 capability, wheel, full producer commit and stable
  evidence identity are frozen. Renaming research output cannot pass intake.
- Verification: 19 focused and 541 combined tests pass; full suite is 1347
  pass/1 skip/36 unchanged isolated-output failures.
- Propagation: latest main `f309c97`; 207 consumer versus 153 main paths with
  zero overlap; merge tree `d69c3b3...62f`; fresh merge 15 pass/4 expected
  skip because its detached worktree has no ignored exact-wheel evidence.
- Producer: latest `ef9a4cd` closes ROGR-v0 negatively and publishes no
  successor capability. No automatic substitution or consumer action opens.
- Handoff: P45 prevents accidental admission while leaving P43/P44 reusable
  for a genuinely new versioned package.

## 2026-07-28 - Freeze P46 P44 failure-signature analysis

- Node/parent goal: P46A-D / product-side algorithm feedback after rejection.
- Input boundary: consume only the two existing ignored P44 progress files.
  Do not invoke the wheel, reread known targets, tune thresholds or implement
  a competing matching algorithm.
- Analysis: stratify known rows by source/reference, quantify candidate/source
  error ratios and clipping association, count transform diversity, and
  aggregate photographic/context failure dimensions.
- Purpose: distinguish clipping-driven failure from global overcorrection and
  source-context dependence so the producer can choose a discriminating next
  hypothesis rather than repeat a failed family.
- DoD: strict input validation, canonical deterministic output, synthetic
  mutation tests, exact repeat on both runs, propagation and equal-peer
  handoff.

## 2026-07-28 - Execute P46 and isolate failure mechanisms

- Implementation: `d55142e` adds a strict timing-independent aggregator and
  nine mutation/identity/signature tests.
- Replay: both complete P44 progress inputs produce byte-identical report SHA
  `3512d4f4...c9fc` and stable evidence `b3410168...8fe3`.
- Known rows: 30/30 candidate errors exceed source errors; median/min/max
  magnification `3.013/1.117/6.694`; 30 unique bundles/transforms.
- Gamut diagnosis: 28/30 regressions occur at clipping <=5%; median clipping
  `0.894%`; clipping/error-ratio Pearson `0.143`. Clipping alone is not the
  dominant mechanism.
- Context diagnosis: 6/6 bundle pairs change and 6/6 probes fail despite
  median invocation clipping `0.401%`. Joint source/reference fitting is an
  A5 hazard.
- Photographic diagnosis: neutral 6/6, boundary 4/6 and semantic hue 1/6
  fail, so shared semantics alone cannot replace bounded neutral/boundary
  controls.
- Verification: 12 focused, 550 combined and full 1356 pass/1 skip/36
  unchanged. Latest main `c20c14e`, zero overlap, merge tree
  `ff015251...00a`; fresh merge 24 pass/4 expected wheel-evidence skip.
- Producer propagation: RGIN-v0 at `b964209` declares the relevant
  reference-only/shared future semantics and uncertainty shrinkage, but has
  no pass/model/capability/package. It remains below P45 intake.

## 2026-07-28 - Freeze P47 shared-reference operator batch semantics

- Node/parent goal: P47A-D / consumer support for a genuinely reference-only
  successor.
- Contract: exactly one opaque operator is fitted from the reference, fixed
  model and options. Ordered N source applications each bind their own source,
  diagnostics/result and exact output pixels to that same operator.
- Rejection boundary: per-source refits, mixed operator/bundle identity,
  foreign reference/profile, duplicate or missing sources, reordered receipts,
  mutable/non-finite pixels and partial batches fail closed.
- Claim ceiling: candidate-only awaiting P44 and numeric/product guards. The
  contract cannot create applied, promoted or delivered state.
- Ownership: consumer defines recipe/batch/output binding only. Producer owns
  fit/apply wire names, bundle payload, algorithm, model and conformance.
- DoD: additive schema/module/tests, no change to P28 per-source semantics,
  adjacent/full/latest-main propagation and equal-peer handoff.

## 2026-07-28 - Implement and verify P47 shared-reference batches

- Implementation: `02ba183` adds consumer-owned operator, exact apply receipt
  and ordered batch identities plus one strict schema and 14 tests.
- Identity split: operator binds reference/model/options/capability and has no
  source; each apply binds one source/result/diagnostics/output to it.
- Exact pixels: output arrays are copied, float32, finite, hash-bound and
  read-only. Batch identity binds ordered complete receipts.
- Fail closure: per-source semantics, mixed bundle/operator, foreign
  reference/profile, reordered/duplicate/missing sources or results,
  partial batch, non-finite/shape-invalid pixels and JSON mutation reject.
- Claim ceiling: candidate-only awaiting A1/A4/A5 and numeric/product guards.
  P28 remains the distinct per-source-bundle path.
- Verification: 25 focused, 564 combined and full 1370 pass/1 skip/36
  unchanged. Latest main `cc453e7`, overlap zero, merge tree
  `8204e002...742`; fresh merge 38 pass/4 expected wheel-evidence skip.
- Producer: two-chain build/apply semantics are agreed, but RGIN still has no
  calibration result, model, fixture, capability or package. No compatibility
  mapping opens.

## 2026-07-28 - Freeze P48 shared-batch numeric guard

- Node/parent goal: P48A-D / delivered-pixel safety after P47.
- Reuse: exact P29 numeric thresholds and boundary definition; no relaxed
  policy for shared operators.
- Facts: each source receipt must bind producer diagnostics identity and
  finite OOG/clipping/projection/output-range facts. Consumer independently
  verifies output min/max and measures new-boundary fraction from exact pixels.
- Atomicity: one complete ordered decision per P47 source under one policy;
  any failed source makes the entire batch identity fallback.
- Boundary: numeric safety is not A1/A4/A5, aesthetic approval, product
  authorization, staging or delivery.
- DoD: additive module/schema/tests, adversarial binding mutations,
  adjacent/full/latest-main propagation and peer handoff.

## 2026-07-28 - Implement and verify P38 local delivery authorization

- Node/parent goal: P38B-D / authorization boundary after P37.
- Implementation: `18eb813` adds one no-write chain authorizer, strict schema,
  public exports and ten adversarial tests.
- Binding: P37 is rerun live, then its P36 run is cross-bound to the exact P35
  plan, P34 core verification and P30 product staging authorization. P30 must
  remain atomically `authorized-for-staging` with every source accepted.
- State/scope/ceiling: `authorized-for-local-delivery` /
  `local-user-export` / `authorized-local-delivery-not-committed`.
- Prohibitions: no destination path, copy, rename, applied state, public
  sharing or synthetic-candidate eligibility is created.
- Adversarial result: live output tamper, valid foreign P30/P34/P35 chain
  members, scope, state, output-label, claim and authorization-ID mutations
  all fail closed. A byte snapshot proves the successful authorizer writes no
  files.
- Verification: 10 dedicated and 457 combined color-match/FilmFX tests pass.
  Full suite is 1297 passed, one skipped and the unchanged 36 isolated
  output/hash failures; no color-match or FilmFX failure.
- Latest-main: `5819b48`, 180 consumer versus 119 main changed paths, zero
  exact overlap, merge tree `523b4e0d...`; fresh detached synthetic merge
  passes 127 P30-P38/FilmFX tests and is removed.
- Producer propagation: D-PCT `d3e41bc` cleanly freezes BMKL as its strongest
  local PST50 development candidate, but explicitly does not promote it and
  changes no producer schema, ABI, receipt or HDR rail. P38 takes no action.
- Handoff: P38 evidence is ready to commit. A later writer may consume this
  exact authorization, but real use remains closed on genuine producer
  invocation and A1/A4/A5.
- Evidence commit: `fcd3ace` (`docs: record local delivery authorization`).
  P38 is complete as authorization, not file delivery.

## 2026-07-28 - Freeze P39 atomic local export

- Node/parent goal: P39A / local file transaction after P38.
- Contract: reconstruct and compare the exact P38 authorization immediately
  before mutation, then copy every ordered P36 verified output byte-for-byte
  to caller-selected local destinations and atomically commit all files plus
  one canonical report.
- Protection: P33/P36 staging inputs and reports are immutable transaction
  sources and cannot be destinations. Existing destination files may be
  replaced only through the shared rollback-safe batch commit primitive.
- State/ceiling: `committed-local-delivery` /
  `local-files-delivered-reference-look`.
- Prohibitions: no app-level applied state, stock identity, public/cloud share
  or producer promotion inference.
- Scope: additive consumer transaction/schema/tests/docs only. Producer,
  media/HDR, FilmFX arithmetic and main-project files are forbidden.
- Coordination: both equal peer tasks received intent and need not wait.
