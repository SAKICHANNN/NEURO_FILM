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

## 2026-07-28 - Complete P87 deterministic sRGB output quantization ABI

- Added a freestanding C11 float32-linear to sRGB8/sRGB16 quantizer using
  255 + 65,535 frozen first-input thresholds instead of platform `powf`.
- Threshold identity is `fae645ef...674c`; the exact float32 tolerance
  endpoints match the existing Python staging contract.
- MSVC and LLVM-MinGW execute all boundaries, both adjacent float32 values and
  one million random inputs per depth with zero mismatch; invalid input is
  preflighted before output mutation.
- Android arm64/x86_64 libraries link and macOS/iOS ARM64 objects compile.
  These remain link/object evidence, not device runtime.
- Verification: 6 focused and 41 adjacent colour-output tests pass. The wider
  colour selection has 1102 passes/3 skips; its only failure is the pre-existing
  absent ignored FILM-R manifest in `test_rec2020_visual_ood`.
- Producer contracts, A1/A4/A5, media rails and delivery authority are
  unchanged.

## 2026-07-28 - Prove P88 24MP output-quantizer streaming

- Two independent 24MP/72M-scalar audits have stable evidence identity
  `sha256:dc40141f...318a`; MSVC/LLVM, 8/16-bit and two replays all match the
  Python staging oracle exactly.
- Native apply observations are 0.25-0.72 seconds for sRGB8 and 0.92-1.67
  seconds for sRGB16. Tracked live array payload is bounded at 6,291,474 and
  8,388,632 bytes respectively; timings are excluded from stable identity.
- The audit uses create-only reports and unloads DLLs after success or injected
  failure. No further quantizer optimization is justified by this result.

## 2026-07-28 - Publish and verify P89 integration v13

- V13 binds P1-P88 payload `61842db` to main `9301cba` with 361/261 changed
  paths, zero overlap and immutable v12 ancestry. Manifest SHA is
  `64ef7be0...147e`.
- Direct/schema rebuild and v11-v13 tests pass 54/54. A real detached merge
  produces tree `82874da...f66` and passes 1022 color-match tests with 22
  platform/data skips and zero failures; the temporary worktree was removed.
- Review state remains `review-ready-not-merged`; main owns any actual merge.

## 2026-07-28 - Freeze P90 Android quantizer Test Lab package

- Added a minimal arm64 instrumentation package around the exact P87 core.
  Its 4096-sample vector covers endpoint/sRGB-knee/random inputs and exact
  sRGB8/sRGB16 arrays, two inner and two outer replays, plus two atomic
  rejection paths.
- The initial manifest-only target package was rejected by Test Lab before
  device execution as `NO_CODE_APK`. The corrected package includes a minimal
  target DEX anchor. The final dual-ABI package identity is
  `f5d76926...cadd`, with app/test APK hashes `82b42b99...875e` /
  `a3ce1c7a...2851`; the anchor, lifecycle and ABI-specific JNI/core sonames
  are included in the package identity.
- Eleven package/runner tests pass. This is locally build-verified only until a
  physical Test Lab result is parsed; no device-runtime claim opens yet.

## 2026-07-28 - Freeze P91 owned Test Lab runner

- The runner is default dry-run, verifies exact APK bytes, billing and Cloud
  Testing reachability, and uses only configuration/prefix
  `nf-019f9f37-agent` / `nf-019f9f37-`.
- Execution is capped at one Pixel 8/API34 matrix, two minutes and USD 1 worst
  case; PENDING over 120 seconds is cancelled. Bucket/matrix identities enter
  the local ledger before creation and only the exact ledger bucket is cleaned.
- The first submission created a matrix that failed validation before device
  execution. The exact matrix and Tool Results history were recovered into the
  ownership ledger; the result bucket was deleted and prefix bucket count is
  zero. The runner now records identities even when gcloud exits nonzero after
  matrix validation, and the corrected package passes dry-run preflight.

## 2026-07-28 - Close P92 cloud execution before device runtime

- The corrected DEX-bearing package passed APK validation, but the matrix then
  stopped as `SERVICE_NOT_ACTIVATED` with zero test executions. No physical
  device runtime or pixel evidence exists; the sanitized blocked report SHA is
  `80859f86...6d70`.
- Both exact owned result buckets were deleted and the live owned-prefix bucket
  count is zero. The ledger retains both matrices and the reused Tool Results
  history; device charge is zero while transient storage remains subject to
  billing settlement.
- The runner now requires Storage, Cloud Testing and Tool Results to appear in
  the project's enabled-service list before any resource creation. Current
  dry-run fails closed without changing the two-resource ledger. Enabling a
  shared-project API is outside this thread's ownership boundary, so no third
  submission is made.

## 2026-07-28 - Prove P93-P95 Android 14 emulator runtime

- Installed a consumer-owned Android emulator/tool/system-image closure under
  ignored outputs after confirming local hypervisor support. It does not
  modify or depend at runtime on the D-PCT SDK tree.
- Emulator bring-up exposed two real invocation defects before evidence:
  the package lacked x86_64 native payloads/ABI-specific core sonames, and the
  custom instrumentation omitted `onCreate -> start`. Both were fixed and
  covered by the dual-build/package/parser suite.
- Two full cold `-wipe-data -no-snapshot` Android 14 x86_64 runs take 76.16 and
  73.39 seconds and reproduce stable identity
  `sha256:079d19c2...ff225` exactly; report SHAs differ only in observation
  time (`0e533253...c9e4f` / `3dee6441...45127`).
- Each run performs two host instrumentation invocations, two Java outer
  replays per invocation and two native replays per output depth over all 4096
  frozen inputs. sRGB8, sRGB16, threshold/vector identity and both atomic
  failures pass. Twenty-nine related tests pass.
- Claim ceiling is strictly Android-14-x86_64 emulator runtime for the consumer
  quantizer. It is not Pixel/arm64 physical-device, media, D-PCT algorithm,
  arbitrary-look quality or product-admission evidence.

## 2026-07-28 - Publish and verify P96 integration v14

- V14 binds P1-P95 payload `2c33809` to current main `8dcfdac` with 376/310
  changed paths, zero overlap and immutable v13 ancestry. Manifest/schema
  SHAs are `12a951f...c734` / `6d5adbaa...7cfb`.
- Direct/schema rebuild and v11-v14 tests pass 72/72. A real detached merge
  produces tree `8d869107...20d` and passes 1040 color-match tests with 22
  platform/data skips and zero failures; the temporary worktree was removed.
- Review state remains `review-ready-not-merged`; main owns any actual merge.

## 2026-07-28 - Prove P97-P99 Android SDR boundary runtime

- Extended the same dual-ABI package with the existing P78 ICC accessor and
  P82 EOTF libraries. JNI binds ABI-specific sonames and compares the copied
  588-byte profile against a package-generated exact header.
- Two cold/wipe Android 14 x86_64 runs reproduce stable identity
  `sha256:d47fb4e2...2da2` exactly; report SHAs are
  `d7805334...af92` / `e5115cfb...2344`, with 79.45/73.13 second observations.
- Every uint8 and uint16 code passes exact EOTF-to-OETF roundtrip, all 4096
  quantizer vectors and both output depths remain exact, ICC bytes/identity
  match, and EOTF/quantizer failure paths preserve caller output. Fifty-five
  related tests pass.
- Claim ceiling remains Android 14 x86_64 emulator runtime for the consumer
  SDR boundary only. Encoded media parsing, ICC application/conversion,
  physical arm64, Apple runtime, D-PCT matching and product admission remain
  open.

## 2026-07-28 - Publish and verify P100 integration v15

- The first detached merge correctly found that v13's prior-manifest test
  observed CRLF bytes under latest main. The immutable Git blob was unchanged;
  `.gitattributes` now explicitly fixes LF for v13-v15 manifest/schema/build
  inputs, preserving the byte-hash gate rather than weakening it.
- V15 binds payload `10b1e4c` to main `5d355bd`: 380/320 changed paths, zero
  overlap, manifest/schema SHAs `949ed4c6...c440` /
  `28ca8e52...4724`.
- The corrected real detached merge tree is `78d4b707...4548`; 1058
  color-match tests pass, 22 platform/data tests skip and zero fail. The
  temporary worktree was removed. Main remains the merge owner.

## 2026-07-28 - Implement P67 strict path-free encoded decoding

- Code commit: `a2bb95274c26c63a65ec204b5a743e4ea6387a5c`.
  P65 bytes are batch-preflighted before allocation, then decoded as exact
  RGB uint8/uint16 PNG/JPEG/TIFF samples. PNG CRC/chunk/IHDR/IEND, JPEG
  SOI/single EOI/RGB and TIFF one-page/sample/offset-end contracts reject
  malformed, animated, appended or format/depth-substituted inputs.
- No batch is returned if any preflight/decode fails. Arrays are C-contiguous,
  readonly and hash-bound. This proves structure/samples, not colour profile.
- Verification: 767 all-color/three skips; full 1662/four skips plus the
  unchanged 36 historical failures; latest-main `66f0748`, merge tree
  `3269a149...`, 215 related pass/one skip.

## 2026-07-28 - Publish P68 immutable integration manifest v4

- Evidence commit `f9f05b8`. V4 binds P1-P67 `a2bb952`, main `66f0748`,
  287 payload blobs, 220 main paths, zero overlap, 30 exports, 16 schemas and
  exact P66 v3 SHA `fc9373a0...c97eb`.
- V4 manifest SHA-256:
  `612c097715d36d2347d9b9bfd349c72d4ba8a640a59742baa87d734325daae14`.

## 2026-07-28 - Implement P69 decoded-sample MatchView bridge

- Code commit: `272db64b0cd4ff4e7221eb181e02a80c19e64c3f`.
  Exact P67 integer samples pass through a frozen float32 IEC 61966-2-1 EOTF
  into readonly display-relative linear-sRGB `PreparedMatchViewV1` buffers.
- Validation is reconstruction-based: it reruns EOTF from P67 arrays,
  recomputes f32be pixel hashes and rebuilds provenance from decode,
  consumption, run, source, format and depth facts. A simultaneously forged
  pixel buffer, descriptor and self-consistent record still rejects.
- Claim is process-local MatchViews only: no path, persistence, application or
  delivery authority. Embedded colour metadata is not yet attested, so P69
  remains assumption-bound to the P62 encoder contract.

## 2026-07-28 - Publish P70 immutable integration manifest v5

- Evidence commit: `c5487a6cd6782396b968edba1b11286a0f54d344`.
  V5 binds P1-P69 `272db64`, base `c03c321`, main `50b38dd`, 294 payload
  blobs, 223 main paths, zero overlap, 35 exports and 17 schemas.
- V5 manifest SHA-256:
  `782043cc29db7a2489188c980e658fb5cc9f214808ba85b87d146bc07f0c97eb`;
  it binds exact P68 v4 SHA `612c0977...daae14`.
- Verification: 812 all-color/three skips; full 1707/four skips plus the same
  36 historical failures; current evidence-head merge tree `cf93886e...`
  passes 260 related tests/one skip and the temporary worktree is removed.

## 2026-07-28 - Freeze P71 encoded colour-metadata attestation

- P67 format/depth/sample validation is not equivalent to colour-space proof.
  P71 will inspect only P65 bytes and require exact expected sRGB profile
  semantics: PNG iCCP, JPEG reconstructed ICC and TIFF tag 34675.
- Missing, duplicate, oversized, malformed/compressed/trailing profiles,
  conflicting cICP and format substitution fail the whole batch.
- P69 v1 remains immutable and assumption-bound. A new P72 bridge may claim
  metadata-attested sRGB only by consuming exact P71 evidence.

## 2026-07-28 - Implement P65 same-handle process-local byte capture

- Implementation commit:
  `e83c18aa5b9e86282d6c57a49c90a8a25d0bb639`.
  P63's internal session can now retain exact bytes from each already-open
  output handle; it performs the existing second read and final same-handle
  rehash before returning the all-or-nothing immutable tuple.
- Memory boundary: <=256 MiB per output and <=512 MiB aggregate, checked
  before any output read. No spill path, output path or caller callback is
  exposed. The persisted record fixes path, persistent-snapshot and delivery
  authority to false.
- Artifact SHA-256: implementation
  `5b4916de8585556e97c9dbdc0e12749747bf9ea76c87a34c71ba7c5a0ccf4186`;
  P63 refactor
  `d9ec6818daf9cbad506b6c828382785b23e9191269e5bf8f48d36b5bea9deb22`;
  schema
  `9298948724462d0903752eb3a76f888ddf686429e07b1de63aed92bc40c81bc6`;
  tests
  `07da585ac163e32bd822e78465cb3a5daaa14c810afffce4bd3cf5387a68951b`.
- Verification: 34 P63/P65 pass; 99 adjacent pass/one skip; 735 all-color
  pass/three skips; full suite 1630 pass/four skips plus exactly the same 36
  historical missing-output/hash failures.
- Latest-main `eef149c82b73ed57955d35315a7bfefcd43acf25` remains read-only.
  Consumer/main changed paths are 280/218 with zero overlap; merge tree
  `7739d1223bff0e2e03cefac9e632c0a90146c69e`; detached merge passes 183
  related tests with one skip and is removed.
- Limit: these are authenticated encoded bytes, not yet proven decodable
  pixels. P67 must validate format/depth/frame/geometry from `BytesIO` only.

## 2026-07-28 - Publish immutable P66 integration manifest v3

- Evidence commit: `00593c7`. V3 binds exact P1-P65 payload `e83c18a`,
  base `c03c321`, main `eef149c`, 280 payload blobs, 218 main paths, zero
  overlap, 25 exports and 15 contract schemas.
- P64 v2 remains byte-identical and is bound as the exact prior manifest with
  SHA-256 `ae67b3ef...d784dd`; P58 remains transitively preserved.
- Artifact SHA-256: builder
  `ebe8a1b9da3a91028ceaa7460d996cae7d451ce6791ff8bf0b76fc5f888a8d6a`;
  manifest
  `fc9373a0452df7f1bff309b1bf9a2be83fc50a1cfa025e20236942b1bd2cf07c`;
  schema
  `3352634fc6fd349a4e4b300f14a36595aa326ef623398a3d6bbdede125fb860f`.
- Direct/module rebuild and 61 v1/v2/v3/P65 focused tests pass. V3 is review
  evidence only; it does not merge, promote, decode or deliver anything.

## 2026-07-28 - Freeze P67 path-free encoded-image decode gate

- Input is only the P65 immutable `bytes` tuple plus its live-validated
  metadata. Filesystem paths and persisted P63/P65 records are forbidden
  inputs.
- The gate must prove actual PNG/JPEG/TIFF format, exact declared bit depth,
  one frame/page, bounded dimensions/pixels and an allowed RGB sample model.
  Alpha, palette, malformed/truncated/polyglot ambiguity, decompression bombs
  and unsupported samples fail the entire batch.
- Output remains process-local decoded evidence with no applied, persistent
  or delivery authority.

## 2026-07-28 - Implement and verify P63 handle-bound staging observation

- Node/parent goal: P63A-D / restart verification of exact P62 staging.
- Implementation commit:
  `46b6ea8a4dda8c9de8221e4c29f24b51495ca924`. The verifier caller-pins the
  P62 report SHA, run ID and runtime qualification ID, opens the report and
  every output once, binds each handle identity, double-reads the bounded
  bytes and performs a final same-handle full rehash.
- Resource boundary: report <=16 MiB, at most 64 outputs, each output <=1 GiB
  and aggregate initial size <=8 GiB. Duplicate handle identities, reparse or
  non-disk inputs, ADS/verbatim paths and format/suffix mismatches reject.
- Platform boundary: Windows opens with read access while denying write/delete
  sharing and binds `FileIdInfo`. POSIX uses `openat`, `O_NOFOLLOW`,
  `O_NONBLOCK` and device/inode identity, but remains sequential observation:
  a finite last read has an irreducible post-read mutation window.
- Claim ceiling:
  `runtime-qualified-shared-staging-handle-observation-only-no-path-consumption-or-delivery`.
  The persisted verification record never authorizes a later path reopen.
- Artifact SHA-256: helper
  `35e27444c2b47a47ef0ef90e3cc813b529634e2d02108eba792f839f62890884`;
  verifier
  `f482f33c3694831920dd6cce230d1989dade78c6dcae9eea09d58a9156341e67`;
  schema
  `e76f4b7a57a5098d5e151fc92fe6effe906ddbba0866e2ba0e4d3be65def305c`.
- Verification: 87 adjacent tests pass with three privilege/platform skips;
  707 all-color tests pass with three skips. Full suite is 1602 pass/four
  skips with the same 36 historical ignored-output/hash failures. Fresh
  latest-main detached merge passes 180 related tests with three skips,
  merge tree `0e0389059b9930d857c4e6769572d0c3e609b0e3`.

## 2026-07-28 - Publish immutable P64 integration manifest v2

- Node/parent goal: P64A-D / review evidence for P1-P63.
- Evidence commit:
  `d6e562a9b2399afc15299a26868f69fd83064c33`. The v2 manifest binds payload
  `46b6ea8`, common base `c03c321`, read-only main `841efaf`, 273 payload Git
  blobs, 213 main changed paths, zero overlap, 20 public exports and 14 exact
  contract schemas.
- P58 v1 remains immutable. V2 binds its exact blob
  `2622c67249977541daac4997c5044fc301461b67` and SHA-256
  `db06eb9b47becb0a2a0f00e3a8a9f8d045e5e88e59cc285cc6f8b638a49a6a5b`.
- Artifact SHA-256: builder
  `dcc68f559c62de24f378e612050505c420c4bfbeb7bbcfd2007739b5c9250c80`;
  manifest
  `ae67b3ef7ec6fcf169aa2a4a97485693c341caf5624a2a8de6c5e866a7d784dd`;
  schema
  `3642f900a8a83039ba4a3fb5512f4f4f929f17879095ed34de3de5beec3f1867`.
- Both direct and module builders reproduce the manifest. Thirty-three v1/v2
  tests pass; all-color is 726 pass/three skips. The manifest is review
  evidence only and performs no merge or product-state promotion.

## 2026-07-28 - Record D-PCT R0bz/R0ca/R0CB final local boundary

- Equal producer task remains read-only. R0bz `c9af66e` reproducibly emits
  macOS/iOS arm64 Mach-O objects only; no Apple SDK link, load, host or device
  execution exists. R0ca `5ae82b8` hardens float32 representability and
  fail-closed portable validation without changing producer wire/capability.
- Integrated producer head `34af2fa5a2d095dab87affa73f169fd5a051bcfa`
  freezes the final local execution boundary. Full producer suite is 519/519
  pass; boundary report SHA is `f5bb0598...56ddd`.
- Freeze remains intentionally `NOT_FREEZE_READY`. Exact failed gates are
  external quality, D1, D3, expert blind review, real RAW, real HDR, real
  video and license/provenance. Windows runtime, Android compile/link-only
  and Apple object-only facts stay separate; no P45 successor or P60-P62
  closure is inferred.

## 2026-07-28 - Freeze P65 same-handle read-and-consume transaction

- Node/parent goal: P65A-D / eliminate the later path-reopen gap after P63.
- Contract direction: verification and byte consumption must occur while the
  exact P63-opened report/output handles remain live. A stored P63 record,
  pathname, inode-like value or matching later hash is not durable authority.
- Required faults: mutation/truncation/replacement, hard-link aliasing,
  partial decode, consumer exception and multi-file partial progress. Any
  failure must close without delivery or persistent consumption authority.
- Scope: additive consumer code/schema/tests/docs only. No producer ABI,
  FilmFX arithmetic, main-project write or D-PCT write is allowed.

## 2026-07-28 - Freeze P49 shared-path staging authorization

- Node/parent goal: P49A-D / product boundary after P45/P47/P48.
- DoR: P45 successor declaration/admission, P47 one-reference shared operator
  batch and P48 exact numeric batch guard are stable and independently
  versioned.
- Contract: staging requires three independent locks: a product-ready P45
  decision; exact P47/P48 batch, operator and ordered-source binding; and a
  consumer promotion binding produced from a `PromotionDecision` for the same
  capability, model, options, frozen gate policy and stable evidence ID.
- Threat model: declaration booleans alone cannot authorize; rejected or
  visual-review-only promotion, identity substitution, foreign evidence,
  model/options drift, numeric fallback or incomplete/reordered rows must
  fall back atomically.
- Claim ceiling: success is only `authorized-for-staging` /
  `staging-only-not-committed`; no pixels are written and no applied,
  committed, FilmFX or delivery claim is created.
- Scope: additive consumer contract/schema/tests/docs only. Producer code,
  algorithms, media/HDR and Neuro-Film main files remain forbidden.

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

## 2026-07-28 - Implement and verify P48 shared numeric guard

- Implementation: `5c50874` adds producer-fact binding, exact per-source
  decisions, atomic aggregation, strict JSON schema and 13 tests.
- Policy: reuses P29 `0.25/0.05/0.25/0.05` thresholds and `1/65535`
  boundary epsilon without a shared-path exception.
- Independent facts: live output min/max and new-boundary are recomputed from
  exact P47 pixels; diagnostics/OOG/clipping/projection remain bound producer
  facts pending future conformance.
- Atomicity: one source failure makes the full batch identity fallback.
  Success reaches only `eligible-for-transaction` under
  `numeric-only-no-visual-claim`.
- Verification: 38 focused, 577 combined and full 1383 pass/1 skip/36
  unchanged. Latest main `2bc968d`, zero overlap, merge tree
  `4eb40c39...5fe`; fresh merge 51 pass/4 expected wheel-evidence skip.
- Boundary: no RGIN compatibility or product authorization opens.

## 2026-07-28 - Implement and verify P49 shared product authorization

- Implementation: `14fc7bf` adds a canonical promotion binding, no-write
  atomic staging authorization, strict schema, public exports and adversarial
  tests.
- Three locks: P45 must be product-ready; P47/P48 must bind one exact operator
  and complete ordered numeric-eligible batch; independent
  `PromotionDecision` must be promoted for the same gate/evidence,
  capability/producer/model/options scope.
- Offline integrity: canonical declaration JSON is embedded in the
  authorization identity and P45 admission is rerun during deserialization.
  Declaration pass booleans alone cannot authorize.
- Failure closure: foreign declaration, commit, capability, profile,
  compatibility, evidence, model/options, operator, batch, guard, source,
  receipt or numeric decision fails closed. Any numeric failure is atomic.
- Claim ceiling: `authorized-for-staging` /
  `staging-only-not-committed`; no write, applied, FilmFX or delivery state.
- Verification: 49 focused, 588 combined and full 1394 pass/1 skip/36
  unchanged. Latest main `1dce729`, zero overlap, merge tree
  `bf2eb959...e82`; fresh detached merge 49 pass and was removed.
- Producer: RGIN-v0 closes at `fd036aa` with 20/20 frozen projections failing
  calibration and no model/capability/wheel/fixture. P49 has no real candidate.

## 2026-07-28 - Freeze P50 shared authorized staging transaction

- Node/parent goal: P50A-D / durable staging after P49.
- DoR: P47 exact applies, P48 numeric guard and P49 no-write authorization
  are stable; the existing P33 rollback-safe writer is proven.
- Contract: revalidate and cross-bind P47/P48/P49 plus every live prepared
  output, preflight unique SDR destinations, encode all outputs, build a
  canonical report and commit outputs/report as one rollback-safe batch.
- Structure: extract P33's generic SDR destination/encoding helpers into one
  package-internal module used by both per-source and shared paths. P33 schemas,
  identities and behavior remain unchanged.
- Failure policy: fallback authorization, foreign guard/operator/apply,
  order/inventory/path collision, unsupported format/depth, encoding failure
  or commit failure must write nothing or restore every prior destination.
- State ceiling: `committed-to-shared-staging` /
  `shared-staging-files-committed-not-delivered`; no local delivery, app
  applied state or FilmFX composition.

## 2026-07-28 - Implement and verify P50 shared atomic staging

- Implementation: `97f4131` adds shared output/report staging, strict schema,
  common package-internal SDR staging I/O and transaction/fault tests.
- Binding: every live P47 apply is cross-bound to exact P48 decisions and P49
  authorization before path handling; the report records authorization,
  batch, guard, operator, reference, source, producer result, diagnostics and
  output identities.
- Transaction: all encoded outputs and canonical report commit through the
  established backup/rollback primitive. Injected report commit failure
  restores every prior destination byte and leaves no staging debris.
- Additional guard: sparse values outside the bounded sRGB encode tolerance
  fail instead of being silently clipped even when their fraction passes P48.
- Structure: P33, FilmFX staging and local delivery reuse `staging_io.py`;
  their wire schemas/identities are unchanged and their adjacent tests pass.
- State: `committed-to-shared-staging` /
  `shared-staging-files-committed-not-delivered`.
- Verification: 54 transaction, 600 combined and full 1406 pass/1 skip/36
  unchanged. Latest main `1dce729`, zero overlap, merge tree
  `a04585ae...24e`; fresh detached merge 54 pass and was removed.
- Producer: SPGIN-v0 is preregistered only, with no calibration/model/
  capability/package/fixture or product rights; no real P50 staging opens.

## 2026-07-28 - Freeze P51 restart-safe shared staging verification

- Node/parent goal: P51A-D / durable trust after P50.
- Contract: caller must retain and provide exact report SHA, P50 run ID, P49
  authorization ID, P48 guard ID and P47 operator ID. The verifier boundedly
  rereads UTF-8 report bytes, reruns strict P50 parsing, checks report path and
  rehashes every ordered output file.
- Output: canonical no-write verification binding report/run/auth/guard/
  operator/reference plus every apply receipt, producer result, path and file
  hash.
- Fail closure: report byte/path/ID substitution, output tamper/missing/
  unreadable file, order/path collision and verification mutation reject.
- State ceiling: `verified-shared-staging` /
  `verified-shared-staging-not-delivered`; no producer invocation, staging
  mutation, composition, applied state or delivery.

## 2026-07-28 - Implement and verify P51 shared staging verification

- Implementation: `9816c4b` adds bounded no-write P50 restart verification,
  strict schema, public exports and adversarial tests.
- Caller-held chain: exact report SHA, run ID, P49 authorization ID, P48 guard
  ID and P47 operator ID are all mandatory. The verifier reparses P50 and
  rehashes every ordered file.
- Immutability: two repeated verifications are exact and leave report/output
  bytes and mtimes unchanged.
- Failure closure: report tamper/relocation, chain-ID substitution, output
  tamper/missing, order/identity duplication and JSON/state/claim mutation
  reject.
- Structure: P34 and P51 share bounded verification I/O; P34 wire/identity is
  unchanged and its adjacent tests pass.
- State: `verified-shared-staging` /
  `verified-shared-staging-not-delivered`.
- Verification: 33 focused, 612 combined and full 1418 pass/1 skip/36
  unchanged. Latest main `1dce729`, zero overlap, merge tree
  `03ffb2de...944`; fresh detached merge 33 pass and was removed.
- Producer: SPGIN-v0 is still running calibration and remains below P45/P49.

## 2026-07-28 - Freeze P52 shared verified composition boundary

- Node/parent goal: P52A-D / product ownership split after P51.
- Contract: one canonical no-write plan binds exact P51 verification,
  P50/P49/P48/P47 identities and optional Neuro-Film procedural FilmFX.
- Colour ownership: the verified shared reference look remains the sole colour
  owner. FilmFX may add only profile-bound grain, halation and dust after
  colour; it cannot add film colour, stock identity or calibrated claims.
- States: `composition-ready-not-rendered`; output label is
  `reference-look` or `reference-look+film-effects`; execution order is fixed.
- Failure closure: foreign verification/chain identity, FilmFX profile/hash/
  strength mutation, reordered execution, film-colour profile, stock claim,
  applied/delivered state or unknown fields reject.
- Scope: plan/schema/tests only; no staging files are read or written and no
  producer or FilmFX arithmetic changes.

## 2026-07-28 - Implement and verify P52 shared composition

- Implementation: `9f75297` adds one no-write shared reference composition,
  strict schema, public exports and claim/order/profile mutation tests.
- Ownership: `external-shared-reference-look` remains the only colour owner;
  optional FilmFX executes afterward and is profile/hash bound.
- Claim closure: film colour is null, stock identity false and calibrated
  reference false in every valid plan. FilmFX cannot upgrade those claims.
- States: `composition-ready-not-rendered`; `reference-look` or
  `reference-look+film-effects`; fixed execution order.
- Verification: 35 focused, 624 combined and full 1430 pass/1 skip/36
  unchanged. Latest main `1dce729`, zero overlap, merge tree
  `2c0438b6...2df`; fresh detached merge 35 pass and was removed.
- Producer: SPGIN-v0 remains below P45; P52 has no real shared run.

## 2026-07-28 - Freeze P53 shared procedural FilmFX staging

- Node/parent goal: P53A-D / deterministic execution after P52.
- Contract: cross-bind exact P52 plan and P51 verification, rerun P51 live,
  protect every P50 input/report path, render only active simple procedural
  FilmFX with deterministic per-source seeds, then atomically commit a
  separate output batch and canonical report.
- Structure: extract the already proven P36 procedural render primitive so
  per-source and shared paths use identical WorkingImage ingress, effect
  ordering, seed derivation and encoding.
- Failure closure: plan without effects, physical halation without controls,
  foreign plan/verification, live P50 tamper, protected-path overwrite,
  unsupported destination, render failure or commit failure must write nothing
  or restore all prior bytes.
- State ceiling: `shared-filmfx-rendered-to-staging` /
  `shared-filmfx-staging-not-delivered`; no local export or applied state.

## 2026-07-28 - Implement and verify P53 shared procedural FilmFX staging

- Implementation: `a2089cf` adds the P53 atomic shared FilmFX run, strict
  schema, public exports and a common procedural staging primitive reused by
  the already-proven P36 per-source path.
- Binding: exact P52 plan and P51 verification must agree on P50 run, P49
  authorization, P48 numeric guard, P47 operator and every ordered
  source-bound receipt/result. P51 is rerun immediately before rendering.
- Transaction: every P50 input and report is protected; all new outputs and
  the canonical report commit together. Injected report-commit failure
  restores every prior destination byte and leaves no staging debris.
- Determinism: the shared and per-source branches use the same WorkingImage
  ingress, effect order, signed seed derivation and SDR encoder. Equal inputs,
  effects and seed reproduce exact output hashes.
- Failure closure: inactive effects, unresolved physical halation, foreign
  plan/verification, live P50 tamper, protected overwrite, state/claim/order/
  seed mutation and unknown JSON fields reject.
- Identities: implementation SHA-256
  `7f02d5dab1fd485dcc6fe7e1969babd3112ea6cf820bf02dd61001f0030ee969`;
  common FilmFX staging helper
  `2f0e5b8e134a456deaa89aec005df28efcf5fa94cb2a01c5f28d0d75aff045f4`;
  schema `0d48d27b8cc120ea6703a5502a3e8ebf9f05bcddcc272d8dec1274c25a36720f`.
- Verification: 24 focused and 548 complete `test_color_match*` tests pass.
  Full suite is 1443 pass/1 skip/36 unchanged missing-output or tracked-asset
  hash failures, with no color-match/FilmFX failure.
- Latest main: `1dce729`, merge tree
  `43f36c7993e729b83e4647dc2dbfa6f069d228cc`; a fresh detached merge
  passes all 24 P36/P53 tests and was removed.
- Claim ceiling remains `shared-filmfx-staging-not-delivered`; no delivery,
  app-level applied state, stock identity or calibrated-reference claim.

## 2026-07-28 - Freeze P54 shared FilmFX restart verification

- Node/parent goal: P54A-D / restart-safe trust boundary after P53.
- Contract: the caller supplies the exact P53 report SHA-256 and run ID. The
  verifier performs a bounded UTF-8 read, validates the canonical P53 run,
  verifies its recorded report path, and rehashes every P50 input and every
  P53 output without writing.
- Identity: the resulting verification preserves P52/P51/P50/P49/P48/P47
  chain IDs and each ordered apply-receipt/producer-result identity, alongside
  file paths, hashes, encoding depth and deterministic FilmFX seeds.
- Failure closure: report-byte tamper, relocation, missing/tampered input or
  output, foreign expected run, order/seed/state/claim mutation and unknown
  JSON fields reject.
- State ceiling: `verified-shared-filmfx-staging` /
  `verified-shared-filmfx-staging-not-delivered`; no authorization, export,
  app-level applied state or product promotion.

## 2026-07-28 - Implement and verify P54 shared FilmFX restart verification

- Implementation: `e987c11` adds a canonical no-write shared FilmFX verifier,
  strict schema, public exports and adversarial restart tests.
- Caller authority: exact report SHA-256 and P53 run ID are mandatory. The
  verifier then independently parses the bounded canonical report, checks its
  original location, and rehashes every recorded P50 input and P53 output.
- Preserved lineage: P52 composition, P51 verification, P50 run, P49
  authorization, P48 numeric guard, P47 operator and each ordered
  apply-receipt/producer-result remain inside the verification identity.
- Read-only proof: two identical verifications reproduce the same identity
  while bytes and mtimes for every output/report remain unchanged.
- Failure closure: report tamper/relocation, input/output tamper or absence,
  foreign expected run, state/claim/order/seed/chain mutation and unknown
  fields reject.
- Identities: implementation SHA-256
  `c3e036040235a0f56fdb4ae44ca34faee0b90a504e16f58346edf8de2edd7c39`;
  schema `70aafd69a66eda74dab76361cd1a3f95022c7a835ef743ad4463e53ea79fad0f`.
- Verification: 36 focused, 560 complete `test_color_match*`, and full 1455
  pass/1 skip/36 unchanged failures. Latest main `1dce729`, merge tree
  `d657e1bc30b8c1a88e70eec1a9e6e3a468b5b470`; a fresh detached merge
  passes all 36 P53/P54/P37 verifier tests and was removed.
- Producer propagation: SPGIN-v0 closed negative at `a2e5ed9`; 0/12 frozen
  safety configurations pass, and no model/capability/package/shared fixture
  exists. P45/P49 remain closed and P54 consumes nothing from SPGIN.

## 2026-07-28 - Freeze P55 shared local-delivery authorization

- Node/parent goal: P55A-D / no-write product authority after exact P54.
- Contract: rerun P54 live, then bind its P53/P52/P51/P50 chain to the exact
  P49 authorization. P49 must remain product/evaluation ready with every
  ordered source action `authorized-for-staging`.
- Output: one canonical capability with scope `local-user-export`, state
  `authorized-for-shared-local-delivery`, and ceiling
  `authorized-shared-local-delivery-not-committed`.
- Prohibitions: no destination path, file copy, delivery, app-level applied
  state, public sharing, stock identity, calibrated-reference claim or
  producer promotion inference.
- Evidence label: synthetic promoted fixtures can prove only authorization
  mechanics. Real use remains closed until a genuine callable producer passes
  P45/P49 and the frozen product gates.

## 2026-07-28 - Implement and verify P55 shared delivery authorization

- Implementation: `8fcc59e` adds one canonical no-write shared-path local
  authorization, strict schema, public exports and adversarial chain tests.
- Live trust: P54 is rerun immediately. P52/P51/P50 IDs must match the exact
  P49 authorization, P48 guard, P47 operator/reference and source count.
- Product locks: P49 must remain evaluation-ready and product-ready, state
  `authorized-for-staging`, with every ordered source action authorized and
  numeric transaction decision accepted.
- Scope/ceiling: `local-user-export`,
  `authorized-for-shared-local-delivery`,
  `authorized-shared-local-delivery-not-committed`; no destination exists.
- Failure closure: live P53 output tamper, valid foreign composition/P51/P49
  member, state/scope/output-label/claim/identity mutation and unknown
  destination field reject. Successful authorization leaves all bytes and
  mtimes unchanged.
- Identities: implementation SHA-256
  `fed1d5fb4af0839602b39c3c924a0e872ae21f9baa34c4da442dcea40006a66d`;
  schema `bfbf365666d98bf642a5af6b4c1c517bfc5885f648e72f483063be75eb0c3a73`.
- Verification: 11 dedicated, 571 complete `test_color_match*`, and full
  1466 pass/1 skip/36 unchanged failures. Latest committed main `9d23a9b`,
  merge tree `a772eeec315e1801b30c8309fb49fedb2d69bfc2`; fresh detached merge
  passes all 36 P53-P55 tests and was removed.
- Reality gate: SPGIN and every current shared producer candidate remain
  rejected/non-callable, so this authorizer has only synthetic mechanics
  evidence and cannot authorize a real user render.

## 2026-07-28 - Freeze P56 shared atomic local export

- Node/parent goal: P56A-D / rollback-safe local file transaction after P55.
- Contract: reconstruct exact P55 immediately before mutation, then copy every
  ordered P54-verified P53 output byte-for-byte to caller-selected local
  destinations and atomically commit all files plus one canonical report.
- Protection: P50 bases/report and P53 outputs/report are immutable sources
  and cannot be destinations. Existing destination bytes may change only
  through the shared rollback-safe batch primitive.
- State/ceiling: `committed-shared-local-delivery` /
  `local-files-delivered-shared-reference-look`; neither means app-level
  applied, public share, film-stock identity or producer promotion.

## 2026-07-28 - Implement and verify P56 shared atomic local export

- Implementation: `d4a5d82` adds shared-path local delivery types, strict
  schema, public exports and a rollback-safe exact-copy transaction.
- Live authority: exact P55 is reconstructed before any destination staging.
  Every delivered row preserves its P47 apply receipt and producer result,
  and its bytes must equal the P54-verified P53 output.
- Protection/rollback: P50 bases/report and P53 outputs/report cannot be
  overwritten. Invalid count/extension fails before writes; injected report
  commit failure restores every previous destination byte and leaves no
  staging debris.
- State/ceiling: `committed-shared-local-delivery` /
  `local-files-delivered-shared-reference-look`; local files only, not app
  applied, public, stock-calibrated or algorithm-promoted.
- Identities: implementation SHA-256
  `1a4b4337ecd01c74a83e3e1d3e13d5963a4911febf01c24aeb923d4895979c0e`;
  schema `adccb9b822b51e8048f3e89e069ef84b1fbd2d89837abc1c8893112fd6541efc`.
- Verification: 8 dedicated, 579 complete `test_color_match*`, full 1474
  pass/1 skip/36 unchanged failures. Latest main `1f61119`, merge tree
  `c65ec723e69aecf40a4be6ec9e537d8775d9a553`; fresh detached merge
  passes all 19 P55/P56 tests and was removed.
- Reality gate remains closed: the transaction is proven only with synthetic
  promoted fixtures because no current producer satisfies P45/P49.

## 2026-07-28 - Freeze P57 shared local-delivery restart verification

- Node/parent goal: P57A-D / restart integrity after P56.
- Contract: caller supplies exact P56 report SHA-256 and delivery ID. A
  bounded read-only verifier parses the canonical report, checks its recorded
  location, and rehashes every P53 staging source and every delivered file.
- Lineage: preserve P55 authorization, P54 FilmFX verification and every
  ordered P47 apply-receipt/producer-result identity in the verification.
- State/ceiling: `verified-shared-local-delivery` /
  `verified-local-files-shared-reference-look`; this remains file integrity,
  not app-level applied state or real producer admission.

## 2026-07-28 - Implement and verify P57 shared delivery restart integrity

- Implementation: `62e2579` adds a canonical read-only P56 verifier, strict
  schema, public exports and restart/tamper tests.
- Caller binding: exact P56 report SHA-256 and delivery ID are mandatory. The
  verifier checks the report's recorded location and rehashes every P53
  staging source and every delivered local file.
- Preserved lineage: P55 authorization, P54 FilmFX verification and each
  ordered P47 apply-receipt/producer-result are part of the verification
  identity. Repeated verification preserves bytes, mtimes and identity.
- Failure closure: report tamper/relocation, foreign delivery ID, staging or
  delivered file tamper/absence, order/hash/state/claim/chain mutation and
  unknown JSON fields reject.
- Identities: implementation SHA-256
  `8e37189b519a83c27f7940c85842920aa384983cfc877f6d195cb54d0275333e`;
  schema `0325a907cfc9fb8dc3562382283e55cf10b52b10a22d1b1ae3c77170ec091ac4`.
- Verification: 19 focused, 590 complete `test_color_match*`, full 1485
  pass/1 skip/36 unchanged failures. Latest main `1f61119`, merge tree
  `eb0cd14116dcb1f38e39a9ae0584dcd5e59ba8e5`; fresh detached merge
  passes all 19 P56/P57 tests and was removed.
- Producer propagation: CGIN-v0 closed negative at `ffdfd98`; 0/9 configs
  pass and no model/capability/package/fixture exists. ROGR/RGIN/SPGIN/CGIN
  now form a saturated same-60-image search family; any new attempt requires
  materially new rights-cleared paired evidence, neutral companion/baseline,
  human semantic constraints or separately governed foundation prior.

## 2026-07-28 - Freeze P58 main-integration evidence bundle

- Node/parent goal: P58A-D / consumer-to-main reviewability after P57.
- Contract: a deterministic, non-self-referential manifest pins the reviewed
  consumer payload commit, common base and read-only main commit; enumerates
  every changed path with Git mode/blob; records exact main-overlap paths; and
  inventories P47-P57 public exports and JSON schemas.
- Verification: regeneration from the same commits must be byte-identical.
  Wrong commit/blob/export/schema, unexpected overlap or a dirty payload commit
  fails closed. The manifest records commands but does not execute a main merge.
- Scope: consumer script/fixture/test/docs only. No main or producer write,
  package release, branch push, applied state or real-candidate admission.

## 2026-07-28 - Implement and verify P58 deterministic integration manifest

- Node/parent goal: P58B-D / deterministic consumer-to-main review evidence.
- Implementation: `7e490bd` adds
  `scripts/build_reference_match_integration_manifest.py`, a committed
  canonical manifest and eight adversarial/replay tests.
- Frozen inputs: payload `1aee24f1d0a76da91079f6b88c58036dbcf6c57e`,
  common base `c03c321b9fc642e2e092d59e20dd1b145b96192d` and
  main `1f61119087cdb72d939b8db0c7b915e4adb7c5ce`.
- Inventory: 253 payload paths with exact Git modes/blobs, 178 main paths,
  zero overlap, 11 P47-P57 public exports and nine shared-path schemas.
  The manifest excludes its own builder/fixture commit by design, avoiding a
  self-referential identity.
- Exact artifacts: builder SHA-256
  `17b85d99bde95befde4ac2ba7664774340d7c8bcfa6f768d38ef40f52272b742`;
  manifest SHA-256
  `db06eb9b47becb0a2a0f00e3a8a9f8d045e5e88e59cc285cc6f8b638a49a6a5b`.
- Verification: manifest CLI exact replay, eight dedicated tests, 598
  complete `test_color_match*` tests and full suite 1493 pass/one skip/36
  unchanged ignored-output or historical-hash failures.
- Main preflight: read-only merge tree
  `ac6751f8992d329e257911b8cc5f86bc0b51c6fc`; a fresh detached merge
  passes all 19 manifest/P57 tests and was removed. The owner's dirty main
  worktree was not touched.
- Ceiling: `review-ready-not-merged`; only the main owner decides integration.

## 2026-07-28 - Freeze P59 strict integration-manifest wire contract

- Node/parent goal: P59A-D / schema-first main-owner review after P58.
- Finding: P58 exact Git regeneration is authoritative but the committed
  manifest has no independent strict JSON Schema. A reviewer cannot reject
  malformed structure before invoking Git-backed reconstruction.
- Contract: add one Draft 2020-12 schema with exact fields/types, SHA-1 Git
  object identities, modes, shared-schema SHA-256 identities, zero overlap,
  fixed verification commands and the immutable
  `review-ready-not-merged` ceiling. Validation runs before commit access.
- Scope: consumer schema/builder/tests/docs only. No main or producer write,
  merge, package release, candidate admission or product-state change.

## 2026-07-28 - Implement and verify P59 strict review schema

- Node/parent goal: P59B-D / schema-first main-owner review.
- Implementation: `0e3b376` adds a strict Draft 2020-12 manifest schema,
  schema-first validation in the P58 builder, six new adversarial tests and an
  explicit `jsonschema==4.26.0` dependency for Windows and Apple manifests.
- Schema SHA-256:
  `dcaa3caca5cee9082eb8ece9b5206ce2a337528f8ffb894019e76c2bb0f08a06`.
  Builder SHA-256:
  `fb9809015c0371f203d0339a6b955d0f6ff85b3f5ff56c7d5bd96898306dd154`.
- Fail-closed order: missing schema, unknown/missing field, invalid Git mode,
  traversal path, negative count, nonempty overlap or ceiling/export
  escalation rejects before any Git command. Shape-valid hash tampering still
  rejects against commit-derived reconstruction.
- Verification: 14 dedicated, 19 focused schema/manifest, 604 complete
  `test_color_match*`; full suite 1499 pass/one skip/36 unchanged ignored
  output or historical-hash failures.
- Latest-main propagation: main `349db2e2866985e2b818289dfcb2160e13e10ba4`;
  259 consumer versus 182 main paths with zero overlap; merge tree
  `d0c94491832b980005ce5e85dd606870c6e66432`; fresh detached merge passes
  25 P59/P57 tests and was removed. Main dirty files were not touched.
- Ceiling remains `review-ready-not-merged`; schema validity does not admit an
  algorithm, authorize a transaction or imply product application.

## 2026-07-28 - Freeze P60 successor runtime-evidence binding

- Node/parent goal: P60A-D / factual target-runtime closure after P45.
- Finding: P45 declares `windows_x64`, `macos_arm64`, `ios_arm64` and
  `android_arm64` as booleans. A true value is not source-bound evidence and
  cannot distinguish native execution from object compilation, link-only or
  cross-compilation.
- Contract: a consumer-owned bundle binds the exact P45 declaration,
  producer/capability/profile, target, proof class, OS/architecture,
  device/backend/driver, runner/executable/report hashes and repeated
  conformance/failure-injection facts. Windows/macOS require host runtime;
  iOS/Android require device runtime. Weaker evidence remains visible but
  cannot close product runtime readiness.
- Scope: consumer contracts/schema/tests/docs only. No CUDA, D3D11, JNI,
  Swift, RAW/HDR/video or other producer/native implementation; no product
  admission from runtime evidence alone.

## 2026-07-28 - Implement and verify P60 factual runtime binding

- Node/parent goal: P60B-D / target-runtime evidence after P45.
- Implementation: `eee47a2` adds one immutable runtime record/bundle/decision
  contract, strict schema, public API and 17 dedicated tests.
- Exact artifacts: implementation SHA-256
  `4c2402d5cbea4496751e81a7c171e1c8d632dcabb9e1eed3c8cb1e3c8a794b8a`;
  schema SHA-256
  `67f9ac7505f01905476b0b4c2c9b52abe67862c96cd2e72b916959fca2c83a8c`.
- Semantics: every record binds the exact successor declaration,
  producer/capability/profile, target, proof class, OS/architecture,
  versioned environment-matrix count/hash/summary, backend,
  runner/executable/report hashes and replay/conformance/failure-injection
  facts. A matrix can represent the producer's NVIDIA+AMD Windows evidence
  without pretending it is one device.
- Product proof classes: Windows/macOS require `host-runtime`; iOS/Android
  require `device-runtime`. Cross-compile, link-only and object-only records
  remain reportable but never satisfy the target.
- Fail-closed evidence: declaration/wire substitution, record/bundle identity
  tamper, duplicate/noncanonical target, missing target, weak proof, false
  declaration claim, replay count below two or any failed factual gate
  prevents `runtime_ready`.
- Verification: 47 focused, 621 complete `test_color_match*`; full suite 1516
  pass/one skip/36 unchanged ignored-output or historical-hash failures.
- Latest-main propagation: main `4b762be1ecd4f9591564392c78ce39c9a3c3932a`;
  262 consumer versus 187 main paths, zero overlap; merge tree
  `88198dd70551e7b4872c2b84d327d0376cf330ee`; fresh detached merge passes
  42 P60/P59/P57 tests and was removed.
- No real producer mapping is claimed: D-PCT has not published a new exact P45
  declaration for its ongoing native work. Runtime readiness is also not
  algorithm promotion or transaction authorization.

## 2026-07-28 - Freeze P61 runtime-qualified shared authorization

- Node/parent goal: P61A-D / close P49's unbound runtime-boolean gap.
- Finding: historical P49 correctly replays P45, P47 and P48, but P45 v1
  classifies runtime from four declaration booleans. Therefore a P49
  `authorized-for-staging` state is not factual four-target runtime evidence.
- Contract: preserve every P49/P60 v1 identity and add one qualification
  decision binding the exact P49 authorization, embedded declaration and
  exact P60 evidence. Only upstream authorization plus all four factual
  runtime targets yields `runtime-qualified-for-staging`; otherwise the
  decision is atomic identity fallback with canonical reasons.
- Scope: consumer contract/schema/tests/docs only. No staging write, producer
  call, native implementation, product application or retroactive P49 relabel.

## 2026-07-28 - Implement and verify P61 runtime-qualified authorization

- Node/parent goal: P61B-D / factual runtime guard between P49 and staging.
- Implementation: `77a8d84` adds one self-contained qualification containing
  canonical successor-declaration and P60 evidence JSON, strict schema, public
  API and 15 dedicated tests.
- Exact artifacts: implementation SHA-256
  `b9f13af5ffe4e173a7d6caae17ec8c619cca97fa0e21ec67db9ee624d0c49a67`;
  schema SHA-256
  `045e8dea47ece8526658bc67f9844aa9e6d4faafefce6f666dc77b92c1deebf0`.
- State rule: exact P49 `authorized-for-staging` plus P60
  `runtime_ready=true` is the only path to
  `runtime-qualified-for-staging`. P49 fallback, missing target, weak proof or
  any P60 reason produces atomic `identity-fallback`.
- Fail-closed evidence: foreign P49 ID/batch/operator/declaration, foreign P60
  declaration/evidence, embedded evidence tamper, readiness/upstream/state/
  ceiling/identity mutation and malformed arrays/JSON all reject.
- Verification: 62 focused, 636 complete `test_color_match*`; full suite 1531
  pass/one skip/36 unchanged ignored-output or historical-hash failures.
- Latest-main propagation: main `4fac70db92f4f7eed4c2569d9951ab4aa6d736b3`;
  265 consumer versus 192 main paths with zero overlap; merge tree
  `6b05bf245516c26e106088e15a5e378137b3dc56`; fresh detached merge passes
  57 P61/P60/P59/P57 tests and was removed.
- Change propagation: historical P49/P50 identities remain immutable. P50
  cannot be relabeled runtime-qualified; a separately versioned P62 durable
  staging consumer must require exact P61.

## 2026-07-28 - Audit corrected producer R0bw/R0bx evidence without qualifying P60

- Node/parent goal: P60D/P61D / read-only producer evidence propagation.
- Corrected producer snapshot: fixed D-PCT commit
  `3a4948a2ce280c064871f9302603a0efd204579b` supersedes the earlier
  `346b8cf` artifact identities. R0bw uses Vulkan 1.1-safe local size 128 and
  executes one reproducible SPIR-V on NVIDIA `10de:2f58` and AMD
  `1002:13c0` Windows devices, with two byte-exact 65-cube and UHD replays per
  device and unchanged cross-vendor UHD max/RMSE `3.814697e-6` /
  `4.45567e-7`.
- Pinned producer evidence: SPIR-V
  `e17ef3941ef4a9b20c97439710feef7624433b5b72adb47a14c55bbed5585d46`;
  reports `a0b99bf0bc7893d00ee7a61a26a449942ffa1a31eea23849b38287369f3ecd49`
  and `bb44d28c6bb2a461d9443c25290afa4799276a421bd418234d4d548fef90186d`;
  canonical non-timing identity
  `b08d75fe94b0bc3e95aecc3f14941f735a263e38b6cc54176bb519a268ff901b`;
  reproducible executable
  `6c2d8351692655a76f4f0265d817718ca682dda58d8123b3c03bb0cd598c1b06`.
- R0bx additionally produces reproducible Android API24 arm64-v8a and x86_64
  libraries, but its own report class is exactly
  `COMPILE_LINK_ONLY_NOT_RUNTIME`; neither library nor shader was loaded or
  executed on Android. It therefore cannot satisfy P60 device runtime.
- Decision: this is factual Windows host-runtime evidence, but there is no new
  exact P45 successor declaration/capability/package to bind it to. Android
  still lacks device runtime and Apple lacks host/device runtime. Therefore no
  real P60 bundle can become ready and P61 remains atomic identity fallback.
- Isolation: the producer repository and its evidence were read-only. No
  Vulkan, D3D11, CUDA, platform or algorithm source was copied into the
  consumer.

## 2026-07-28 - Freeze P62 runtime-qualified durable staging

- Node/parent goal: P62A-D / first durable consumer of exact P61.
- Gap: P50 correctly commits P49-authorized shared outputs, but predates P60
  factual runtime evidence and P61 qualification. Its report cannot be
  retroactively relabeled or treated as runtime-qualified.
- Contract: add a separately versioned atomic staging report that binds the
  exact P61 qualification, P60 runtime evidence, P45 declaration, P49
  authorization, P48 numeric guard, P47 batch/operator and each encoded output
  byte. Exact P61 state `runtime-qualified-for-staging` and factual
  `runtime_ready=true` are required before any directory, temporary file or
  destination is written.
- Publication: every destination must be absent and is created by an
  operating-system no-replace primitive. Outputs publish first and the report
  publishes last as the sole commit marker. A pre-report failure may leave
  immutable, report-less output orphans; they are not a committed run and
  must never be consumed. P62 never check-then-unlinks published names, so a
  non-cooperating replacement winner is not erased.
- Identity/versioning: preserve all P47-P51 and P60-P61 v1 identities. P62
  gets a new schema/state/claim ceiling and carries qualification,
  runtime-evidence and declaration IDs explicitly. A later P63 read-only
  verifier, rather than historical P51, must restart-verify this report.
- Scope: consumer transaction/schema/tests/docs only. No producer call,
  algorithm promotion, FilmFX, delivery, main merge, platform implementation
  or use of the Windows-only R0bw evidence as a four-target substitute.

## 2026-07-28 - Implement and verify P62 manifest-last runtime staging

- Node/parent goal: P62B-D / durable consumer after exact P61.
- Implementation: `e3372a9d00a87bd11a5669720c2d966a57a338eb` finalizes
  `neuro-film.runtime-qualified-shared-staging-run.v1`. The callable requires
  the caller's exact `expected_runtime_qualification_id`, snapshots pixels,
  rejects reparse/colliding destinations and creates outputs plus report
  without replacing any existing name.
- Exact artifacts: implementation SHA-256
  `40aece8b7a040f0593c0230f60ee3a35c192484ce066e4a40c19416d3fdb836a`;
  schema SHA-256
  `ff11bfd3c2c9117e4deb29a859f2ddbf6cb117532da82647c7141ec642534c51`;
  shared publication helper SHA-256
  `28622259384741dc179834f709ad0e92b74f46618d6bedfd02024e4225d6ea4f`.
- Final v1 boundary: prior-hash replacement fields were removed before stable
  evidence freeze. P62 is create-only and manifest-last. Failure before the
  report can leave uncommitted output or hard-link-stage orphans; automatic
  cleanup of published names is forbidden because conditional deletion cannot
  be made race-free against non-cooperating writers.
- Adversarial closure: initial/commit-boundary/final-operation destination
  creation, post-publication replacement, stage mutation, caller-buffer
  mutation, reparse paths, in-process/cross-process locks, invalid temp root,
  lock-close failure and POSIX hard-link stage-cleanup failure are covered.
  Historical P50 replacement remains separately regression-locked.
- Verification: 31 P62 tests pass with one real-symlink privilege skip; 160
  adjacent P47-P62/transaction tests pass with one skip; all color-match tests
  are 674 pass/one skip. Full suite is 1569 pass/two skips with the same 36
  missing ignored-output or historical tracked-hash failures.
- Latest-main propagation: read-only main
  `473b5773121bd4e059625a551432838b3fdbebb9`; 268 consumer versus 205 main
  changed paths, zero overlap; merge tree
  `d11d4fa736589e64e78dcb7bd587c67dd995e11a`; fresh detached merge passes
  92 P62/P61/P60/P59/P57 tests with one privilege skip and was removed.
- Limit: P62 is not process-crash/power-loss atomic, not a cryptographic
  attestation and not safe to consume from report presence alone. P63 must
  open each object once, bind file identity and hash those same handles before
  any replay, cleanup or delivery.

## 2026-07-28 - Record producer R0by CPU ABI without changing admission

- Producer fixed snapshot:
  `f1d3709bef1ae8e9b484080ba2c7c67483d4f0a2`. R0by supplies one C11
  `zhuise_dpct_cpu_apply_v1` Windows x64 host-runtime/failure-boundary ABI for
  the existing source-bound transform, with exact NumPy agreement and
  fail-closed output/diagnostics preservation.
- Evidence identities: report
  `7dac31905e8ee7f67bf53ced639080a89229205b222b6b29bc6b35f0dc394649`;
  canonical
  `651d9135b7511ddcedb293b2196d85561139fc434f9e8913d3f001763dedcc91`.
- Decision: this is no new P45 successor declaration, capability, package,
  quality promotion, receipt or schema. It neither closes P60/P61 nor changes
  P62. Planned Mach-O relocatable objects remain object-only until linked and
  actually executed on the declared Apple targets.

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

## 2026-07-28 - Complete P71 exact encoded colour-metadata attestation

- Node/parent goal: P71 / close the semantic gap between P67 integer samples
  and any metadata-proven display-linear bridge.
- Implementation commit: `01496ea7e6d0fae14843cf5f083e67b050bc2cdf`.
  The process-local contract re-inspects only P65-bound bytes and requires the
  exact frozen sRGB ICC: PNG iCCP, JPEG APP2 and TIFF tag 34675.
- Fail-closed scope: PNG CRC/order/keyword/decompression budget and conflicting
  chunks; every JPEG scan plus an APP allowlist; unique TIFF ICC/orientation
  and rejection of XMP/Photoshop/EXIF/transfer/primary/range alternatives.
  Duplicate JSON keys, non-standard constants and mutable receipt outputs also
  reject.
- Identities: implementation
  `11701d3a3a517b5003da3f4afece0237e3f749e8bb3924770d38889d2d151377`;
  schema
  `4db512eb2ced28d9cdf0b176393d917cff35c65396f07b45a7e06632b793a93a`;
  tests
  `c3e05d441f50729c5c55ef0afe0bbc315ee9ec35d7db0cb9da7d1a72bdb55060`.
- Evidence: 48 focused pass; 150 qualification-to-MatchView pass with one
  platform skip after hardening. Two independent read-only adversarial reviews
  ended with no blocker or major.
- Claim ceiling: embedded metadata is proven for this exact process-local
  bridge input. ICC conversion/application, persistence, application and
  delivery remain false.

## 2026-07-28 - Complete P72 attestation-consuming MatchView v2

- Node/parent goal: P72 / make metadata evidence a mandatory dependency
  without relabeling P69 v1.
- Implementation commit: `31266572fc27cf90480bdcf4e45f948f00558d6e`.
  The v2 bridge accepts only `RuntimeStagingColorAttestedDecodedBatchV1`,
  reruns P71 from original encoded bytes, reruns the frozen float32 IEC sRGB
  EOTF and binds the top-level/per-output attestation identities, exact ICC
  hash and profile binding into MatchView provenance.
- Hardening: nested decoded/sample/snapshot collections and P72 views must
  remain tuples; the runtime ICC hash, Python record and JSON Schema all pin
  `217fe48e...4356`; cross-batch, permutation, single-row substitution and
  self-consistent pixel/descriptor rewrites reject.
- Identities: implementation
  `9c7ce3dcfb9a327e46152c0fb81c9bd014213f708ca2bdb3cb21ba961f203033`;
  schema
  `ce4d6919e6b9e729a24fe2b1f37f7b1096391846d7879a80098d5aca4c9269f5`;
  tests
  `0b553d5e322a561bd4e2b2a6b67f45762d1954e074d3f00c712540c7610fa40d`.
- Evidence: 72 focused pass and 150 full staging-chain pass/one skip; final
  read-only contract audit found no blocker or major.
- Claim ceiling remains process-local MatchViews only. P69 v1 remains
  immutable assumption-bound evidence.

## 2026-07-28 - Publish P73-P75 checkout-safe integration evidence

- P73 commit `7eb4749` publishes v6 for P1-P72. It binds payload `3126657`,
  main `2f9a7c0`, 305 payload paths, 236 main paths, zero overlap, 45 exports
  and 19 schemas. V6 SHA is `01aa9137...412d024`.
- First detached merge correctly exposed one historical evidence failure:
  `reference_match_main_integration_manifest_v1.json` checked out as CRLF
  under the fresh worktree, changing SHA `db06eb9b...` to `0bb35a0a...`.
  The other 888 color-match tests passed with eight skips.
- P74 commit `a305497` adds complete v1-v7 LF checkout rules and an automated
  chain policy test. A fresh detached merge then passed 890 color-match tests
  with eight skips and zero failures.
- P75 evidence commit `d3b8f42` publishes v7, binding P1-P74 payload
  `a305497`, main `2f9a7c0`, 309 payload paths, 236 main paths, zero overlap,
  45 exports and 19 schemas. V7 SHA is
  `18a7a31dde03d448357d31bd94d6d204f5cc42c7f61e23690227dbdc5c72afc7`;
  it preserves exact v6 SHA `01aa9137...412d024`.
- Final detached merge tree is `226a2ce888c3757fd8af45f8fc447aba81663518`;
  all-color evidence is 908 pass/eight platform skips/zero failures. The
  temporary worktree was removed.
- Final local evidence is 913 all-color pass/three skips. Full suite is 1808
  pass/four skips plus the exact same 36 historical ignored-output/hash
  failures; no color-match failure.
- Main and producer repos stayed read-only. D-PCT remains at `34af2fa` and
  `NOT_FREEZE_READY`; no producer schema, admission or HDR mapping changed.

## 2026-07-28 - Pin P76 ICC bytes and publish P77 integration v8

- Node/parent goal: P76-P77 / remove runtime ICC-generator drift from the
  encoded-metadata-to-MatchView evidence chain and refresh main review facts.
- P76 commit `31f03fd88376f1e89301fb2677d6d23fa1e507d0` adds one
  canonical 588-byte sRGB ICC asset, strict schema, canonical Base64 fixture
  and public accessor. Profile SHA-256 remains
  `217fe48e...4356`; fixture SHA-256 is `82944c5d...aa9dd`.
- P71/P72 now validate the pinned asset directly. Tests prove profile/Base64
  drift fails closed and that, after real encoded bytes are created, the
  attestation/MatchView chain does not call the encoder's generator. Current
  host encoder equality is recorded only as local conformance, never proof of
  ICC application.
- P77 commit `eba586e` publishes v8 for payload `31f03fd`, main `93a7b66`
  and base `c03c321`: 318 payload paths, 238 main paths, zero overlap, 47
  public exports and 20 schemas. Manifest SHA-256 is
  `f40e32b9...4ee9`; v7 remains immutable.
- Verification: 69 focused and 919 pre-v8 all-color tests pass; final
  detached main/evidence merge passes 937 all-color tests with three skips,
  zero failures, merge tree `509df481...f3`. Local full suite is 1832 pass,
  four skips and the same 36 historical missing-output/tracked-hash failures;
  no color-match failure. Temporary merge worktree was removed.
- Claim boundary: exact profile bytes and consumer dependency are closed.
  Arbitrary ICC conversion/application, target-runtime parity, non-identity
  algorithm promotion, main merge, RAW/HDR/video and delivery remain open.

## 2026-07-28 - Complete P78 freestanding ICC profile ABI

- Node/parent goal: P78 / make the P76 profile identity consumable by native
  platform shells without adding media decoding or colour arithmetic.
- Implementation commit `983810a` adds one generated C11 accessor with exact
  three-symbol ABI, a C++ independent-hash verifier and pinned-toolchain build
  helpers. Null/short-capacity failure occurs before any caller-buffer write.
- Portability correction: the first Apple compile exposed a `string.h`
  dependency. The final generated C uses an explicit byte loop, then compiles
  freestanding for both Apple targets.
- Runtime evidence: reproducible MSVC executable
  `e7a33530...d8e4e` and LLVM-MinGW executable
  `5abb5e9d...9bd7a` both execute on Windows x86_64 and return the exact P76
  profile hash/header plus passing failure-boundary result.
- Non-runtime evidence: NDK r27d arm64/x86_64 shared libraries
  `a2239827...37169` / `8e4d4b3a...ac85c` link and export exactly the three
  ABI symbols; macOS/iOS Mach-O objects `3846352e...af6e4` /
  `f9e6b6d3...fc5c` define exactly those symbols. All builds repeat
  byte-identically.
- Verification: five focused and 27 adjacent ICC/product-chain portability
  tests pass. Android remains link-only; Apple remains object-only. No device
  runtime, ICC application, algorithm promotion, media rail or delivery claim
  opens.

## 2026-07-28 - Publish and verify P79 integration v9

- P79 commit `3595821` publishes manifest v9 for complete P1-P78 payload
  `d661fa9`, main `60b9bfa` and base `c03c321`: 329 payload paths, 243 main
  paths, zero overlap, 47 Python exports and 20 contract schemas. Manifest
  SHA-256 is `041e28a2...04f8f` and exact v8 identity is preserved.
- Main advanced during verification to `1ffbb5e`. A fresh comparison is 333
  consumer versus 244 main changed paths with zero overlap; evidence-head
  merge tree is `b0d2c19d...feb1`.
- Verification: local and detached-merge color suites both pass 960 with
  three platform skips. Full local suite is 1855 pass/four skips plus the
  exact same 36 historical missing-output/tracked-hash failures, with no
  color-match failure. The detached worktree was removed.
- P79 remains review-ready-not-merged. It does not convert Android link or
  Apple object evidence into runtime, does not promote an algorithm and does
  not change the producer interface boundary.

## 2026-07-28 - Execute P80 Windows dynamic ICC ABI

- Node/parent goal: P80 / close dynamic invocation after P78 static host
  execution without changing the three-symbol ABI or profile bytes.
- Initial dynamic link correctly failed: the optimizer folded the byte loop
  into an unresolved `memcpy`, and the LLVM driver first parsed the export
  definition as C input. The final source uses a volatile destination loop;
  LLVM compiles objects separately and links a deterministic minimal entry
  object plus explicit export definition.
- Commit `b550d8e` produces reproducible MSVC DLL
  `5213657c...c06b8` and LLVM-MinGW DLL `3a6d9267...03e2f`, each exporting
  exactly the three declared functions.
- Independent `ctypes` calls verify size/hash, null rejection,
  short-capacity unchanged memory and exact 588-byte copy/hash for both DLLs.
  Thirteen focused ICC tests pass.
- Claim ceiling: Windows x86_64 dynamic C ABI for the pinned profile only.
  No image I/O, ICC application, main integration, Android/Apple runtime,
  algorithm promotion or delivery opens.

## 2026-07-28 - Publish and verify P81 integration v10

- P81 commit `40b4768` publishes v10 for P1-P80 payload `c87d366`, main
  `21a877f` and base `c03c321`: 333 payload paths, 247 main paths, zero
  overlap, 47 Python exports and 20 schemas. Manifest SHA-256 is
  `6663dbca...7ea1`; exact v9 identity is preserved.
- Detached evidence-head/main merge tree is `6f8512fb...52fc`. Local and
  detached color suites both pass 980 with three platform skips and zero
  failures. Full suite is 1875 pass/four skips plus the unchanged 36
  historical missing-output/tracked-hash failures.
- The detached worktree was removed. P81 remains review-ready-not-merged and
  changes no D-PCT producer interface, algorithm admission, media rail or
  Android/Apple runtime claim.

## 2026-07-28 - Complete P82 exact portable sRGB EOTF ABI

- Node/parent goal: P82 / make the consumer-owned P72 decoded-sample EOTF
  portable without importing D-PCT media or algorithm code.
- Commit `8bb9e0b` generates an exact float32 lookup for every uint8 and
  uint16 code from the existing Python reference. Combined table identity is
  `1b8f915b...0a1ca`; source/header identities are
  `0835a1a6...3ba35` / `74884f0c...0dd5a`.
- MSVC DLL `f01d82bb...b086a` and LLVM-MinGW DLL
  `67d740d7...96d0e` are reproducible, export exactly two functions, load via
  FFI and match all 65,792 reference values bit-for-bit.
- Null input/output, invalid depth, zero count, short capacity, `size_t`
  overflow and overlapping ranges reject before output mutation.
- Android arm64/x86_64 libraries link reproducibly with exact exports; macOS
  and iOS arm64 objects compile reproducibly with exact definitions. These
  remain link-only/object-only, not runtime.
- Claim ceiling: decoded SDR sample-to-linear adapter only. No encoded media,
  ICC application, RAW/HDR/video, colour-match algorithm, product
  authorization or delivery claim opens.

## 2026-07-28 - Publish and verify P83 integration v11

- P83 commit `431729d` publishes v11 for P1-P82 payload `20fb34a`, main
  `a264a82` and base `c03c321`: 343 payload paths, 253 main paths, zero
  overlap, 47 Python exports and 20 schemas. Manifest SHA-256 is
  `b3bb7dee...ccfe`; exact v10 identity is preserved.
- Main advanced during verification to `4bbf362`. A fresh comparison is 347
  consumer versus 254 main changed paths with zero overlap; the conflict-free
  evidence-head merge tree is `f1df3ff9...b065`.
- Local and detached-merge color suites pass 1003 with three platform skips
  and zero failures. The local full suite is 1898 pass/four skips plus the
  same 36 historical missing-output/tracked-hash failures; no color-match test
  fails. The detached worktree was removed.
- P83 remains review-ready-not-merged. It neither changes the D-PCT producer
  interface nor converts Android link-only or Apple object-only evidence into
  runtime, and it opens no algorithm, media or delivery claim.

## 2026-07-28 - Harden P84 exact EOTF ABI preconditions

- Node/parent goal: P84 / close representation and typed-pointer assumptions
  in the P82 freestanding consumer EOTF ABI without changing its two symbols,
  table identity or decoded-sample scope.
- The generated C now requires IEEE binary32 compile-time parameters, verifies
  the little-endian bytes of `1.0f` before writing, stores exact result bits
  through character lvalues and rejects unaligned float output or uint16 input
  pointers before mutation.
- Both Windows DLLs still match all 65,792 Python reference values exactly.
  New unaligned input/output sentinel cases reject byte-unchanged; the prior
  null/short/overflow/overlap cases remain closed.
- Android arm64/x86_64 link-only and macOS/iOS arm64 object-only builds remain
  reproducible with exact two-symbol boundaries. The refreshed source SHA is
  `33a31fae...b579f`; platform identities are recorded in
  `REFERENCE_COLOR_MATCH_SRGB_EOTF_PORTABILITY.md`.
- Propagation: no wire schema, profile, D-PCT producer capability, algorithm,
  media, authorization or delivery state changes. Android/Apple runtime
  remains explicitly open.

## 2026-07-28 - Publish and verify P85 integration v12

- P85 commit `49cc6ae` publishes v12 for P1-P84 payload `091e688`, main
  `bc04943` and base `c03c321`: 347 payload paths, 258 main paths, zero
  overlap, 47 Python exports and 20 schemas.
- Manifest SHA-256 is `2e082062...1a66`; exact v11 SHA
  `b3bb7dee...ccfe` is embedded and prior manifests remain immutable.
- Direct/module reconstruction and v11/v12 focused tests pass 36/36.
  Conflict-free merge tree `e778299c...1ecc` was mounted in a detached
  worktree; all-color passes 1003 with three platform skips and zero failures,
  matching local P84 regression. The temporary worktree was removed.
- Last local full-suite evidence remains 1898 pass/four skips plus the same 36
  historical output/hash failures and no color-match failure.
- Claim propagation is unchanged: review-ready-not-merged, identity fallback,
  Android link-only, Apple object-only, no D-PCT schema/capability mutation and
  no media, algorithm-admission or delivery expansion.

## 2026-07-28 - Prove P86 large-image EOTF streaming

- Node/parent goal: P86 / add factual large-image consumer runtime evidence
  while main continues AM1 and D-PCT independently audits its next producer
  leaf.
- The fixed audit processes 24 MP / 72 million scalar samples at uint8 and
  uint16 through both MSVC and LLVM-MinGW DLLs, using 1,048,579-sample chunks,
  two replays and an independent P72 NumPy oracle for every scalar.
- Two complete audits have exact non-timing evidence identity
  `sha256:5f7a5150...a579a`. Output hashes are `994c30d3...aa12c`
  (uint8) and `ad7007c3...3f148` (uint16); both compilers and all four replays
  agree.
- Maximum simultaneously live tracked array payload is 9,437,211 bytes for
  uint8 and 10,485,790 bytes for uint16. This is explicitly not process RSS
  or target-device memory evidence.
- The first focused run exposed a Windows DLL lifetime leak in the auditor:
  successful arithmetic left the loaded library undeletable. The final
  implementation closes every handle in `finally`; success and injected
  runtime-failure cleanup are tested.
- Seven focused and 1028 all-color tests pass with three platform skips.
  Claim ceiling remains Windows x86_64 host
  streaming only: no media, producer algorithm, Android/Apple runtime,
  authorization or delivery state changes.

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

## 2026-07-28 - Prove P101-P104 canonical product chain on Android 14

- Reused the existing dual-ABI instrumentation package and compiled the exact
  P42 freestanding canonical core into its JNI library. A generated header
  carries all ten frozen P28-P30 payloads (14,254 canonical bytes) and their
  independently checked SHA-256 values; Android performs no JSON
  reinterpretation.
- JNI executes all ten hashes, proves invalid-input SHA rejection preserves
  the caller digest, and exhausts the eight-input staging-authorization truth
  table. Existing 4,096 quantizer vectors, exhaustive 8/16-bit EOTF roundtrip,
  ICC bytes and failure-atomicity checks remain in the same invocation.
- Two cold `-wipe-data -no-snapshot` Android 14 x86_64 emulator runs take
  78.97 and 73.75 seconds. Runtime report SHAs are
  `d11cf10c...3883` and `7f554b62...5628`; package/runtime facts and the
  stable identity `sha256:a3fa50e0...e03f20` are exact across both.
- Sixteen package/parser tests and 71 related regressions pass. Claim ceiling
  is Android x86_64 virtual runtime for consumer identity/staging arithmetic;
  physical arm64, Apple runtime, producer matching quality and product
  admission remain open.

## 2026-07-28 - Publish and verify P105 integration v16

- V16 binds P1-P104 payload `e73a42c` to read-only main `4eb571c` with
  384/333 changed paths, zero overlap and exact v15 ancestry. Manifest/schema
  SHAs are `0174ab22...b67ae` / `86459d13...ea8ad`.
- Direct/schema rebuild plus v15/v16 tests pass 26/26. A fresh detached merge
  produces tree `7b0da229...9b6a` and passes 1076 color-match tests with 22
  platform/data skips and zero failures; the temporary worktree was removed.
- State remains `review-ready-not-merged`; the main task owns review and merge.

## 2026-07-28 - Close P106 local arm64-emulator shortcut

- The official SDK repository exposes an Android 14 AOSP arm64 image, but
  Emulator 36.6.11 on this Windows x86_64 host exits before boot because QEMU2
  requires the image architecture to match the host. No APK code executes and
  no arm64 runtime claim opens.
- The runner now has explicit ABI-scoped AVD/port/report identities and rejects
  a host/image architecture mismatch before launch. Its x86_64 default remains
  backward-compatible and replays stable identity
  `sha256:a3fa50e0...e03f20`.
- The exact unused arm64 AVD and system image were removed after the failed
  preflight; the working x86_64 AVD was preserved. Five parser/target tests
  pass. Physical arm64 or a matching-host arm64 emulator remains external.

## 2026-07-28 - Add P107 capability-neutral invocation and P108 v17 handoff

- P107 adds a strict, hash-bound invocation profile that pins the complete
  producer/package/wheel/runtime/wire/capability/rights identity. The generic
  v2 transport executes the existing exact wheel while preserving the v1
  request bytes, and returns a profile-bound outcome. This is transport
  readiness only: the currently rejected capability is not renamed or
  promoted.
- The verifier now independently reconstructs the source/reference request
  views and domain-separated request ID before reading output. Capability,
  request ID or pixels-file mutation fails closed. Fifty related tests pass.
- P108 v17 binds P1-P107 `ee6595f` to read-only main `ebda2e1f` with 392/340
  changed paths, zero overlap, 47 exports and 21 schemas. Manifest/schema
  SHAs are `115e6bcb...652c4` / `efa1b18e...f2df3`.
- A fresh detached merge produces tree `5d55747e...22c8` and passes 1098
  color-match tests with 24 platform/data skips; its temporary worktree was
  removed. State remains `review-ready-not-merged`.
- D-PCT R0cm is recorded only as a future veto-only interface intent:
  `invalidate-reuse` may force refit; `not-invalidated-veto-only` can never
  authorize reuse. No consumer mapping exists before exact producer schema
  and fixtures are published.
