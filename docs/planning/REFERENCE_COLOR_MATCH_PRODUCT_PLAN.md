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
- Main-chat stock-first research and standalone D-PCT media work are
  concurrent and explicitly out of this branch's write scope. The latest
  read-only delivery-preflight boundaries are main `350b687` and D-PCT
  `f8ab191`; no mutable file from either checkout is consumed.

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
9. Shared input colours must remain stable across unrelated source-image
   contexts; reusing one recipe ID is not by itself batch consistency.

## Implementation tree

| Step | Status | Scope | Required verification | Commit point | Rollback |
|---|---|---|---|---|---|
| P0 | DONE | Coordination, contract, plan | Git/other-chat snapshot and diff check | `ec001b2`, `acb0c82` | revert commits |
| P1 | DONE | `LookRecipe` types, fitting and validation | 30 focused/existing colour tests pass | `1820af1` | revert commit |
| P2 | DONE | deterministic single/batch render | 44 focused/existing colour tests pass | `cabe6fc` | revert commit |
| P3 | DONE | JSON roundtrip and replay | 51 focused/existing colour tests pass | `8c7d45e` | revert commit |
| P4 | DONE | file-level SDR image adapter | 67 focused/preprocess tests pass | `23221fa` | revert commit |
| P5 | DONE | regression/integration evidence | 84 focused tests; full-suite result classified | `8d3f60c` | release claim |
| P6 | DONE | thin CLI + deterministic provenance report | 103 focused/preprocess tests pass | `33f5735` | revert commit |
| P7 | DONE | language-neutral recipe/report/composition schemas | 105 focused/preprocess tests pass | `c9fb7d9` | revert commit |
| P8 | DONE | language-neutral canonical recipe/plan identity | 112 focused/preprocess tests pass plus full-resolution CLI replay | `e1f03f6` | revert commit |
| P9 | DONE | fail-closed photographic-tail and promotion adjudication | 125 focused tests; 30-pair full-resolution repeat; full suite 981 pass/36 known fail | `8e03ed7` | revert commit |
| P10 | DONE | shared-colour cross-context batch-consistency gate | 131 focused tests; six-reference replay; full suite 987 pass/36 known fail | `4bfd5da` | revert commit |
| P11 | DONE | unpromoted-algorithm delivery fail-close + explicit research override | 134 focused tests; default/override CLI; full suite 990 pass/36 known fail | `adae6cb` | revert commit |
| P12 | DONE | propagate delivery certification into film-effects composition | 136 focused tests; full suite 992 pass/36 known fail | `ba6f2c1` | revert commit |
| P13 | DONE | CFSM fixed explicit-operator challenger and source-batch prior falsification | 126 focused tests; two byte-exact 30-pair v0 runs; full suite 998 pass/36 known fail | `ec437e1` | revert commit |
| P14A | DONE | preregister and test three analytic photographic priors | 10 dedicated tests; byte-exact validation repeat; confirmation correctly closed; full suite 1002 pass/36 known fail | `3debf13` | revert commit |
| P14B0 | DONE | pin and fail-closed adjudicate stable main-task W1 development evidence | 8 dedicated tests; 138 focused tests; current absent-report decision is identity; full suite 1010 pass/36 known fail | `8e602f5` | revert commit |
| P14B1 | DONE / ROUTE CLOSED | consume a repeated W1 decision without importing external code | byte-identical reports; strict source/hash/decision validation; full suite 1049 pass/36 known fail | `9cf7d6c` | keep identity default |
| P15 | DONE | freeze language-neutral portable recipe/render conformance vectors | 9 dedicated tests; 147 focused tests; two exact-ID and bounded-numeric cases; full suite 1019 pass/36 known fail | `8b505ca` | revert commit |
| P16 | DONE | replay one stored LookRecipe across a transactional N-file batch without the original reference | 9 dedicated tests; 40 adjacent tests; 156 focused tests; full suite 1028 pass/36 known fail | `378c846` | revert commit |
| P17 | DONE | commit outputs, optional recipe and provenance report as one rollback-safe run transaction | 5 dedicated fault-injection tests; 32 adjacent tests; 161 focused tests; full suite 1033 pass/36 known fail | `2f30826` | revert commit |
| P18 | DONE | bind film-effects composition to the actual delivered run safety decisions | 13 dedicated tests; 174 focused tests; full suite 1046 pass/36 known fail | `b44fc96` | revert commit |
| P19 | DONE / ROUTE CLOSED | audit and test a research-only empirical neutral-photography moment prior | 16 dedicated tests; 167 focused tests; three byte-exact artifact builds; two byte-exact confirmation runs | `9d82eda`, `e7317f1` | retain negative evidence; product stays identity |
| P20 | DONE / ROUTE CLOSED | test a bounded affine plus monotone-quantile non-moment challenger | 14 dedicated tests; 171 focused tests; two byte-exact stress-confirmation runs | `f73e3ef` | retain negative evidence; do not tune same candidate |
| P21 | DONE / ROUTE CLOSED | test source-batch-conditioned monotone quantiles against source-batch Gaussian | 12 dedicated tests; 175 focused tests; two byte-exact unused-row confirmations | `bdffb94` | retain Gaussian mechanism evidence; reject quantile extension |
| P22 | DONE / LINEAR ROUTE CLOSED | reproduce spatially invariant Lab statistics and test a generated-data residual ridge mapper | 8 dedicated tests; 183 focused tests; two byte-exact reserved-row runs | `104d7b3`, `0e8235a` | keep extractor; reject linear mapper and nonlinear rescue on same contract |
| P23 | DONE / EXTERNAL ROUTES CLOSED | reproduce obtainable CanonCGT/NLUT assets and audit StatLUT/SA-LUT/Neural Preset release boundaries | CanonCGT 20 focused tests and two byte-exact 6x6 reports; pinned rights/assets; no RGB retained | `208dfa8`, `f406234` | retain controls; no product promotion |
| P24 | DONE / DELIVERY READY | refresh both concurrent tasks, exercise fit/replay/research CLI, run full and clean-merge test evidence | 1081 pass/1 skip/36 known environment failures; fit/replay output hashes exact; latest-main synthetic merge 196/196 pass | delivery evidence commit | repository owner may merge after normal review |
| P25A | DONE | freeze equal-task Mode C ownership and D-PCT consumer semantics | current clean HEADs, durable claim, explicit allowed/forbidden files, peer intent and stable `59b72e0` snapshot | coordination commit | revert documentation commit |
| P25B | DONE | implement strict `MatchViewV1`, `TransformBundleV1`, `DiagnosticsV1` and `CapabilitiesV1` consumer contracts | 20 dedicated and 54 contract/schema/canonical/replay tests; compileall; four strict schemas | contract commit | revert additive contract commit |
| P25C | DONE | implement `WorkingImage` to compatible MatchView adapter | 16 adapter tests; 36 core tests; 123 adjacent product/preprocess tests; isolated read-only buffer and exact capability gate | adapter commit | revert additive adapter commit |
| P25D1 | DONE | bind valid core execution to A1/A4/A5 without bypassing delivered-pixel safety | 12 acceptance tests; 48 combined core tests; strict decision schema; only identity or candidate-for-product-guard | acceptance commit | revert additive acceptance commit |
| P25D2 | DONE | add synthetic consumer conformance fixtures without inventing a D-PCT producer artifact | 9 dedicated tests; 57 combined core tests; exact IEEE-754 vectors, descriptor/hash/schema checks and repeatable result | conformance commit | revert additive conformance commit |
| P25E | DONE | run change propagation, adjacent regressions and latest-main integration preflight; publish peer evidence bundle | 282 adjacent tests; full suite 1138 pass/1 skip/36 known failures; zero path overlap; synthetic merge 282 pass | evidence commit | retain prior stable commits |
| P26A | DONE | freeze a consumer-owned apply receipt that binds exact returned pixels without defining producer `ApplyResultV1` | Mode C intent, source/output/diagnostics/capability identity and candidate-only contract | `96fec4a` | revert documentation commit |
| P26B | DONE | implement prepared output, strict receipt schema/roundtrip and binding validation | 11 dedicated and 68 combined core tests; finite dense float32, same-profile/shape v1, mutation and swap tests | `f9fbd7d` | revert additive receipt commit |
| P26C | DONE | bind the exact receipt into a v2 guard-candidate admission decision | 11 dedicated and 79 combined core tests; no direct applied state; A1/A4/A5 and delivered-pixel guard remain mandatory | `497748c` | revert additive admission commit |
| P26D | DONE | run propagation, adjacent/full regression and peer handoff | 304 adjacent tests; full suite 1160 pass/1 skip/36 known failures; zero main-path overlap | evidence commit | retain P25 fallback |
| P27A | DONE | independently pin and audit D-PCT producer v1 schemas/exact-bit conformance | 4 lock tests; producer commit plus four schema, fixture, Python and C++ hashes; explicit closed verdict | compatibility-lock commit | revert lock/docs |
| P27B | DONE | consume corrected producer DiagnosticsV2/ApplyResultV2 from fixed producer line through `b1b68b6` | strict v2 lock; success `60e7466d...` and failure `9f7a3581...` fixtures; schema/code/native hashes | `60b3be8`, `1a8f6c4` | keep producer v1 and superseded v2 fixture closed |
| P27C | DONE | implement explicit producer-v2 to consumer adapter | exact byte length, pixel/view/bundle/diagnostics/result recomputation; no mutable import | compatibility-v2 commit | identity fallback |
| P27D | DONE | pass corrected producer exact fixture through consumer receipt/admission | dual producer/consumer identities, exact output receipt, candidate-only A1/A4/A5 chain | compatibility-v2 commit | identity fallback before guard |
| P27E | DONE | run adjacent/full/latest-main propagation and peer handoff | 1179 pass/1 skip/36 known environment failures; latest-main synthetic merge 284 pass; zero path overlap | evidence commit | retain P26 boundary |
| P28A | DONE | freeze atomic one-reference/N-source producer intake | ordered source ownership, all-or-fallback, no applied state | `ccc721c` | retain per-source P27 receipts |
| P28B | DONE | implement success/failure batch binding | verify every producer/consumer source and shared reference identity | batch implementation commit | reject mixed/unbound outcomes |
| P28C | DONE | bind per-source A1/A4/A5 admission into atomic batch result | any producer/admission failure makes the entire batch identity fallback | batch implementation commit | never expose partial delivery |
| P28D | DONE | run mutation/permutation/full/latest-main propagation and handoff | 1191 pass/1 skip/36 known environment failures; latest-main synthetic merge 296 pass; zero path overlap | evidence commit | retain P27 single-source path |
| P29A | DONE | freeze exact-receipt numeric delivery guard | producer factual fractions plus consumer new-boundary metric; no aesthetic claim | `7ca407b` | retain pending-product-guard |
| P29B | DONE | implement per-source guard decision | exact source/output/admission binding and conservative frozen thresholds | `0fca9d9` | identity fallback |
| P29C | DONE | implement atomic batch guard | all sources eligible or the full batch falls back | `135f0b9` | no partial/applied state |
| P29D | DONE | run adversarial/full/latest-main propagation and handoff | 1218 pass/1 skip/36 known environment failures; latest-main synthetic merge 65 pass; zero path overlap | evidence commit | retain P28 resolution |
| P30A | DONE | freeze product transaction authorization boundary | bind P28/P29 plus original acceptances; forbid research override | `ad836ba` | retain numeric-only state |
| P30B | DONE | implement atomic authorization envelope | every row must be promoted, non-research and identity-bound | `88b7d53` | full identity fallback |
| P30C | DONE | prove research override and binding mutations fail closed | promoted control plus override/status/order/hash negatives; 76 focused pass | `88b7d53` | no staging authorization |
| P30D | DONE | run full/latest-main propagation and peer handoff | 1229 pass/1 skip/36 known environment failures; synthetic merge 76 pass | evidence commit | retain P29 |
| P31A | DONE | freeze language-neutral P28-P30 product-chain vectors | two discriminating cases and ten canonical identities | `7e34a83` | retain Python contracts |
| P31B | DONE | independently recompute identities and staging rule in C++17 | MSVC `/W4 /WX`, 10/10 identities, research override negative | `7e34a83` | retain Python authority |
| P31C | DONE | cross-compile/link consumer verifier for Android | pinned NDK r27d arm64-v8a and x86_64 ELF evidence | `52d4cd8` | no device-runtime claim |
| P31D | DONE | run full/latest-main propagation and peer handoff | 1235 pass/1 skip/36 known failures; latest-main synthetic merge 82 pass | evidence commit | retain P30 |
| P31E | DONE | independently execute the same verifier with a second Windows compiler | pinned LLVM-MinGW Clang 22.1.8, 10/10 identities and both states; full 1236 pass | `5d809ef` | same-host evidence only |
| P32 | DONE | audit the complete long-term goal and freeze the real integration critical path | requirement-by-requirement ownership, evidence, status and next-owner matrix | `f1ea3fb` | remove documentation only |
| P33A | DONE | freeze an external-candidate staging transaction without defining producer invocation | exact P28/P30/intent/receipt binding; all-or-nothing SDR files and report; staging-only claim | `ce0ceaa` | retain P30 authorization |
| P33B | DONE | implement strict transaction report and atomic file staging | rollback fault injection, schema/roundtrip, exact output hashes | `0cb94c3` | revert additive module/schema |
| P33C | DONE | prove mutation and fallback branches cannot write | authorization/order/receipt/profile/intent/path/commit negatives | `0cb94c3` | no staged artifacts |
| P33D | DONE | run adjacent/full/latest-main propagation and peer handoff | 1245 pass/1 skip/36 known failures; latest-main synthetic merge 107 pass | `36a1a98` | retain P30/P32 |
| P34A | IN PROGRESS | freeze restart-safe verification of a committed P33 run | expected report hash/run ID, bounded strict read, exact output file hashes | intent commit | retain P33 staging report |
| P34B | PENDING | implement canonical verified-staging binding | report/run/output/receipt/path identities; no write or delivery state | implementation commit | revert additive module/schema |
| P34C | PENDING | prove report and output tampering fail closed | report bytes, run ID, output bytes/path/order/hash mutations | implementation commit | no verified binding |
| P34D | PENDING | run adjacent/full/latest-main propagation and peer handoff | focused, full and clean synthetic merge evidence | evidence commit | retain P33 |

At most one row may be `IN_PROGRESS`. A row becomes `DONE` only when its
verification evidence and commit are recorded in the branch log.

## Equal-task D-PCT integration decision

Parent Plan Node: Neuro-Film uploaded-reference product capability. P25 is an
additive consumer-integration child of the delivered P1-P24 module; it does not
reopen or replace D-PCT's independent research tree.

The standalone D-PCT task and this Neuro-Film task are equal, autonomous
collaborators. D-PCT is the sole authority for matching algorithms, canonical
scene/display rails, RAW/HDR/video media interpretation, portable transform
components and native execution. This module is the sole authority for the
Neuro-Film product boundary, transaction/replay/report behavior, delivered
safety decisions, A1/A4/A5 acceptance and FilmFX composition.

The integration model is producer/consumer, not source copying. Neuro-Film
consumes only fixed schemas, package/ABI identities and conformance fixtures.
It does not import a mutable D-PCT checkout. D-PCT does not reproduce
Neuro-Film transaction, film-business or FilmFX behavior.

One batch-shared reference intent does not imply one batch-shared algorithm
transform. Some D-PCT candidates fit from both source and reference. Their
`TransformBundle` is therefore source-bound unless independent A1/A4/A5
evidence proves a source-independent operator. Neuro-Film may persist the
shared intent/policy and an ordered set of source-bound transform bindings
without overstating batch consistency.

Change propagation is mandatory after every P25 interface leaf. Upward impact
checks the product claim and identity fallback; downward impact checks adapter,
replay and conformance children; sideways impact checks recipe/report/safety,
FilmFX composition, D-PCT producer compatibility and latest-main consumers.
Validation and documentation evidence are then re-integrated bottom-up before
the leaf or parent can close.

Parallel integration owner: this task integrates only the Neuro-Film consumer
branch. The D-PCT peer independently integrates its producer branch. A future
cross-repository release requires an explicit compatibility/integration leaf
that pins both commits and schemas; neither task may silently integrate the
other's mutable checkout.

P26 closes an additional consumer integrity gap without extending producer
authority. The consumer copies the bytes returned by a future core, computes
its own exact output identity and issues a `CoreApplyReceiptV1`. The receipt is
not D-PCT's `ApplyResultV1`, does not prescribe an ABI and cannot be accepted
as a delivered image. V1 permits only finite dense float32 RGB with the same
shape and exact profile as the bound source. Any future output-profile
conversion requires a separately versioned trusted bridge. A v2 candidate
admission must bind the receipt ID as well as A1/A4/A5; the existing v1
acceptance record remains valid historical evidence but is insufficient to
authorize future external pixels by itself.

P27 is a separate compatibility program, not a continuation by naming. Its
lock independently reproduces producer artifact hashes from a fixed Git
commit. Producer and consumer canonical IDs remain distinct. A compatibility
adapter may strip the producer hash prefix for the identical f32be pixel hash,
but must preserve producer IDs as authoritative aliases and regenerate
consumer IDs under consumer rules. Producer v1 is deliberately closed because
its diagnostics do not supply the facts required by the consumer and its
ApplyResult does not bind source geometry. Only a fixed producer v2 schema,
fixture and cross-language result may reopen P27B.

P27B-D require the corrected producer line beginning at `11c581e`; the final
lock advances to `b1b68b6` to include an exact failed-diagnostics vector. The earlier
`281b13f` v2 fixture is permanently rejected: its hashes were internally
self-consistent but its output reused unclipped source pixels while reporting
hard clipping. The corrected fixture binds a real `[0,1]` output and reports
three changed channel samples out of twelve. The adapter independently
recomputes every producer identity and then regenerates consumer identities;
it never uses dynamic producer diagnostics/result IDs as recipe or cache
identity. Even a valid fixture produces only a consumer `candidate-only`
receipt. Product use remains identity fallback until A1/A4/A5 and the
delivered-pixel guard pass.

The failed vector has `bundle_id=null`, `measurements=null`, no result and an
explicit producer failure code. Neuro-Film can verify its canonical ID but
cannot manufacture a transform, diagnostics measurements or receipt from it;
its only action is identity fallback. Both pinned fixtures are forced to LF so
their producer-published artifact SHA-256 survives a fresh Windows checkout.
The later proposed absolute BT.2020 HDR rail is a separate, currently unmapped
profile and does not inherit this relative-sRGB compatibility verdict.

P28 is a consumer product-transaction child, not a producer algorithm API.
The user-visible source order is explicit and contiguous. Every per-source
candidate must bind its own source and the same reference in both producer and
consumer identity spaces. A producer failure short-circuits before admission;
otherwise every source must have a receipt-bound A1/A4/A5 admission. Any
fallback makes the whole batch identity fallback. Even an all-success batch
stops at `pending-product-guard`; partial delivery and `applied` are absent
from the contract.

P29 is a numeric delivered-pixel gate after P28, not a replacement for visual
severe-artifact review or aesthetic evidence. It consumes producer-owned
factual OOG/clipping/projection fractions without recomputing or overriding
their meaning, and independently measures only the consumer-owned fraction of
new source-relative boundary pixels from exact receipt bytes. A per-source
pass means only `eligible-for-transaction`; a complete N-source pass means the
whole batch is eligible to enter a future durable transaction. Every source
must pass the same frozen policy or the full batch becomes identity fallback.
Neither state means delivered or applied. See
`docs/drpt/REFERENCE_COLOR_MATCH_NUMERIC_GUARD_EVIDENCE.md`.

P30 is the product transaction authorization boundary, not another promotion
evaluator. It rebinds the original acceptance identities already referenced by
P28 and requires every source to have `promotion_status=promoted` with
`research_baseline_override=false`. This prevents an explicitly accepted
research baseline from crossing a product staging boundary merely because its
pixels pass P29 numeric thresholds. Success means only
`authorized-for-staging`; commit and applied delivery remain later states.
See `docs/drpt/REFERENCE_COLOR_MATCH_PRODUCT_AUTHORIZATION_EVIDENCE.md`.

P31 freezes the P28-P30 identity chain independently of Python object layout.
The C++17 verifier hashes only the published language-neutral canonical bytes
and implements only the final promoted/non-research staging state rule; it is
not a second producer algorithm. Windows execution and Android cross-linking
do not establish Android device, Apple, JNI, image-I/O or product-delivery
readiness. See
`docs/drpt/REFERENCE_COLOR_MATCH_PRODUCT_CHAIN_CONFORMANCE.md`.

P32 is a completion audit rather than a new interface. It distinguishes the
finished fail-closed consumer module from the still-open real external
algorithm path. A synthetic promoted fixture, a schema-compatible receipt or
an Android cross-link cannot substitute for a genuinely promoted candidate,
a fixed producer invocation package, atomic committed delivery, or target
runtime evidence. See
`docs/drpt/REFERENCE_COLOR_MATCH_COMPLETION_AUDIT.md`.

P33 may prepare the consumer-owned transaction endpoint before a real
producer invocation exists, but cannot weaken the dependency. It accepts only
an already authorized P30 batch and the exact in-memory P27 receipts that
P28 bound. Every source must share one reference intent while retaining its
source-bound transform. Success commits files only to product staging and
remains neither delivered nor applied. Synthetic tests prove mechanics only;
a real product claim still requires the P32 critical path.

P34 is the restart boundary after P33. It trusts neither the in-memory commit
result nor path existence alone: the caller supplies the expected report hash
and run ID, the consumer rereads bounded strict JSON, validates its canonical
identity and verifies every staged file hash. Success is only a
`verified-staging` binding. It cannot create final delivery or FilmFX
authority.

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
- P6: `103 passed` after adding a thin one-reference/N-source CLI and atomic
  JSON report. Reports bind reference file/pixel hashes, recipe identity,
  source/output hashes, candidate diagnostics and safety/fallback decisions.
  Batch cardinality failure exits with code 2 before any output, recipe or
  report is created.
- P7: `105 passed` after adding strict JSON Schema 2020-12 contracts for the
  recipe, composition plan and run report plus composition JSON parsing.
  Independent validators reject unknown fields, invalid hashes/ranges, hidden
  film-colour claims, inconsistent composition order and contradictory
  safety/action/reason combinations.
- P8: `112 passed` after replacing Python JSON-float-dependent identity hashing
  with a typed canonical byte stream. A frozen multilingual/multitype vector
  binds strings, integers, binary64 floats, arrays and sorted objects. The
  full-resolution CLI smoke repeats with identical pixel decisions and the new
  portable recipe/report identities.

### Pre-existing worktree-line-ending failure

The unrelated `tests/test_render_contract.py` has five failures in this
worktree because `configs/color_rendering_profiles.yaml` is checked out with
48 CRLF line endings. Its worktree SHA-256 is `a29663b2...`, while the Git blob
and tracked profile manifest both use the LF-byte SHA-256 `d919402a...`. No
reference-match commit modifies either file. This branch records the failure
but does not rewrite protected legacy profile hashes or shared renderer assets.

The latest complete CPU collection reached
`1046 passed, 1 skipped, 36 failed`.
All failures were either the same checked-out-byte hash class or tests whose
ignored `outputs/` evidence is not copied into a new Git worktree. No
`src/color_match` test failed and no new failure family appeared.

P15 adds a versioned portable conformance bundle rather than assuming Python
repeatability proves cross-platform agreement. Reference, source and expected
output pixels use exact IEEE-754 binary32 big-endian hexadecimal bits; recipe
and reference-pixel identities must match exactly, while rendered pixels use
explicit linear-RGB, Delta E76 and diagnostic tolerances. The current vectors
exercise both linear-sRGB source-segment gamut compression and linear-Rec.2020
chroma-only compression. See
`docs/drpt/REFERENCE_COLOR_MATCH_PORTABLE_CONFORMANCE.md`.

This makes independent Android/iOS/macOS/Windows ports testable. It does not
claim that any such port has passed; real platform implementations and device
results remain required before a cross-platform product-freeze claim.

## Phase-two gates

The product shell is ready, but the statistical recipe is not frozen as the
final photographic/aesthetic algorithm.

| Gate | Status | Dependency | Allowed next action |
|---|---|---|---|
| A1 reference identifiability | OUTPUT-ONLY, FIXED-MOMENT AND MARGINAL-QUANTILE ROUTES CLOSED | local known-operator cross-content falsification, repeated main W1 evidence, P19 empirical moment confirmation and P20 stress confirmation | do not rescue the same descriptor/distribution fitter with capacity or tuning; require user information or a separately justified learned perceptual objective |
| A2 film-business composition | CONTRACT DONE / DELIVERY-AWARE | existing v1 render-profile contract plus guard-v2 certification state | default identity cannot masquerade as reference colour or silently compose effects; explicit research mode may bind effects provenance only |
| A3 media portability | CONTRACT MAPPED / PIXEL BRIDGE CLOSED | D-PCT `bd3ff70` executes 67/67 local compression-7 DNG mosaics, 52/52 observed profile-look paths, all 17 CR2 entropy paths and LibRaw unpack for 32/39 vendor RAW files; all 67 DNGs and 686,122,932 post-linearization sensor codes agree with pinned LibRaw, but `real_raw_paths=FAIL`, seven Nikon HE/HE* files remain unsupported and the current NFRM relative-SDR rail is not equivalent | wait for independent crop/black/demosaic/profile and trusted scene-render agreement plus a versioned scene/display-to-MatchView bridge; do not copy decoders |
| A4 photographic preference | BASELINE REJECTED | six-image known-operator slice completed; broader frozen suite and blind review remain open | compare identified challengers under severe-artifact veto and blind aesthetic review |
| A5 album/batch consistency | BASELINE FAILED | six fitted recipes on shared-colour/different-context probes | require a fixed explicit operator or bounded adaptation that passes median/p95/max shared-colour drift |

Until A1/A4/A5 pass, this implementation is an operational deterministic
baseline, not the claimed final or strongest colour-matching algorithm.

The first A4 falsification uses one known Velvia-look target as the reference
and applies its recipe to six neutral sources. The same-content positive control
improves median Delta E76 by 59.8%, but every held-out-content image regresses
(-3.8% to -175.4%). See
`docs/drpt/REFERENCE_COLOR_MATCH_QUALITY_BASELINE.md`. This rejects global
reference moments as the final algorithm and prevents parameter-tuning from
being mistaken for content-independent look recovery.

The default file path now applies `reference-render-guard.v2` after candidate
rendering. Because the current algorithm fails A1/A4/A5, default delivery
returns identity with `algorithm-not-promoted`, even when its pixel-tail
thresholds pass. Gamut repair above 25% or newly introduced encoding-boundary
pixels above 5% add independent reasons. Candidate diagnostics remain in the
report.

Research can explicitly request `--allow-research-baseline`. The override is
recorded as `research_baseline_override=true` on every output and does not
disable gamut/new-boundary vetoes. This preserves experimentation without
presenting a rejected algorithm as the product default.

A six-reference/full-cross matrix confirms that distinction. All six
same-content controls improve, but 25/30 cross-content candidates regress
(median -92.2%, worst -344.9%). The guard admits none of those 25 regressions,
while conservatively rejecting three improvements. Its thresholds remain a
severe-tail veto; they must not be tuned to make the baseline appear more
successful.

Full-covariance Gaussian/MKL transport was also rejected on the same matrix.
It raises same-content median improvement from +60.0% to +65.2%, but reduces
cross-content improvements from 5/30 to 3/30 and worsens cross-content median
from -92.2% to -109.5%. The next algorithm must recover a content-independent
grade; stronger unpaired global-distribution fitting is no longer an eligible
product direction.

CFSM-v0 is the first challenger to repair the old batch and severe-tail
failures without collapsing to identity. It estimates an orientation-preserving
Gaussian transport from an image-independent canonical RGB cube to the
reference, converts the residual to a boundary-pinned tetrahedral 17-cube, and
bisects strength until output range, residual, smoothness, neutral-axis and
positive-Jacobian constraints all pass. The resulting LUT is fixed across N
sources, canonically identified and JSON replayable.

On the same frozen 30 cross-content rows, CFSM-v0 improves 19/30, reaches
`+3.851%` median and `-9.649%` worst improvement, and introduces zero boundary
pixels. All 6/6 photographic probes and 6/6 context-invariance probes pass;
shared-colour median/p95/max drift is exactly zero. It is nevertheless
`rejected` because the frozen gates require 75% improved rows and +10% median.
The repeat report is byte-identical: report ID
`1ead5590...574c7f`, SHA-256 `ccf34c09...acf243`.

Increasing neutral-axis allowance from 0.04 to 0.08/0.12 worsens the tail and
does not pass A1. Replacing the fixed cube with the uploaded N-source batch as
the prior also fails: 16/30 improve, median is `+0.571%`, worst is `-11.524%`.
This closes naive source-batch distribution fitting. P14B must replace the
unidentified prior with independent canonicalizer evidence rather than tune
strength or absorb the current batch's content distribution.

P14A separately preregisters three data-free photographic priors and keeps
generated-operator development, validation and confirmation slices disjoint.
The uniform-cube control wins both visible splits. On development it improves
43/48 observations with median captured style `+9.70%`; the best analytic
prior reaches only 25/48 and `+0.47%`. On validation uniform reaches 19/24 and
`+6.92%`; the best analytic prior reaches 13/24 and `+1.02%`. Validation is
byte-repeatable at report ID `d08ff8b...53fe2`, SHA-256
`816ba59e...f6549`. No analytic prior is selected, so synthetic confirmation
and the real 30-pair matrix remain unopened. P14B therefore requires learned
or otherwise independently identified canonicalization, not another
hand-shaped moment prior.

P14B0 pins the stable main-task W1 implementation boundary at ancestor
`77df1b9` and exact SHA-256 identities for its config, runner, descriptor,
explicit flow and synthetic utilities. Its receiver compares two reports
byte-for-byte, revalidates the external Git source at the recorded commit,
recomputes the development decision branch from the gate booleans and rejects
reserved-confirmation access or source drift. A single-reference unseen-look
development pass can open only untouched synthetic confirmation. Multi-
reference or seen-bank-only passes do not satisfy arbitrary one-reference
upload, and no development branch opens product integration. With no W1
reports currently published, the canonical decision is `not-ready` and
delivery remains identity.

P14B1 consumes the completed repeated W1 decision without importing its Python
implementation. Both external reports are byte-identical at
`9b42e9a8...8dec68`, bind software commit `b72b594...`, leave confirmation
untouched and independently recompute `paired_upper_bound_only_passes`.
The exact paired neutral/styled upper bound succeeds, but every output-only
single/four-reference method fails content, baseline or identity controls.
The committed decision is therefore `development-route-closed`, canonical ID
`f8661315...f60e`; product integration remains false and delivery remains
identity. A larger, semantic or neural descriptor is not an eligible rescue on
the same evidence.

P19 tests a genuinely independent neutral-photography population rather than
another hand-shaped RGB distribution. A strict builder reads only the 128
FiveK `raw_default_srgb16` controls, validates every file and official licence
assignment, and aggregates each image with equal weight into linear-sRGB mean
and population covariance. Expert/target images are never read. Three builds
are byte-identical at artifact SHA-256 `d590f75a...895d`, prior ID
`bb823874...b0ae`, covering 201,547,776 pixels; 59 source images map to the
Adobe list and 69 to the Adobe+MIT list.

The official terms are research-only and prohibit commercial advantage, so
the artifact cannot open product or commercial use even if it succeeds. It
does not succeed. On the previously unopened synthetic confirmation split,
uniform CFSM reaches 31/48 improvements and median captured style `+4.63%`;
the empirical moment prior reaches 15/48 and `-6.04%`. Median gain over the
control is `-10.67` points and worst-case loss is `5.22` points. Both repeated
reports are byte-identical at report ID `ca7e768d...cd0e`, SHA-256
`2a995096...f387`. Constraints, zero fallback and zero new boundary pixels
pass, but identification gates fail. This closes fixed global first/second
moment priors as the arbitrary-reference solution; more neutral photos cannot
rescue the same statistic.

P20 changes the estimator, not merely its data. It alternates a symmetric
Gaussian colour transform with eight fixed monotone per-channel quantile
splines, samples that operator on the same 17-cube, and applies the existing
boundary, smoothness, neutral-axis and positive-Jacobian projection. It
therefore remains one replayable global LUT across the complete batch.

On development, quantile improves 44/48 versus uniform 43/48 but has slightly
lower median captured style (`+9.49%` versus `+9.70%`). The gate and the
previously unopened stress operator/palette split were frozen before this was
observed. On stress confirmation it improves 38/48 versus 37/48 and raises
median captured style from `+11.70%` to `+12.96%`, but the `+1.26` point gain
misses the frozen `+3` point minimum and the worst row loses `1.15` points.
All structural gates pass. Two reports are byte-identical at report ID
`785d0d14...a478`, SHA-256 `cf39f27c...1c99`; the route closes without
real-photo review or product integration.

P21 allows a materially different input contract: the candidate may inspect
the complete N-source batch, but must remain one fixed LUT inside that batch.
It compares Gaussian source-batch CFSM with an affine plus marginal-quantile
extension on previously unused stress operator indices 8--15. Gaussian
improves 40/48 with median `+9.32%`; quantile also improves 40/48 with median
`+9.13%`. Quantile's median gain is `-0.19` points, although its worst case is
`1.51` points better. The repeated report is byte-identical at ID
`e5a2d670...c14a`, SHA-256 `c9d261ad...05b1`. The extension closes, and the
Gaussian result remains synthetic mechanism evidence only because the earlier
real matrix failed its product gates and a batch-conditioned recipe is not a
reference-only reusable recipe.

P22 follows the July 2026 StatLUT paper's published spatially invariant feature
idea without copying a model or claiming reproduction: soft-binned Lab
lightness, square-root joint chroma and chroma-conditioned lightness. The
extractor is exactly invariant to pixel permutation and passes strict
normalization tests. A compact generated-data ridge maps source/style/residual
features to nine bounded explicit-operator parameters.

The selected ridge (`alpha=100`) looks adequate on the visible validation
distribution (`0.0279` median grid RMSE) but collapses on the frozen unused
stress half: `0.1267` median, `0.1814` p90, median captured style `-104.94%`,
and only 16.67% improve identity. Same-look replicate error is `0.1222` and
identity-reference maximum error `0.1637`. Every gate fails, with byte-exact
report ID `f1e2bdb3...41eb`, SHA-256 `0b4cf6c9...8716`. The extractor remains
useful infrastructure; this linear residual mapper and an unprincipled
capacity rescue on the same generated contract close.

### Film-business composition boundary

`ReferenceCompositionPlan` makes reference matching and film simulation peer
colour choices and now binds delivery certification.

1. With the current unpromoted algorithm, the default plan is
   `identity_color`, output label/claim `identity`, and
   `reference_color_status=identity-fallback`.
2. Film effects cannot be silently attached to that unavailable reference
   colour; users must choose the separate film-simulation mode.
3. Explicit research override changes the status to `research-baseline`,
   makes `reference_color` the sole colour stage, and records
   `research_baseline_override=true`.
4. In that research-only plan, an optional verified film profile may
   contribute only procedural `film_effects`.
5. `film_color_profile_id` remains null and effects never set
   `film_stock_identity_claimed`.
6. Research output is labeled `reference-look` or
   `reference-look+film-effects`, with claim ceiling `reference-look`.
7. Product composition is valid only after binding the transactional run
   report and delivered output hashes. Uniform applied delivery may bind
   reference colour and optional effects; uniform fallback may bind only
   identity without effects; mixed delivery cannot form one batch plan.

This prevents a UI choice such as "match this photo, add film grain" from
silently applying effects after a colour stage that actually fell back to
identity, stacking two colour looks, or escalating a user reference into a
stock-authenticity claim. Stock-selection mode continues to use the existing
render-profile path and is not replaced by this module.

### D-PCT media boundary

The standalone D-PCT contract has two canonical algorithm rails:

- scene-relative linear ACEScg/AP1/D60 float32;
- display-absolute linear CIE XYZ/D65 float32 with explicit reference-white
  luminance.

The current reference matcher accepts display-linear relative linear-sRGB SDR.
Those are different colour states, so a D-PCT frame cannot be re-labeled as a
`WorkingImage`. A future adapter must perform a versioned pixel conversion and
record at least rail/domain, reference-white nits, render-bridge ID, decoder
provenance and gamut/OOD decision.

Until that bridge exists:

- SDR decoded by D-PCT is not automatically compatible;
- scene-relative RAW requires a versioned scene-to-display render bridge;
- PQ/HLG/gain-map HDR remains rejected by this matcher;
- D-PCT keeps ownership of RAW/HDR/video decoding and cross-platform media
  execution.

This is an intentional fail-closed integration result, not an unfinished
metadata rename.

## Current executable entrypoint

```powershell
python scripts/match_reference_color.py `
  --reference reference.png `
  --source source-a.jpg --output matched-a.png `
  --source source-b.tiff --output matched-b.png `
  --recipe reference-look.json `
  --report reference-match-report.json `
  --bit-depth 16
```

The command writes one recipe, N ordered identity-fallback outputs and one
deterministic provenance report while the only implemented algorithm remains
unpromoted. Its stdout is a compact JSON summary with applied and
identity-fallback counts. Add `--allow-research-baseline` only for explicit
research comparison. The script is a thin product adapter; algorithm, safety
and transactional image behavior remain in `src/color_match`.

P17 makes those run artifacts one transaction rather than writing the report
after the image batch. The reference/replay path validates report collisions
before staging, computes output and recipe hashes from staged bytes, stages
the report from that bound state, then commits outputs, optional recipe and
report through one backup/rollback sequence. A report-build or report-commit
failure leaves either no new run artifacts or restores every prior
destination byte. See
`docs/drpt/REFERENCE_COLOR_MATCH_RUN_TRANSACTION_EVIDENCE.md`.

The saved recipe can later process a new N-image batch without retaining or
re-reading the original reference:

```powershell
python scripts/match_reference_color.py `
  --recipe-input reference-look.json `
  --source new-a.jpg --output matched-new-a.png `
  --source new-b.tiff --output matched-new-b.png `
  --report reference-match-replay-report.json `
  --bit-depth 16
```

Fit and replay are mutually exclusive modes. Replay reads and hashes the exact
same bounded UTF-8 recipe bytes once, verifies the embedded canonical ID, and
transactionally commits either all N outputs or none. Its separate provenance
schema records `operation=recipe-replay`, the loaded recipe-file hash and the
reference-pixel identity stored in the recipe; it does not invent a current
reference path or require that the original reference file still exists. See
`docs/drpt/REFERENCE_COLOR_MATCH_REPLAY_EVIDENCE.md`.

Language-neutral consumers should validate:

- `configs/schemas/reference_look_recipe_v1.schema.json`;
- `configs/schemas/reference_composition_v1.schema.json`;
- `configs/schemas/reference_match_report_v1.schema.json`;
- `configs/schemas/reference_match_replay_report_v1.schema.json`;
- `configs/schemas/reference_match_promotion_report_v1.schema.json`.

Python validation remains authoritative for canonical IDs and constraints such
as shadow floor below highlight ceiling; platform implementations must enforce
both the JSON Schema and the documented canonical SHA-256 construction.
The canonical byte grammar and frozen parity vector are specified in
`docs/reference/REFERENCE_MATCH_WIRE_FORMAT_V1.md`.

## Promotion adjudication boundary

`src/color_match/promotion.py` and
`scripts/evaluate_reference_match_promotion.py` make the A4 decision order
executable:

1. known-operator cross-content recovery and boundary tail;
2. independently fitted recipe probes for neutral-axis chroma, tone reversal,
   tone plateaus, new boundary pixels and extreme skin/sky/foliage hue motion;
3. shared input-colour probes embedded in dark/cool and bright/warm unrelated
   surroundings, evaluated across every independently fitted recipe;
4. only after all automated gates pass, an independent blinded aesthetic
   review with a zero-severe-artifact veto.

Automated success returns `eligible-for-visual-review`, never `promoted`.
Promotion requires at least 12 blinded rows, preference strictly above 50% and
zero severe artifacts. The default known-operator gate requires at least 12
cross-content rows, 75% improvement, positive 10% median gain, worst-row gain
no lower than -10% and at most 5% new boundary pixels.

The six-reference/30-cross-content Velvia known-look replay is rejected:
5/30 rows improve, median improvement is -92.19%, worst is -344.87% and the
maximum new-boundary fraction is 21.86%. Only two of six fitted recipes pass
the photographic probe; the worst probe introduces 12.69% new boundary
pixels. The portable report ID is
`2f7b8b2c...736411`; paths remain provenance but are excluded from its
content-bound identity.

All six baseline recipes fail the shared-colour context-invariance gate. The
worst median/p95/maximum drift is `60.8633/77.3292/80.1493` Delta E76, against
frozen limits `0.5/1.0/3.0`. This proves that per-source Lab normalization can
change the same colour solely because the rest of the photograph changes.
The current implementation remains deterministic and order-stable, but it is
not strong album consistency. A replacement head must emit a fixed explicit
operator or demonstrate bounded adaptation under this gate.

## Verification and rollback

Focused tests run before broad tests. Each verified leaf receives a scoped
commit. No generated images, datasets, weights or output caches are committed.
Rollback is commit-level and never requires resetting or rewriting another
chat's work.
