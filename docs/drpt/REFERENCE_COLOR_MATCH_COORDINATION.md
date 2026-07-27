# Reference Color Match Coordination

## Claim: NFCM-P1 product reference-look engine

- Mode: C (same-project multi-chat concurrency)
- Owner/chat: `019f9f37-91d9-7b11-b135-ad62bcb32214`
- Parent: neuro-film product capability — uploaded reference photo matching
- Status: ready-for-integration (first vertical slice); phase-two claim remains
  active for evidence-gated algorithm upgrades
- Scope: add an image-first product API that fits one replayable reference-look
  recipe from one uploaded reference and applies it consistently to one or more
  `WorkingImage` sources.
- Files/artifacts allowed:
  - `src/color_match/**`
  - `scripts/match_reference_color.py`
  - `tests/test_color_match_*.py`
  - `configs/schemas/reference_*.schema.json`
  - `docs/planning/REFERENCE_COLOR_MATCH_PRODUCT_PLAN.md`
  - `docs/drpt/REFERENCE_COLOR_MATCH_COORDINATION.md`
  - `docs/drpt/REFERENCE_COLOR_MATCH_AGENT_LOG.md`
  - `docs/drpt/REFERENCE_COLOR_MATCH_EVIDENCE.md`
  - `docs/reference/REFERENCE_MATCH_WIRE_FORMAT_V1.md`
- Files/artifacts forbidden:
  - `src/roll2film/reference_look_identifiability.py`
  - `tests/test_reference_look_identifiability.py`
  - `configs/u5_r2w1_*`, `configs/u5_r2w2*`
  - `docs/U5_R2W0_*`, `docs/planning/U5_R2W1_*`,
    `docs/planning/U5_R2W2*`
  - S4/U1 experiment code, configs, reports and outputs
  - `IMPL_PLAN.md`, `TASK_BOARD.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`,
    `docs/drpt/AGENT_LOG.md`
  - the separate `C:\Users\hhvrf\Documents\追色` repository
- Dependencies:
  - hard: existing `WorkingImage`, D65 Lab primitives, safe-Lab and gamut
    policies at base commit `c03c321b9fc642e2e092d59e20dd1b145b96192d`
  - soft: W1/W2 reference-look identifiability evidence from chat
    `019f4b76-e70a-75c0-b7ea-b473ab38c200`
  - soft: D-PCT RAW/HDR/video/media findings from chat
    `019f9f3b-d0c2-7f21-b486-1dd902148739`
- Interface contract:
  - input is one reference `WorkingImage` and one or more source
    `WorkingImage` objects;
  - fitting produces immutable, versioned, JSON-serializable `LookRecipe`
    data;
  - full-resolution output is produced only by deterministic explicit colour
    operators;
  - the first slice accepts explicit supported working spaces and
    display-linear SDR only, and fails closed elsewhere;
  - output claim ceiling is `reference-look`; it is never stock identification
    or `calibrated-reference`;
  - recipe, report and composition payloads have strict language-neutral JSON
    Schema contracts for non-Python consumers;
  - an algorithm that has not passed A1/A4/A5 defaults to identity delivery;
    research execution requires an explicit, report-bound override and retains
    all independent pixel-tail vetoes;
  - composition propagates the same state: default output is labeled identity
    and cannot attach film effects; explicit research override may attach
    verified procedural effects but never film colour or stock claims;
  - recipe and composition identity use the frozen typed canonical byte stream,
    never implementation-specific JSON float formatting;
  - a fitted recipe is shared across the batch and the reference target/policy
    cannot change;
  - per-source adaptation cannot support a strong batch-consistency claim
    unless shared input colours pass the frozen cross-context median/p95/max
    drift gate.
- Expected evidence:
  - focused unit tests for fitting, replay identity, batch invariants,
    fail-closed boundaries and no input mutation;
  - existing Lab/gamut/preprocess regression tests;
  - scoped commits and clean diff review.
- Stop condition:
  - pause if another chat claims or modifies `src/color_match/**`, or if the
    implementation requires changes to a forbidden file/public contract;
  - do not start GPU training, external downloads, app/UI work or media codec
    work under this claim.
- Claim expires: 2026-08-03 or immediately after integration/release.
- Integration owner: this chat prepares the branch evidence bundle; the
  neuro-film main development owner decides final merge after its S4/W1/W2
  dirty state is committed or otherwise reconciled.

## Concurrent work snapshot

Checked on 2026-07-27 before this claim:

- neuro-film main chat is active on S4-B and uncommitted W1/W2 research files;
- the standalone colour-match chat is active on D-PCT RAW/HDR/video and media
  execution;
- this worktree is `codex/reference-color-match` at `c03c321...` and had no
  file changes before this coordination record.

## D-PCT contract refresh

Read-only refresh at standalone commit
`413d7134101e50eb6b9663238dc1c2e14fb42c33` confirms that
`src/zhuise/contracts.py` and `schemas/media_frame.schema.json` have no
uncommitted changes while the other task develops a narrow lossless-JPEG DNG
decoder.

The committed D-PCT scene and display rails are not byte/pixel compatible with
the current relative linear-sRGB `WorkingImage` matcher. This branch therefore
claims no RAW/HDR/video decoder or media-frame implementation. Its A3
integration state is contract-mapped but pixel-bridge closed until D-PCT
publishes, or neuro-film approves, a versioned render bridge with explicit
luminance and provenance.

## CFSM dependency boundary

P13 adds only isolated deterministic research code and frozen evaluation
evidence. It does not consume the main task's uncommitted W1/W2/S4 files or the
standalone task's uncommitted DNG/WIC code. P14B may consume a canonicalizer only
after the owning task publishes a stable commit and this branch revalidates
interfaces, rights, leakage and A1/A4/A5 evidence.

## Stable dependency refresh after P13

- Main neuro-film advanced from `c03c321` to `ed1dbb5`. Its committed S4
  result says correct generated conditions improve distribution matching but
  still miss hidden-operator oracle gates. This supports fail-close and does
  not publish a W1 canonicalizer or open P14B integration.
- Standalone D-PCT advanced from `413d713` to `e1f67d3`. It now executes the
  observed local lossless-DNG/profile family to scene-linear ACEScg, while its
  authoritative `real_raw_paths` gate remains failed for lack of a trusted
  renderer, representative vendor RAW coverage and professional reference
  pairs.
- Uncommitted work remains present in both owning tasks and is not consumed.
  A3 stays pixel-bridge closed; P14B stays dependent on a future stable,
  independently identified canonicalizer.

### 2026-07-27 P14B0 intake refresh

- Main neuro-film has a stable W1 implementation commit `77df1b9` under its
  current `ed1dbb5` head. The W1 contract says execution remains queued behind
  repeated S4/U1 adjudication; no repeated W1 report is currently published.
- This worktree pins only the stable W1 source identities and implements a
  read-only report receiver. It imports no external module and reads no
  uncommitted W1 state.
- Standalone D-PCT is now stable through `77e64e1`. Commits `12d5767` and
  `820497e` add complete local compression-7 DNG/profile execution, all-local
  CR2 entropy execution and a pinned LibRaw vendor-unpack audit; `77e64e1`
  independently agrees with the project decoder on 28,682,816
  post-linearization sensor codes across the three observed DNG strata.
- The active 67-file LibRaw agreement expansion remains uncommitted and is not
  consumed. Stable evidence still stops before crop/black normalization,
  demosaic, camera/profile colour, trusted scene rendering or MatchView
  compatibility; seven Nikon HE/HE* files remain unsupported and
  `real_raw_paths=FAIL`.
- Current propagation: P14B0 returns `not-ready` and identity. P14B1 activates
  only after stable repeated W1 evidence; A3 remains separately closed.

### 2026-07-27 portable conformance leaf

- This branch added only its own language-neutral reference-match conformance
  bundle, schemas, verifier and tests in `8b505ca`.
- The bundle consumes neither main-task W1 code/reports nor standalone D-PCT
  decoder/probe code. It validates the existing display-linear SDR algorithm
  boundary after ingress.
- D-PCT stable commit `bd3ff70` completes the 67-file agreement queue:
  67/67 files and 686,122,932 post-linearization sensor codes agree with the
  pinned LibRaw probe. A3 nevertheless stays pixel-bridge closed because this
  comparison stops before the independent scene-render/MatchView boundary.
- Main remains stable at `ed1dbb5`; no repeated W1 report exists, so P14B1
  remains ready but unexecuted.
- The conformance fixture enables future platform-port adjudication; it does
  not claim current Android/iOS/macOS/Windows parity.

### 2026-07-27 stored-recipe replay leaf

- P16 commit `378c846` changes only this branch's reference-match file,
  replay, reporting, CLI and tests plus one new replay-report schema.
- It does not consume main-task W1 state or standalone D-PCT media code.
- The new `--recipe-input` path starts after existing display-linear SDR
  ingress, so it neither overlaps nor weakens D-PCT's ownership of RAW/HDR
  decoding and MatchView conversion.
- Main still has no stable repeated W1 report. Standalone D-PCT stabilized its
  corpus/report result as `bd3ff70`; P16 consumes none of its code because
  recipe replay begins after the separately owned ingress boundary.
- Integration impact: a future app can persist one LookRecipe and reuse it for
  later albums without retaining the reference file; algorithm promotion and
  A3 remain independent gates.

### 2026-07-27 completed D-PCT corpus dependency

- Read-only stable source: D-PCT `bd3ff70`; its worktree is clean and its task
  has completed.
- Evidence accepted: 67/67 local compression-7 DNG files, 686,122,932 sensor
  samples, 14 explicit and 53 identity LinearizationTable paths, three storage
  strata and zero fingerprint disagreements against pinned LibRaw.
- Evidence ceiling: agreement is after linearization but before independent
  crop, black normalization, demosaic, profile colour and display rendering.
  It supplies no trusted scene-to-display/MatchView conversion.
- Propagation: the earlier “corpus queue uncommitted” note is superseded. A3
  remains `CONTRACT MAPPED / PIXEL BRIDGE CLOSED`; no D-PCT source file,
  decoder, probe or report was copied.

### 2026-07-27 complete run transaction leaf

- P17 commit `2f30826` changes only this branch's file adapter, CLI and
  transaction fault-injection tests.
- It promotes no algorithm and consumes no main-task W1/U1 state.
- D-PCT ingress ownership is unchanged: transactionality starts only after
  the existing input has become a supported display-linear SDR WorkingImage.
- Fit and stored-recipe replay now treat N outputs, optional new recipe and
  provenance report as one rollback unit. This removes a product integration
  ambiguity without altering film-colour/effects composition.
- Main U1-B remains active with no stable W1 report, so P14B1 stays
  unexecuted.

### 2026-07-27 run-level composition binding leaf

- P18 implementation commit `b44fc96` changes only this branch's
  reference-match composition binding, schema, exports and tests.
- Composition now follows actual per-output guard decisions from the
  transaction-bound report, rather than inferring delivery from the requested
  research override.
- Only a uniform all-applied run can bind reference colour and optional film
  effects. Uniform fallback binds identity without effects. A mixed run must
  be split or represented per output; it cannot masquerade as one batch plan.
- Report SHA-256, recipe ID, every output hash and every safety action are
  bound. Original references and sources need not be retained after the
  transaction.
- Read-only refresh: main stable HEAD remains `ed1dbb5`; U1-B is active and no
  stable repeated W1 report exists. Standalone D-PCT is idle and clean at
  `bd3ff70`; its completed 67-file sensor-code agreement does not open the
  scene/display bridge.
- No external task was navigated, awakened, modified or used through
  uncommitted state. P14B1 and A3 remain closed independently.

### 2026-07-27 stable-main integration preflight

- Read-only comparison base:
  `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- This branch changes 66 paths; stable main `ed1dbb5` changes 29 paths from the
  same base; the exact path intersection is zero.
- `git merge-tree` against `HEAD` and `ed1dbb5` produced no conflict marker or
  both-modified/both-added conflict record.
- This is a preflight against the current stable commit, not authorization to
  merge and not evidence about main's uncommitted U1-B work. Integration must
  refresh the same checks after main publishes its final stable result.

### 2026-07-27 completed repeated W1 dependency

- Main stable commit `86b484b` publishes the W1 decision and is contained in
  current stable main `c5b7c04`.
- External reports A/B are byte-identical at `9b42e9a8...8dec68`; the report
  software commit is `b72b594...`, a validated descendant of the pinned W1
  implementation.
- This branch's receiver recomputes `paired_upper_bound_only_passes` and freezes
  canonical decision `f8661315...f60e`.
- P14B1 closes: no single-reference confirmation, larger descriptor rescue,
  visual candidate or product integration opens. Default delivery stays
  identity.
- Main's later FilmSet W2F0/Z0 work is paired/fixed-bank mechanism evidence and
  does not reverse the arbitrary uploaded-reference result.
- D-PCT is active in its own repository; no mutable D-PCT state is consumed.

### 2026-07-27 empirical neutral-photo prior isolation

- P19 reads only the stable ignored FiveK `freeze_v1` under main
  `c5b7c04`; it does not modify that checkout or consume its uncommitted
  Goal-Engine/harness files.
- Official FiveK rights are research-only. The aggregate prior is therefore
  an isolated falsification control, never a product/commercial dependency.
- All implementation, tests, configs and reports remain in this worktree.
  Source pixels and ignored reports are not copied into main or D-PCT.
- The empirical route closes on repeated confirmation. It creates no merge
  dependency and does not change main W2F0/FilmSet or D-PCT ownership.
- Main and D-PCT remain read-only concurrent tasks; neither task was
  navigated, awakened, messaged or mutated by P19.

### 2026-07-27 bounded quantile experiment isolation

- P20 is entirely data-free and local to this worktree. It reuses only the
  already committed synthetic manifest and explicit CFSM projection.
- The previously unopened stress split was frozen before execution. No main
  W1/W2 report, FilmSet/FiveK pixel or D-PCT decoder state is consumed.
- The candidate closes below its frozen median-gain threshold and creates no
  external dependency, merge requirement or product promotion.

### 2026-07-27 source-batch quantile isolation

- P21 uses only generated stress palettes and previously unused committed
  operator rows 8--15. Main, D-PCT, FiveK and FilmSet are not read or changed.
- The batch-conditioned contract is explicitly non-reusable and research-only.
- Its negative result adds no cross-task dependency and changes no product or
  film-business boundary.

### 2026-07-27 StatLUT-lite isolation

- P22 uses only programmatically generated palettes/operators and a paper
  specification. It imports no third-party code, weight or real pixel.
- All mapper outputs are bounded parameters consumed by an explicit operator;
  no final-RGB neural generation exists.
- The linear route closes and creates no main/D-PCT dependency or product
  promotion. The feature extractor remains isolated under research.

### 2026-07-27 external-baseline isolation

- P23 evaluates ignored, pinned external checkouts/assets only through a
  committed evaluator and immutable hashes. No third-party source or weight is
  copied into the product tree or Git history.
- CanonCGT is Apache-2.0 and locally reproducible, but both its published
  source-conditioned path and the fixed-reference-pivot adaptation fail the
  frozen 6x6 matrix. No product dependency or promotion opens.
- SA-LUT and Neural Preset are non-commercial; SA-LUT also lacks the actual
  inference checkpoint. NLUT is MIT and its official weight loads, but shared
  batch and representative tuning checks fail. These assets remain ignored
  research controls.
- Read-only task refresh: main `019f4b76...` has published W1/W2 negative
  reference-identifiability evidence and continues its stock-first program;
  D-PCT `019f9f3b...` is actively validating separate RAW/DNG/media ingress.
  P23 neither messages nor mutates either task and does not duplicate D-PCT's
  decoder work.

### 2026-07-27 final delivery preflight

- Stable read-only heads are main `350b687` and D-PCT `d2c7eff`; both
  project worktrees are clean apart from main's own untracked `.codex/`.
- From common base `c03c321`, this branch changes 95 paths and main changes
  81; the exact path intersection is zero.
- `git merge-tree --write-tree` succeeds. A temporary detached latest-main
  worktree accepted the full branch merge and passed 196/196 module plus
  adjacent-ingress tests, then was safely removed.
- Final local CLI fit/replay outputs are hash-identical, and an explicit
  guarded research application succeeds on its positive control.
- No merge, push, main-worktree edit, D-PCT edit or external task message
  occurred. Integration remains a repository-owner action.
