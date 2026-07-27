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
standalone task's uncommitted DNG/WIC code. P14 may consume a canonicalizer only
after the owning task publishes a stable commit and this branch revalidates
interfaces, rights, leakage and A1/A4/A5 evidence.

## Stable dependency refresh after P13

- Main neuro-film advanced from `c03c321` to `ed1dbb5`. Its committed S4
  result says correct generated conditions improve distribution matching but
  still miss hidden-operator oracle gates. This supports fail-close and does
  not publish a W1 canonicalizer or open P14 integration.
- Standalone D-PCT advanced from `413d713` to `e1f67d3`. It now executes the
  observed local lossless-DNG/profile family to scene-linear ACEScg, while its
  authoritative `real_raw_paths` gate remains failed for lack of a trusted
  renderer, representative vendor RAW coverage and professional reference
  pairs.
- Uncommitted work remains present in both owning tasks and is not consumed.
  A3 stays pixel-bridge closed; P14 stays dependent on a future stable,
  independently identified canonicalizer.
