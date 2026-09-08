# K-MCFM: fresh-chat handoff, 2026-09-08

## Paste this into the new task

Continue the existing K-MCFM project in `C:\Users\hhvrf\Documents\neuro_film`. First read this file and `docs/ops/AI_RECOVERY_20260907.md`, then inspect current Git state and the relevant evidence. Do not start a new model run just because this is a new conversation. Preserve existing work and negative results. Treat this as a fresh, critical continuation, not a requirement to execute every historical tracker node.

The user wants AI to learn and determine the important film-style transform, producing visibly convincing, appealing film-inspired photographs without damaging people, text, objects or geometry. The user cannot supply paired digital/film data or physical captures. Hand-authored curves and palettes are comparison baselines, not a substitute for the AI objective. Learned curves/LUTs/bounded operators are valid; directly generating final RGB is outside the current content-preserving branch. Research value is conditional on real evidence, not a prerequisite for a useful product or a promised paper.

The immediate concern is: recent results look weak or worse than early results. Recover and identify actual earlier best outputs and their generating code/checkpoints; compare them on common inputs with originals, simple adjustments and current learned candidates. Do not assume an older attractive image was neural, or that a newer model is better. Do not simply restart a closed parameter grid. Distinguish recoverable implementation errors, inadequate supervision, unstable operators and genuinely weak visual results.

Keep one genuinely long-term objective if using Goal; do not ask the user to repeatedly delete/recreate it for each experiment. A Goal is not needed to perform this initial handoff audit. The old task's Goal was paused when checked on 2026-09-08; this document does not transfer or resume it automatically. Confirm task-local state before any creation, and do not run two competing mainline loops.

## Current truth and instruction reconciliation

- The latest direct user direction is AI-driven film-inspired appearance. Older deterministic Look Approximation documentation describes the existing product and historical branch, not completion of the AI request.
- The current default remains safe_lab/safe-rich with optional effects. Recent NLUT/VCG experiments have NOT been promoted into it.
- Film-inspired / Look Approximation is the honest label. Specific stock calibration is unproven and is not a product dependency.
- User feedback repeatedly says outdoor results are too subtle, insufficiently film-like, and sometimes worse than early algorithms. Numerical accuracy, hashes and passing unit tests cannot answer that complaint.
- Severe photographic artifacts block promotion, but similarity to the original is not the aesthetic objective. Strong intentional style is allowed.
- Old source-identity, licensing and held-out evidence remain valid. Do not reopen failed scientific experiments by renaming them. Also do not apply calibrated-stock data requirements indiscriminately to a separately authorized creative AI task.
- Plans may be criticized and corrected prospectively; never rewrite an old result to manufacture success.

## Verified repository snapshot

Base HEAD before this handoff: `5c1ed3a04` (`research(ai): preserve VCG preprocessing ablation and quality limits`). The handoff itself will be a separate documentation-only commit.

Existing changes observed; NOT owned or modified by this handoff:

```text
 M scripts/compare_creative_looks_v2.py
 M tests/test_selective_print_development.py
 M tests/test_u7_22b_100mp_product_render_resource.py
?? configs/creative_print_development_v6.json
?? scripts/audit_u7_22b_100mp_product_render_resource.py
?? .codex/
?? tmp/
```

Never stage those together with unrelated work. Refresh status before writing. No process inventory was taken for this handoff: verify live processes before running or cleaning anything. Do not infer that every other task is idle.

## What has actually been tried

| Family | Execution and bounded conclusion |
|---|---|
| SD1.5 IP2P | Trained on pseudo-pairs; painterly smearing/content damage. Retired default. |
| SDXL film LoRA + SDEdit | Trained and rendered; weak strength nearly unchanged, useful strength rewrote details. |
| SDXL full UNet IP2P | Memory-limited smoke attempts, not a successful complete model. |
| Lab statistics / safe-rich / explicit effects | Usable deterministic infrastructure; insufficient appearance/diversity, not AI achievement. |
| AO6 / bounded global and adaptive operators | Actual research comparisons; fixed AO6 beat a triangular adaptive challenger 21/30 vs 9/30 choices. Digital-retouch proxy, not calibrated film. |
| Early neural LUT distillation | Actual training; one recovered s800 checkpoint collapsed all tested stock conditions to the same output. Do not generalize this to all checkpoints. |
| FilmCase / Roll2Film | Retrieval, nuisance and grouped tests; no established robust stock/physical-roll advantage. Many later mode branches were never admitted. |
| CLIP-guided LUT fitting | Loss improved but false colors/edges; negative. |
| Real-photo RGB/deep-feature supervision | Actual small pilots; weak style or unstable false colors. Bounded triangular successor safer but weak. |
| Grain learning preflight | Scene/JPEG contamination found; not a completed trained grain model. |
| FilmSet paired digital-recipe learning | Global/conditional LUTs trained; stronger fitting did not establish photographic appeal. Not physical-film truth. |
| ClassNeg | Frozen 17-photo assessment: 6/17 wins vs original and old baseline, no promotion. |
| Full pretrained NLUT adaptation | Three actual pairs; adapted results had severe issues, no promotion. Distinct from using its feature encoder only. |
| ICCV2025 Video Color Grading (VCG) | Pretrained AI generates LUT, not final RGB. Three-pair development: one attractive alternative, one severe chroma-noise failure, no promotion. |
| VCG no-precorrection ablation | Snow defect improved; fruit/flowers over-chromatic, 0/3 clear preference over original/simple. Same-pair option family ended. |

Hundreds of RAW/HDR/format/replay/source-audit nodes are NOT hundreds of film-style algorithms. Paper/source audits without runnable assets are not model experiments. Read original evidence before broader conclusions.

## Recent evidence and recovery entry points

- `docs/ops/AI_RECOVERY_20260907.md`: detailed model loading, supervision, recovered checkpoints, exact outputs and negative history. Entries contain both earlier plans and later terminal updates; do not execute an outdated "next" paragraph without reading its closure.
- `docs/evidence/AI_VCG_REFERENCE_DEVELOPMENT_20260907.json`
- `docs/evidence/AI_VCG_STAGE_GAIN_AND_NCC_DIAGNOSTIC_20260907.json`
- `docs/evidence/AI_NLUT_REFERENCE_DEVELOPMENT_20260907.json`
- `docs/evidence/AI_CLASSNEG_PHOTOGRAPHIC_REVIEW_20260907.json`
- `docs/evidence/AI_CLASSNEG_SUPERVISION_DIAGNOSTIC_20260907.json`
- `docs/CURRENT_STATUS_2026-05-27.md`: historical diffusion pivot, not current runtime authority.
- `docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md`: reusable deterministic infrastructure.
- `docs/ULTIMATE_EXECUTION_TRACKER.md` and `docs/drpt/AGENT_LOG.md`: historical indices, not a mandate to extend every branch.

Latest saved VCG outputs are under `outputs/ai_vcg_reference_development_v3` and `outputs/ai_vcg_ncc_ablation_v1`; check manifests before loading. The original VCG run took about80s /8.23GB peak CUDA allocation; no-precorrection about51s. These are small development measurements, not independent product validation. The three pairs are consumed, not fresh holdout.

## Recommended first continuation, not a frozen new experiment

1. Read the handoff and refresh Git, Goal, environment and relevant artifact availability. Do not dump the entire historical tracker into context.
2. Make a small inventory of recoverable earlier preferred outputs (historical anchors include 01/09/53/55/56), identifying exact input, algorithm, model/recipe and output. Some anchors have known severe artifacts; retain that fact. Do not simply declare all anchors winners.
3. Reproduce the strongest recoverable distinct families on common development inputs. Show whole images and full-detail crops; separate film character, appeal, defects and content retention. No new user votes are a prerequisite; autonomous judgments must be labeled as such.
4. Decide what is actually worth retaining. Only then propose one bounded learned successor with explicit supervision and an independent assessment plan. A new name, larger model or reduced loss is not sufficient motivation.
5. Product integration follows credible visual advantage and full-resolution safety/replay checks, not vice versa. Do not return to wrapper/format/100MP expansion as a substitute.

## Operations and permission boundaries

- Use project `.venv` first; target hardware RTX5070Ti Laptop12GB and M5 32GB, verify live availability. No dependency install or GPU job was started for this handoff.
- Persistent datasets, checkpoints and outputs use repo-relative junctions into project-owned P storage. If unavailable use only project-namespaced D fallback under the existing policy, not disk roots or C data copies. Verify actual junctions/free space rather than old numbers.
- Preserve v1, learned weights, negative evidence and foreign changes. No further cleanup is authorized by this migration request.
- Do not inherit cloud resource ownership merely because an old document mentions a budget. A new task must verify authorization and exact ownership; no cloud, email, purchase, public release or unknown-resource mutation implied.
- Do not resume the other two chats automatically. Coordinate only if current work requires it and preserve their file ownership.
- Existing related task IDs for context lookup only: producer `019f9f3b-d0c2-7f21-b486-1dd902148739`; consumer `019f9f37-91d9-7b11-b135-ad62bcb32214`. Original main task `019f4b76-e70a-75c0-b7ea-b473ab38c200`.

## Handoff audit

Prepared using dev-research-reliability. Checked live Goal (accurate AI objective, paused), Git status/log and current VCG terminal recovery text. Documentation-only; no model, image, configuration, default, historical evidence or foreign diff changed. This is a continuation map, not a new scientific result. Fresh task/model creation is not performed by this document; the user's spelling "astia-native" is retained as an unconfirmed model/UI label, not an API model identifier.
