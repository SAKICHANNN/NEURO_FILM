# Early model recovery: descriptive smoke, not promotion

## Actual-photo supervision intake, 2026-09-07

### Photo-supervised pilot result

Executed committed implementation `867988a13`: 150 fresh Adam updates per arm,
no checkpoint reuse, output `outputs/ai_photo_distribution_pilot_v1`.
Report SHA256 `1e4cd6d8e9364260e051ab867ca7a73db01cef8ace4e61c9d66e409f805a52ee`.
Both arms started distribution loss .00714744. At step100 LUT distribution
loss .00063522 vs simple .00120565; this proves fitting, not visual superiority.
All three transfer rows' originals/LUT/simple were visually inspected at768px.
No obvious prior CLIP-style green/magenta contour corruption at this resolution.
However LUT mainly lightens/desaturates the building, shifts the foliage/wall,
and cools indoor light. It does not establish convincing film character or a
clear photographic advantage over the simple learned control. NO PROMOTION;
independent/full-resolution validation not opened. This exact fixed run stops,
without strength/loss/palette rescue. Missing texture/context in a pooled RGB
distribution is a plausible limitation, not a demonstrated causal conclusion.
Next method must learn more than global colour-distribution matching.

Implementation lint initially flagged an unused noqa, but the shell proceeded
to commit/run. Removed only that comment afterwards; current Ruff/compile pass.
The execution commit is preserved and no claim of pre-run lint pass is made.

Photo-distribution pilot frozen before fitting: 19 references inspected as a
contact sheet, natural travel scan appearances, repeated bridges/architecture
and lighting confounds explicitly retained as a single development group.
QA dHash minimum within=16, against historical stored signatures=20; these are
heuristics, not proof of independence, and historical pixels remain unread.
New `ai_photo_distribution_pilot_v1.json` fixes 150-step equal-image sampled
RGB sliced-quantile distribution learning, 64 seeded projections, versus a
learned simple logit-affine control and identity. Shared 9-cube residual LUT
and simple arm use the prior explicit bounded executor, no CLIP objective,
no hand palette. This tests actual-photo supervision, not methodological novelty.
Spatial content cannot be invented by this pointwise executor, but false colour
and contour artifacts remain possible and require visual veto. Frozen source
development rows0..5 and transfer rows6..8 are reused development, never a final
independent result. Run once, preserve parameters/losses/images and do not rescue
the fixed run by changing strengths, palette, losses or references afterwards.

AI-first child after the failed CLIP-only pilot: acquire the 19 explicitly
CC-BY-4.0 photographs on Nick Rudzicz's Italy 2024 page, with a 64MiB total cap.
The page states Nikon FM-2 / Portra400. This is author/scan appearance supervision,
not identified Portra response. All 19 are development references in one group;
no random within-series split is an independent test. Old A3U 2023 references
and sealed assessment remain untouched. Exact new URL search of configs,
docs/data and docs/evidence found no matches before this lock.

Sequence: freeze config -> bounded source/rights snapshot + image hashes ->
check duplicates and view references -> separately specify photo-supervised
learning and controls before fitting. Abort on rights/URL drift, decode failure
or budget excess; retain explicit partial state, never substitute another image.
No training or production change is authorized by acquisition success alone.
Candidate learning must beat identity/simple controls visually after artifact
review; photo-distribution loss is not a photographic-quality result.

Owner: main single-agent. New acquisition script/config and this recovery note
only, plus minimal agent log; foreign dirty files are excluded. Revert the scoped
code commit for rollback; preserve source manifests and historical evidence.
Data layout: source.html, manifest.json and ID.jpg under the configured P-backed
repo-relative directory; manifest carries bytes/SHA256/dimensions/URL/role and
license attribution. Verify bounded reads, create-only destination and JPEG
integrity before recording completion. Intake is DONE; learning and
independent validation are NOT_STARTED.

Result: 19 JPEGs / 11,797,708 bytes. Manifest SHA256:
`7a7cf18824151f656cdd94e82abcdd4e4186c2167f273dbd32d27bd61da8ff06`.
JPEG verification and historical nicknick URL/byte duplicate checks pass.
Perceptual overlap remains unchecked, sealed pixels unread. Image04 visually
inspected only: natural building/sky scene with visible scan texture; not full
source QA or learned-candidate evidence. Parser/hash tests 2 PASS; Ruff/compile
PASS. Initial inline shell smoke had a quoting SyntaxError before execution;
replaced with pytest tests. No training or production change.

The current user asks for AI-learned film character without supplied paired
captures. Manual creative-v6 changes remain interrupted and untouched.

## Live evidence

- `scripts/train_neural_lut.py` constructs supervised targets by calling
  `pipeline_color_baseline.style_transfer`; checkpoint
  `outputs/neural_lut/challenge_color6_s800_b12_init/model.pt` exists. Its
  800-step training loss is teacher approximation, not film authenticity.
- `outputs/neural_film_lut_v2/scheme_a_distilled_v1/metrics.json` records
  20 images and teacher `outputs/eval/color_engine_challenge/film_response_v1_s1p0`.
  `fit_distilled_film_lut.py` fits curves/residual LUT and context gain from that
  teacher. This is data fitting, but not independently learned real-film style.
- The numpy model SHA256 is
  `1be4dce9b99bb9565997e22b6265c87fdbfe11f85ac1d6155f7359f45e2059a0`.

## Executed recovery

Unchanged `scripts/evaluate_distilled_film_lut.py`, existing `.venv`, first three
historical manifest rows, Ektar/Portra/Velvia configuration IDs, strength 1.0,
max-side 768, historical output margin 4. Output directory was checked absent
before execution: `outputs/ai_recovery_20260907` (47 files, 20,561,301 bytes).
No training, new downloads, assessment-set reads, model changes or product changes.
Summary SHA256 `36b3bce76a6ac144c58ed4ba863b5201b385e25c4a45987a26585595f63ab809`.

Ektar and Velvia contact sheets visually inspected: snow scene, fruit/chart and
flowers. Changes mainly appear as contrast/chroma differences. This inspection
does not establish compelling film character, full-size safety or user appeal.
Portra executed but was not separately visually adjudicated. Same historical
training-era images are intentionally reused for recovery, never held-out claims.
Legacy clipping margin/SSIM are reported by the old runner, not current promotion
criteria. This run does not establish which early algorithm was the user's best.

## Next useful action

### First new semantic-supervision learning pilot: rejected

Frozen implementation/config commit `d09b953ce`, script
`scripts/run_ai_clip_lut_pilot.py`. Official local CLIP ViT-B/16 SHA256 matches
the public model URL `5806e77c...df416f`; MIT software license and research
limitations read from https://github.com/openai/CLIP. No downloads or paid run.
Model is frozen and backpropagates image-text loss into a shared17-cube field;
the competing simple control learns six diagonal-color parameters. Final RGB is
an explicit logit displacement, not a generated image. Both60 steps, seed fixed.
Inputs are nine existing CC0 rawpixls derivatives: six training, three transfer
development rows. These are not new independent assessment or real-film targets.

Identity, finite-bound and gradient smoke passed before learning. Two local CUDA
fits and27 renders completed;30 artifacts /18,159,362 bytes. Report SHA256
`c4ae5e50ed58d33ab7e75506eeeca764aedaae34ca1b0df7bfccc3f2cc385292`
under `outputs/ai_clip_lut_pilot_v1`. Model/evaluator unchanged; no prompt sweep.

Visual inspection of all three LUT transfer results rejects the candidate:
station facade, wall/shadows and interior show conspicuous false-color speckle,
contours and patches. Original/station simple-control inspected too; simple
control has a pervasive purple/red cast. Falling CLIP loss does not establish
film appeal. Bounded output did not prevent severe color artifacts. No promoted
candidate, no full-resolution work, no parameter/prompt/step rescue. Remaining
simple transfer rows need not be adjudicated to reject the LUT candidate.

This is a concrete learned baseline failure, not evidence that all AI methods
fail. Next requires a photographic style observation or stronger justified
supervision and structure, not more text-score maximization. Official CLIP model
card explicitly does not establish general deployment readiness.

### Current method/source triage (2026-09-07, no new payload)

| Primary source | Decision for next learner |
| --- | --- |
| https://github.com/ns144/3D-LUT | Direct precedent: CNN-predicted LUT with unpaired GAN/CycleGAN/StarGAN. GitHub API reports no license; not adopted as code/data/weights. Architecture alone is not our novelty. |
| https://github.com/EtonMu/deep-analog | Latest 2026 reference-LUT work, but existing registry records no usable permission/checkpoints; procedural supervision is not physical-film evidence. No adoption. |
| https://github.com/Ry3nG/SA-LUT | Do not reopen R2Q0/BL12: existing source/runtime and resource failures remain authoritative. |
| https://openaccess.thecvf.com/content/WACV2025/papers/Li_D-LUT_Photorealistic_Style_Transfer_via_Diffusion_Process_WACV_2025_paper.pdf | Score learning with explicit final LUT is relevant, but R2R1 published-trajectory orientation/range failures remain closed. Not a ready checkpoint. |
| https://github.com/jonathangranskog/lut_generation | MIT engineering precedent for text-guided LUT optimization, not established film supervision or a paper-quality result. Separate base-model rights and reward-hacking risks; not selected merely because runnable. |

Exa discovery followed by official repository reads and local closed-route
deduplication. No image/model/data downloads or external messages. This is a
decision matrix, not an exhaustive literature survey. A new learner needs legal
style observations or a justified learned supervision signal, not another
hand-authored teacher. Generic look claims do not authorize reopening sealed
physical-stock evaluations. No candidate promoted or new training yet.

### Three-checkpoint diagnostic closure

`scripts/audit_early_lut_conditioning.py` is a read-only CPU probe on one fixed
64x64 synthetic RGB ramp. No training or photographs are used by the diagnostic.
Results: smoke style-LUT span .0090916; s300 and s800 span exactly0. Smoke logits
range -1.65..2.84; s300 -20.21..23.24; s800 -73.12..154.51. Softmax saturation is
observed; causal attribution to learning rate, target diversity or preprocessing
is not established by comparing distinct runs.

The training script minimizes only L1 against safe-rich targets, reports loss on
the same examples and has no condition-effect validation. Training uses padded
128px squares while the old evaluation consumes unpadded 1600px images: another
known mismatch to avoid in new experiments, not a proven sole collapse cause.

Unchanged smoke checkpoint inference also completed on the same three inputs.
Its metrics record only Portra/Velvia, six examples and60 steps. Ektar was
additionally executed but is an **untrained ID**, excluded from quality claims.
Velvia sheet inspected: still visually near-input. Summary SHA256
`fc58c54d5b01fd67bf36a638f98dc69d0f87f824d9abc5b0adcd02546ed404c2`, under
`outputs/ai_recovery_20260907/neural_smoke`. No safety/full-size promotion.

This closes the bounded old-LUT recovery pass: do not expand old-weight tests or
repair pseudo-teacher fitting as the new film-learning method. Next audit legal
non-paired style supervision and official learned operator implementations;
retain these recovered outputs as bland learned baselines.

### Neural s800 recovery and conditioning failure

Executed unchanged `evaluate_neural_lut.py` with
`outputs/neural_lut/challenge_color6_s800_b12_init/model.pt`, first same three
historical inputs at their 1600px source size, Ektar/Portra/Velvia IDs and legacy
output margin4. CUDA inference completed nine outputs; 28 artifacts / 31,297,190
bytes under `outputs/ai_recovery_20260907/neural_s800`. This differs in resolution
from the preceding 768px numpy recovery; do not compare their metrics directly.

Checkpoint SHA256:
`83c21d2121ab9bb2a3e99ffa49653a947ae16c1ccddb0222a7a152f098919899`.
Summary SHA256:
`7a9e69c9a71605fe7c7de9c112ca8c40adefafe22a5afb360f40b96f707d0dd9`.
For each input, all three style output PNG hashes are identical:

- 01: `77b815776025e5311df93c42a182661749e36d1c404f97a6e3d21fc786c40c83`
- 02: `703dad793922e41b7476f0b52bf464b9d8704280b68a44b54e0aed8e95c64366`
- 03: `a47ad16647b119fdfab1bdfffb2b00edb22d72a534f628b719bab537c47cedfd`

Read-only CPU probe on input01 through the exact loaded encoder: every one of
the eight checkpoint style IDs returns one-hot weight at zero-based basis7
of12, all other weights zero; maximum cross-style LUT difference is exactly0.
Thus this checkpoint exhibits collapsed conditioning on this probe. This is not
proof of the training cause, nor a universal statement about all images/models.
Ektar sheet inspected: very small visual change. SSIM .99902-.99968 is not a
style-success measure. No promotion, retraining or checkpoint modification.

The documented owner anchors 01/09/53/55/56 are recovered in
`render_filmcase_anchor_set.py` as deterministic pipeline recipes, explicitly
anchor-inspired rather than exact historical replay. They do not identify a
preferred AI checkpoint. Keep that distinction instead of retroactively labeling
those preferences as neural-model success.

Recover neural checkpoint inference and trace the genuinely preferred early
outputs separately. Compare on the same development photographs before deciding
what is reusable. Keep diffusion-era output evidence, but do not restart final-RGB
generation. A new learned route must change the supervision, not merely distill
another hand-designed palette. Independent assessment remains sealed.
