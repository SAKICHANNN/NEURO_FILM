# LOOK2 - Creative Look Approximation product value

Date: 2026-09-05. Owner/integrator: main thread. DRPT L2, Mode A.
Role: active executable product plan, not a stock experiment or release claim.
Parent: ULT; components: U2 explicit operator, U4 evaluation, U7 product.

## Current facts and decision

Starts at `e04118be5`. U7.22A private capsule is retained. U7.2C/RF3.D15
and U4.2A show insufficient v1 distinction/value, not a broken replay engine.
The user's trial found the looks very similar. U7.22B's modified test and
untracked runner are unfinished and must not be staged, removed or run here.
No current v1 asset, recipe, catalog or active desktop process is changed.

The user approved continuing after project adjudication. This authorises a
new creative product version, not rescue/reinterpretation of frozen science.
Do not wait for physical-film photos, email replies or the calibration branch.

## Design hypothesis and alternatives

The candidate uses explicit tonal shape and luminance-zone RGB relationships,
not legacy unpaired film means/stds. Three tentative directions are amber-soft,
cool-clear and copper-rich. These names describe design intent, not measured
film response. No direction has yet passed a photographic or aesthetic gate.

Competing explanations for v1's weak value: (1) nearly shared operator/weak
parameters; (2) generic saturation/contrast already explain any benefit;
(3) changes are visible but do not improve photographic appeal. The first core
test only establishes execution. The photograph comparison must distinguish
all three; numerical diversity alone cannot promote a look.

Reject global v1 strength increases on old confirmation images, new ML/router
capacity, final-RGB generation and further wrappers as substitutes for value.

## First implementation contract

- Private `src/color_engine/creative_look_v2.py`; no package export/integration.
- Parameters in `configs/creative_looks_v2_development.json`; original manual
  art direction, no copied LUT, image fitting or film statistics.
- Input: owned or strided finite HxWx3 float32 **encoded display-sRGB**, [0,1].
  It is not linear RGB, RAW, HDR or colour-unidentified input.
- Per-channel monotone cubic Bezier tone `T(x)` with controls [0,a,b,1],
  0.05 <= a <= b <= 0.95. Compute encoded luma `Y` from the toned RGB using
  fixed weights .2126/.7152/.0722. This is a design coordinate, not radiometry.
- Log gain per channel: `(1-Y)^2 * shadow + Y^2 * highlight + chroma*(T-Y)`.
  Shadow/highlight in [-.5,.5], chroma in [-1,1.5].
- Apply positive gain `g=exp(log_gain)` by `T*g/(1+T*(g-1))`, then convex
  interpolation with input for amount [0,1]. No post-hoc clipping/margins.
- Float64 arithmetic inside bounded tiles, one float32 return; finite/bounded
  check, black/white endpoints, zero-amount owned exact identity, immutable
  input and shape/tile/order consistency. No geometry/resampling/randomness.
- Bounds prove neither aesthetic quality nor absence of quantization artifacts.
  Highlight/chroma stress remains mandatory on photographs before integration.

## Execution and verification

| Step | Status | Scope and acceptance | Commit / rollback |
| --- | --- | --- | --- |
| Direction and candidate contract | DONE | Plan/config frozen at `32c552180`; active authority corrected, old evidence unchanged | Scoped plan commit; revert only owned files |
| Private core | DONE | 44 synthetic tests pass; with v1 non-CLI behavior, 57 pass / 2 CLI tests deselected. No photographic value claim | Core/test commit; no product imports to undo |
| Photo development | IN_PROGRESS | Lock `creative_looks_v2_photo_development_v1.json` before pixels; first 8 hash-ranked rows from parent RF3.D0's 16, never the extra manifest row | Separate source/run commit; outputs under P-backed outputs/ |
| Confirmation | NOT_STARTED | Freeze parameters, metrics, rubric and source-disjoint confirmation before first confirmatory render | Immutable report/observations; no tuning or revote |
| Product integration | NOT_STARTED | Only surviving looks; catalog/recipe/preview/export regressions, full-resolution severe checks and ordinary-photo resource measurements | Separate opt-in integration commit |

## Photo evaluation readiness (not yet ready)

The initial labelled development uses eight existing CC0 digital display
derivatives, no raw decode/download. The config locks their parent manifest and
hash-ranked selection. Eight other historically consumed rows remain unaccessed
within LOOK2 until a later confirmation contract. This is candidate-specific
withholding, not globally fresh or author/scene-independent evidence (groups
are unknown). Source/size/hash validation precedes every development pixel read.
The first contact sheets contain identity, three current v1 outputs, three new
full-amount directions, saturation 1.2 and contrast 1.15 controls at <=512px.
The basic sliders are diagnostic controls, not yet strength-matched primaries.
Readout is per-image byte identity, mean RGB change and new quantized endpoints;
there is no automatic aesthetic/promotion gate in this development run. No
original-resolution severe, preference, calibration or performance claim opens.

Use a small existing rights-cleared digital corpus, not physical-film targets.
Record source identities and prior consumption; historically viewed images may
be development/regression only, not described as fresh blind confirmation.
The user's trial image is a regression cue, never a sole design target.
First lock the actual source/scene groups, rights and development/confirmation
partition. No source inventory or numerical photographic gate is fabricated here.

Controls are identity, current v1 and simple saturation/contrast adjustments.
Match gross change strength where feasible to test whether the new relationships
add value beyond basic sliders. Use labelled development, anonymous held-out
comparisons, overview plus original-resolution crops (skin, sky/gradients,
text, highlights, saturated objects). Report each look and image separately.

Promotion order: severe veto -> stable visible distinction -> style/appeal over
identity and controls -> workflow and common 12-24MP resource acceptability.
Autonomous review is not human/population preference or stock identification.
Exact thresholds and rubric must be committed in the photo contract before the
held-out run, not inferred from development results or selected after a failure.

Pass: integrate only passing looks, even if fewer than three. Weak/ambiguous:
keep experimental; a separately declared diagnostic may resolve a specific
uncertainty, never rename it a pass. Valid failure: close the frozen candidate.
Invalid test/report: retain the attempt and fix its contract/implementation
before scientific interpretation; do not claim it disproves appearance.

## Boundaries and propagation

No calibration, physical-film, learned operator, novel-algorithm, human preference,
public release or universal safety claim. No outreach, purchase, cloud, installs,
new data downloads or protected holdout reads in the core step. No imagegen.
Dependencies are existing NumPy and existing product test/runtime tools.
Potential risks: zonal casts, over-chroma, input-colour confusion, float32/8-bit
boundary rounding and aesthetically weak output despite mathematical bounds.

Upward: Goal/AGENTS/README/tracker now require visible value. Downward: photo
steps remain gated, not silently opened by synthetic tests. Sideways: v1,
BW severe veto, calibration and U7.22B remain untouched. Stock plan is deferred;
historical registry/evidence are not rewritten. Only one primary writer operates.
No product API migration until confirmation; rollback is scoped commit revert,
not a reset of the shared checkout. Keep one long-lived Goal across these steps.

## Core verification, 2026-09-05

### Development feedback and stronger directions

First reserved assessment is complete: see
`docs/evidence/LOOK2_RESERVED_ASSESSMENT_01_REVIEW.md`. Both candidates remain
unpromoted: 0 established clear preference wins over identity and best supplied
control in the labelled overview review, below frozen 6/8. All eight roles are
now consumed within LOOK2 and must not be called unopened. No further tuning
on this assessment cohort. Detail/portrait/workflow safety remains unproven.
Candidate-fitted oracle parity is a complexity question, not sufficient evidence
against the usefulness of a selectable simple preset; separate these questions
in future scopes without rewriting this negative decision.

The first reserved assessment is now prospectively bound by
`configs/creative_looks_v2_assessment_v1.json`: exact refined config/core hashes,
same eight previously reserved IDs, 1024-side derivatives, two unchanged grades.
Before any reserved body access, commit the lock and runner. Any confirmed new
severe artifact stops a candidate; visible distinction and autonomous labelled
preference must each reach 6/8, with ties not wins and preference compared with
identity and sensible controls. Candidate-fitted affine and closest 21x21
saturation/contrast controls supplement the original sliders/v1. No parameter
tuning on these eight after viewing. This is not blinded/human/global-fresh
evidence; it cannot establish skin/general product quality by itself.

Development-04b narrows to two explicitly creative grades in
`configs/creative_looks_v2_refined_development.json`: cool matte daylight and
warm daylight. The deep grade is not carried forward. Changes lift lower tones
and use near-neutral white endpoints, rather than repeating the strong yellow
paper-white tint. All eight labelled overview sheets were inspected: orchid
detail and rail-yard visibility improve versus the bold settings; warm/cool
casts remain intentional and clearly visible in neutral interiors. This is
author judgement, not blind preference. Neither is approved for integration.

Report `outputs/creative_look_v2_development/development-04b-refined/report.json`
SHA256 `1291ebd26f149b1062a933ea272d0ea725c6a0a49cc90f7e609c1f3c2803f365`;
8 development inputs x 8 arms, no reserved/RAW/network reads. Both candidates
have zero newly quantized endpoint components on all eight previews, which is
not a general severe-artifact guarantee. Corrected verification is 68 PASS,
2 CLI tests deselected; monotone ramp and near-neutral-white checks included.

Transparent development lineage: first `development-04-refined` ran despite a
failed new low-grey channel-lift assertion because the shell commands were not
success-chained. Cool shadow red was 0.08683 for 0.10010 input. That report is
retained, not an accepted verification. Adjusted only cool shadow-red log gain
from -0.5 to -0.2, preserved the assertion, then success-gated tests before 04b.
No old scientific gate or product parameter was changed.

Next stop on development tuning: freeze these two candidates for assessment,
with sensible closest-basic controls as well as magnitude-matched diagnostics.
The current eight development scenes lack meaningful skin coverage; a positive
result on them cannot authorize a general portrait/photo product. Prospectively
lock appropriate assessment roles before any additional candidate image reads.

Development-03-detail keeps the bold parameters unchanged and reads only the
same eight development display derivatives, now at their available maximum
1600 side (not original RAW/full sensor resolution). It adds 1:1 centre crops,
121-point mean-RGB-change-matched saturation/contrast controls and a fitted
per-channel affine approximation. These are candidate-fitted diagnostics, not
independent truth, inference models, globally optimal controls or blind tests.
Central crops are not exhaustive worst-case inspection. Matching mean change
can select an aesthetically destructive contrast setting; defeating such a
control cannot establish preference. Subsequent value testing must also offer
a sensible/basic best-approximation control, not only magnitude matching.

Completed report `outputs/creative_look_v2_development/development-03-detail/report.json`
SHA256 `3479085b6dc4caa3cd812982cc3189c55094030da0e9022cd91b9303314bf814`;
177 files / 689,794,780 logical bytes retained in project P-backed outputs.
Avoid repeating large grids without a specific question. Mean per-image RGB8
residual versus fitted affine is 3.0845 cyan / 6.0180 warm / 5.9207 deep;
these are numerical approximation errors, not perception or preference scores.
Execution was development working-file-bound, not a formal clean-HEAD replay.

Labelled crop review: cyan matte has similar broad appearance to affine on
the hayfield; warm print lifts flower detail but casts neutral building walls
yellow; deep chrome reduces visible backlit flower and rail-yard shadow detail.
These observations motivate refining tonal allocation rather than escalating
strength. No confirmation/promotion is opened. A simple usable grade is an
acceptable outcome; nonlinearity itself is not the product objective.

The owner rejected development-01 as visually too subtle. This is direct
product-value evidence against those settings, regardless of implementation
tests. They are retained unchanged, not promoted. Development-02-bold uses the
same eight development photos, with explicit creative black/white endpoints
and three separately named grades (cyan matte, sunbleached print, deep chrome).
This is ordinary declared creative development, not a rescue of a frozen
scientific experiment. No confirmation images were read.

Eight photos / nine arms were rendered at maximum side 512. Report:
`outputs/creative_look_v2_development/development-02-bold/report.json`, SHA256
`c71b2329523997a263493af1665ccd0faa42635b30be016d4a1f80dc0a8aaab5`.
The report binds working-file hashes; this was a precommit development run,
not a clean-committed-head formal evaluation. Autonomous labelled overview
review of hayfield, orchids and courtyard shows clearly different grades,
but cyan changes purple petals to blue, warm print gives broad yellow casts,
and deep chrome darkens the backlit flower subject. No photographic safety or
preference pass is claimed. Warm print introduces quantized endpoints in all
eight rows; cyan/deep introduce none in these previews. Endpoint counts are
diagnostics, not counts of severe artifacts.

Next: inspect large development images/crops and distinguish intentional colour
design from unwanted casts and shadow loss. Compare against strength-matched
basic controls before freezing candidates and reading confirmation roles.
Do not advance all three merely because they differ. Product integration stays
closed. Core compatibility, new endpoint scalar oracle/tiles/validation and
behavioral adjacent tests: 66 PASS, 2 CLI tests deselected. No stock inference,
new data, GUI change or full-resolution performance claim.

`python -B -m pytest -q -p no:cacheprovider tests/test_creative_look_v2.py
tests/test_u7_2c_three_stock_look_amount.py -k 'not cli'`: 57 passed,
2 deselected (CLI/full media not involved in this private core change).
Independent scalar formula outputs, pointwise tiled/reverse outputs and owned
zero-amount outputs are exact; development ramp/cube/invalid tests pass.
Ruff format/check passes. These are synthetic/behavioral tests, not a
photographic safety, preference, performance or product-promotion result.
