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
| Direction and candidate contract | IN_PROGRESS | This plan + config; active authority corrected, old evidence unchanged | Scoped plan commit; revert only owned files |
| Private core | NOT_STARTED | Independent scalar oracle, 33-cube/ramp, invalid controls, endpoints, owned/strided/tiles/reverse exact; v1 behavioral regressions | Core/test commit; no product imports to undo |
| Photo development | NOT_STARTED | Lock eligible digital source identities/roles and controls before reading pixels; bounded comparison sheets and development measurements | Separate source/run commit; outputs under P-backed outputs/ |
| Confirmation | NOT_STARTED | Freeze parameters, metrics, rubric and source-disjoint confirmation before first confirmatory render | Immutable report/observations; no tuning or revote |
| Product integration | NOT_STARTED | Only surviving looks; catalog/recipe/preview/export regressions, full-resolution severe checks and ordinary-photo resource measurements | Separate opt-in integration commit |

## Photo evaluation readiness (not yet ready)

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
