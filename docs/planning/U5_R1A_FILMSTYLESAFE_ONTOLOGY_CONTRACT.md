# U5.R1A FilmStyleSafe ontology and study-contract freeze

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1A`

Status: frozen before machine-readable schema implementation

## Purpose and claim boundary

U5.R1A supplies a reusable evaluation contract for strong explicit colour
operators. It separates visible style, global exposure/tone choices, local
chromatic artifacts and structural corruption so that a strong look is not
rejected merely for differing from the input, while a glitch is not excused
merely for looking dramatic.

This is supporting evaluation and product QA. It is not a primary-paper claim,
a learned label generator, stock-authenticity evidence, permission to train an
artifact model or evidence that the future human study is powered. Local
autonomous vision reviews may populate development records only and must be
labelled secondary autonomous evidence.

## Ontology

The primary colour-operator severity labels are `safe`, `minor`, `severe` and
`uncertain`. A `severe` event must be introduced or materially amplified by the
transform, objectionable at the declared viewing condition, and belong to one
of these frozen categories:

1. implausible cast over a large smooth region;
2. skin or neutral-region contamination;
3. neon chroma island or high-chroma highlight speckle;
4. banding, posterization or hard colour shelf;
5. clipping/gamut component with a visible hard boundary;
6. local hue jump, grid seam or halo;
7. chromatic instability under a benign exposure/WB/encoding perturbation.

Global exposure-policy instability, strong tone, contrast, saturation, white
balance or palette shift, blandness and target-look mismatch are recorded as
separate quality/style diagnostics. They are not automatically severe. They
become severe only when the observable result satisfies a frozen severe
category, such as a visible hard boundary or destructive chromatic
instability. This preserves RF2.C0's lesson without retroactively declaring a
global aesthetic choice to be a glitch.

Face/limb, text/logo, object/geometry and texture corruption remain a separate
structural ontology used only when a generative RGB control is present. They
cannot be used to imply that a global colour operator is safe in categories it
cannot structurally alter.

Every local event requires a mask or polygon and a 100% crop. A global event
uses an explicit whole-frame region. Grain, halation, bloom and sharpening are
disabled in the colour study; intentional bounded optical effects are retained
as hard negatives when validating future spatial metrics.

## Legitimate-local hard negatives

Metric and reviewer qualification must include local dodge/burn, skin
protection, sky densification, mixed-illuminant correction, intentional local
colour accents and bounded bloom/halation. These examples are not presumed
safe: they exist so a detector cannot equate every local input/output colour
difference with an artifact.

## Scene-level label rule

The primary confidence unit is an independent parent scene, never a crop,
candidate, perturbation or rating. Three initial blind ratings are collected.
If all are `safe` or `minor`, `L_sev=0`. Any `severe`, `uncertain` or missing
rating triggers a separate three-senior-rater blind panel. Senior majority
`severe` yields `L_sev=1`; senior majority non-severe yields zero. Missing or
unresolved adjudication is conservatively severe in the primary analysis.

AI/VLM review is allowed only as development triage or a secondary
reproducibility channel. It cannot fill a human-rater field or support a
population-preference/risk claim.

## Style and deployed-output rubric

Each study freezes one method-independent `look_id`, its rights/source
manifest, board-construction rule and palette/tone rubric. A film name is
allowed only when the board's film origin is traceable; otherwise the task is
named a photographic look.

After severity review, every deployed output including fallback receives:

- look adherence: `pass`, `fail` or `uncertain` against the independent board;
- style strength: ordinal 1--5;
- blinded keep preference: left, tie or right against the frozen comparator.

Mean luminance, contrast, white balance and total-chroma matched controls test
generic explanations. They do not prove causal palette identity. Saturation or
contrast alone is never accepted as style evidence.

## Split and leakage contract

| Split | Purpose | Binding barrier |
|---|---|---|
| A0 | ontology, annotation, metric and hard-negative development | no tuning on A1 or B-hidden scenes |
| A1 | one hidden benchmark confirmation on unseen transform families | one evaluation; never FARO method test data |
| B0 | bank/scorer/monitor/global-policy development | no B1--B4 outcome-driven choices |
| B1 | fixed-bank all-scene empirical ceiling | disjoint selection/evaluation panels; all scenes stay in denominators |
| B2 | complete-policy calibration | select one policy; no threshold changes after B3 starts |
| B3 | one hidden end-to-end test | no error inspection before lock |
| B4 | new-source/cluster/rater replication | no feedback before primary report freeze |

Parent scene, source, uploader/creator, camera, roll, exact/perceptual hash,
transform family and failure family are mandatory split keys. Exact and
near-duplicate leakage is zero. Artifact-monitor confirmation holds out entire
transform/failure families and unseen parameterizations.

The current U4.1 provisional set, U4.2 owner-anchor replays, RF2.C0 external
outputs and ID11 red-speckle case are irrevocably A0-only. They cannot be moved
into A1 or any B-hidden split after their outcomes have been seen.

## Sampling and power worksheet

Before a confirmatory manifest opens, freeze source pools, inclusion
probabilities, strata, target weights, exclusions and one parent scene per
provenance cluster if exact-binomial inference is planned. Stress scenes are a
separate descriptive table.

Planning values remain non-binding: one-sided 95% bounds, 1% overall and
selected severe-risk budgets, 70% look adherence, 50% style-qualified
coverage, all-scene tie score above 0.55 and `K=8`. Roughly 300 independently
accepted scenes are needed even for a zero-event selected-risk upper bound near
1%; 50% coverage implies roughly 600 total scenes before grouping, failures,
label error and replication. The binding sample size remains `unknown` until
cluster, label-error and tie-model pilots are complete.

No external recruitment, payment, public release or participant contact is
authorized. If the eventual annotation budget is infeasible, the legal branch
is a wider preregistered risk budget or descriptive evidence, not correlated
crops/candidates presented as independent scenes.

## Implementation DoR/DoD

DoR is this committed contract plus the existing U4/FARO authorities. The
implementation may add a strict annotation schema, pure validators,
scene-label aggregation and split-manifest leakage checks under `src/eval/`,
with tests and example fixtures. It may not ingest new images, train SCIS,
contact raters or change production rendering.

U5.R1A is done when unknown keys/labels fail closed; autonomous evidence cannot
masquerade as human evidence; escalation and conservative missingness are
tested; A0/A1/B0--B4 leakage is rejected; the planning worksheet reports
unknown rather than invented power; authorities are propagated; and the full
CPU suite passes. Completion opens only U5.R1B synthetic/failure-suite design,
not training, human claims or adaptive routing.
