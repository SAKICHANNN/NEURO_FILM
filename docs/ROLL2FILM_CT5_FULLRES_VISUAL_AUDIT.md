# Roll2Film CT5 full-resolution visual audit

Date: 2026-07-15  
Node: `ULT > U5.CT5`  
Scope: FilmSet internal confirmatory recipe transfer only

## Decision

The frozen per-recipe deterministic bank passes the internal confirmatory
full-resolution severe-artifact veto:

| Recipe | Frozen primary | Target Delta-E00 | Style Delta-E00 | Decision |
|---|---|---:|---:|---|
| Cinema | Lab mean/std | 1.8771 | 3.4607 | pass; no confirmed severe artifact |
| ClassNeg | pooled L2 | 2.3221 | 4.8406 | pass; strong style, no confirmed severe artifact |
| Velvia | pooled L2 | 3.4620 | 4.4127 | pass; strong style, no confirmed severe artifact |

This is an internal survivor decision, not final-628 clearance or product
release. It establishes that a simple fixed deterministic recipe bank can be
visibly stylized without the geometry rewrite, seams, colour blocks or content
damage seen in the retired generative path. It does not establish physical film
or named-stock authenticity.

## Frozen evidence

- Full-resolution report: `outputs/roll2film/ct5_v1/fullres/report.json`
- Report SHA-256: `edc761d25cc87b7321acbb28a744ddb9c0af1800a7e6ce588bcc7d45e2a037d6`
- Software commit: `8bf1f5725fbbd91debfed72f175cbac45b8abd50`
- Images: all 238 untouched confirmatory identities for all three recipes and
  every frozen primary/comparator
- Final test: `final_628_parsed_or_decoded=false`
- Machine checks: finite output, full-resolution target/style error, raw gamut
  excursion rate and magnitude, display clipping relative to source/target,
  luminance-bin occupancy, red/cyan boundary occupancy and chroma-island
  diagnostics
- Visual checks: deterministic primary worst-case sheets plus targeted
  original-resolution cases selected by maximum gamut excursion

The committed decision record is
`configs/roll2film_ct5_fullres_decision.json`. It pins the report, review-sheet
and targeted-audit hashes.

## Why the first gamut alarm was misleading

The earlier metric counted every channel below zero or above one equally. A
dark moon scene therefore reported 99.4% out-of-range pixels even though the
ClassNeg operator's maximum excursion was only `0.00214451` and the candidate
was visually almost indistinguishable from the hidden recipe target. This was
a floating-point boundary tail, not an image-wide glitch.

The rerun separates frequency from magnitude:

| Recipe | p95 image maximum excursion | Maximum excursion | Interpretation |
|---|---:|---:|---|
| Cinema | 0 | 0 | bounded by the baseline implementation |
| ClassNeg | 0.00214451 | 0.00214451 | small numerical tail; explicit output mapping is sufficient |
| Velvia | 0.0616976 | 0.0890066 | mandatory visual review triggered on saturated boundary scenes |

Velvia's largest excursions occur in vivid butterfly/foliage, blue-yellow
aircraft, neon and night-light scenes. Original-resolution candidate/target
inspection found strong but coherent colour and contrast, with no confirmed
unintended large-area clipping, posterized blocks, seams, high-frequency colour
speckle, text breakage, geometry failure or identity/detail rewrite.

## Product and research consequences

1. Do not weaken pooled L2 through an arbitrary RGB blend or add a learned
   gamut module. The confirmatory evidence does not justify sacrificing the
   style strength that made ClassNeg and Velvia succeed.
2. Keep raw explicit-operator output for research diagnostics. Review PNGs use
   a separately recorded hard clip and sRGB8 encoding; the production renderer
   still requires a versioned explicit output transform and provenance.
3. The fixed recipe bank is an independently viable deterministic product
   direction. It does not justify per-photo ML routing: no universal operator
   passed, and no router/Oracle gain has been demonstrated.
4. Roll2Film's special physical-roll claim remains unproved. CT6 must show that
   correct BlueNeg roll groups beat random, same-film wrong-roll,
   scene/date-matched and shuffled controls at fixed budget.
5. A new gamut-safety operator remains a conditional fallback. Reopen it only
   if the one-shot final set or later product stress evidence confirms severe,
   unintended clipping rather than merely crossing an automatic review
   threshold.

## Frozen final-set escalation rule

Before final-628 is opened, the following policy is fixed:

- non-finite output or any confirmed spatial/content severe artifact is a hard
  failure;
- per-image maximum raw excursion at least `0.05`, or newly clipped display
  pixels versus target at least `0.10`, triggers mandatory original-resolution
  adjudication but is not itself an automatic failure;
- promotion still follows severe veto first, then style/fidelity and product
  quality; no post-hoc operator weakening is allowed.

## Claim boundary

FilmSet targets are Capture One recipe outputs. These results support internal
recipe-transfer and deterministic-style engineering claims only. They do not
support physical-film response, calibrated stock identity, population
preference, redistribution or public model/data release claims.
