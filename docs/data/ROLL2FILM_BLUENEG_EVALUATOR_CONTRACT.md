# Roll2Film BlueNeg fixed-budget evaluator contract

Date: 2026-07-15

Node: `ULT > U5.CT6`

Status: frozen before first BlueNeg image decode

## Question

For two held-out `Kodak Gold 100-5` physical rolls, does an explicit colour
operator inferred from three correct-roll unpaired support frames transfer six
hidden aligned query frames better than every fixed wrong-roll, pooled,
shuffled and identity control at the same support budget?

This is a mechanism falsification, not an aesthetic stock-calibration study.

## Access boundary

- Development rolls: `19960816G`, `19960816H`
- Confirmatory rolls: `19960817H`, `19970620C`
- Exploratory `Kodak Gold 100` roll `19941219C` cannot affect promotion.
- The support loader exposes source and target sets without correspondence.
- Hidden query alignment is evaluator-only.
- Any FilmSet final-628 payload remains sealed and unrelated to this run.

All source, frame-manifest, acquisition, download and transformation hashes are
frozen in `configs/roll2film_blueneg_evaluator.json`.

## Alignment contract

Preview and pseudo-GT are decoded through the ICC-aware `WorkingImage` path to
linear sRGB. No resize is allowed. For each frame, intersect the official bbox
with preview bounds, then crop the corresponding offset region from the
pre-warped pseudo-GT. Negative bbox coordinates must never become Python
negative indices. Exact aligned shape and a minimum 64-pixel side are required.

## Fixed budgets

Each operator sees exactly three support frames per roll and 4,096 pixels per
frame from each domain. Source/target support alignment is destroyed. Candidate
selection uses exactly three hidden query frames per development roll and
16,384 aligned pixels per query frame. Confirmatory evaluation uses all six
pre-frozen hidden query frames in each held-out roll at the same per-frame pixel
budget.

Images are equally weighted; a larger image or roll cannot win by contributing
more pixels.

## Development-only operator selection

The fixed ladder is identity, WB/contrast/saturation, Lab mean/std,
per-channel quantile, Gaussian/Bures affine transport and pooled L2. Select the
simplest family within 0.25 Delta-E00 of the development winner, subject to a
median query style displacement of at least 0.5 and improvement over identity
on both development rolls. Freeze exactly one operator family before any
confirmatory pixel is decoded.

If no family passes, stop CT6 without opening confirmatory pixels.

## Confirmatory controls

The correct-roll operator is compared with:

- identity;
- one operator fit from both development rolls pooled;
- each development same-film wrong roll separately;
- the other held-out confirmatory roll's support set;
- deterministic shuffled support groups.

All operators use the same family, frame count and pixel budget. No post-hoc
strength blending or per-query selection is allowed.

## Decision rule

Primary error is hidden-query mean Delta-E00. The gain is best-control error
minus correct-roll error. Query frames are the bootstrap clusters; 10,000
resamples use a frozen seed.

A narrow mechanism pass requires positive correct-roll gain on both held-out
rolls, a 95% clustered-bootstrap lower bound above zero against the best
control, aggregate superiority to every control, the style floor and no
confirmed original-resolution severe artifact.

One-roll-only gain, an interval containing zero, one-frame dependence,
nuisance-stratum dependence or loss to a matched wrong-roll control is
`ambiguous`, not a pass. Nonpositive results on both rolls, decisive control or
identity superiority, style collapse or any severe artifact is a failure.

Even a pass is capped at evidence that this small archive contains reusable
roll-group information. Four rolls of one film string cannot establish a broad
Roll2Film, stock-response or digital-to-film claim.
