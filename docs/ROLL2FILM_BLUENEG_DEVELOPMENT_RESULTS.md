# Roll2Film BlueNeg development-only family selection

Date: 2026-07-15

Node: `ULT > U5.CT6`

## Decision

Freeze `lab_mean_std` as the only operator family allowed in confirmatory CT6.
The decision used only development rolls `19960816G` and `19960816H`; no
confirmatory roll pixels were decoded.

| Candidate | Mean hidden-query Delta-E00 | Mean style Delta-E00 | Beats identity on both rolls |
|---|---:|---:|---|
| Lab mean/std | 5.576 | 5.184 | yes |
| WB/contrast/saturation | 6.255 | 3.560 | yes |
| Gaussian/Bures | 6.286 | 7.389 | yes |
| pooled L2 | 6.962 | 7.544 | yes |
| per-channel quantile | 7.081 | 4.926 | yes |
| identity | 8.289 | 0 | baseline |

Lab is both the best aggregate candidate and the simplest candidate within the
pre-frozen 0.25 Delta-E00 tolerance. It reaches `5.323` and `5.830` on the two
development rolls.

The higher-capacity pooled L2 is more visually forceful by the sampled style
metric, but it is less accurate on average and unstable across rolls: `5.297`
on `19960816G` versus `8.627` on `19960816H`, with 8.72% raw out-of-range
channels on the latter. This is negative evidence against assuming that a more
expressive unpaired operator learns the reusable roll characteristic.

## Reproducibility

- Development report:
  `outputs/roll2film/blueneg_v1/development_report.json`
- Report SHA-256:
  `d66cfbae2777e2522b8587164c21af46c916e5203ec10e723dc75a24fbf3a897`
- Software commit: `fd2a816703b9de1cb919f6aa05cf7ef876e49d9d`
- The report reruns byte-identically.
- Frozen decision:
  `configs/roll2film_blueneg_development_decision.json`
- `confirmatory_roll_pixels_decoded=false`
- `filmset_final_628_parsed_or_decoded=false`

## Nuisance warning

Both development rolls are coupled to Stanford scenes and one capture date.
The held-out rolls have different dates and locations, while support/query
locations still overlap within each held-out roll. Correct-roll advantage can
therefore be caused by location, illumination, deterioration or scan context
rather than a reusable roll emulsion/process look.

The confirmatory report must retain query IDs and location/day/indoor strata.
If gain is present only where support and query locations match, the frozen
decision rule labels the result ambiguous even if aggregate Delta-E improves.

## Handoff

Implement only the fixed Lab family in the confirmatory runner. Compare
correct-roll support against identity, each development wrong roll, pooled
development, the other confirmatory roll and deterministic shuffled support
groups at equal budget. Do not tune Lab strength or reopen L2 after seeing the
held-out rolls.
