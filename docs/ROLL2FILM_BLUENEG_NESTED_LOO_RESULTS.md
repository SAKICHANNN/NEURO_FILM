# RF1.2 BlueNeg nested LOO results

## Outcome

`nuisance_explanation`: the four-roll retrospective diagnosis does not reopen
the physical-roll information hypothesis. A source-content retrieval operator
trained on three frames from **other rolls** is better than the correct-roll
operator.

This is useful architectural evidence for similar-case retrieval of bounded
explicit operators. It is not evidence that BlueNeg teaches a film look:
BlueNeg's hidden targets are archive pseudo-ground-truth restorations, not
digital/film pairs or calibrated scan truth.

## Reproducibility

- 43 complete aligned frames, each used as the query exactly once.
- 13/6/12/12 queries across four Kodak Gold 100-5 rolls.
- Three non-query support frames and equal pixel budgets in every fitted arm.
- 86 used image payloads and all upstream evidence hashes verified before decode.
- Only the already frozen Lab mean/std operator was fitted; no capacity was added.
- Report: `outputs/roll2film/blueneg_v1/nested_loo_report.json`.
- Report SHA-256: `3b87a8604667bceae9bdd9506daa60f7380ef574fd20145732c9fcf96c1fb333`.
- A complete second run was byte-identical.
- Software commit: `ebcc73c9f84c84b004328c6bbbd1b641b425ba48`.

## Primary raw-strength result

| Arm | Mean target Delta-E00 | Difference versus correct roll |
|---|---:|---:|
| content retrieval, wrong rolls | **5.1805** | -0.4239 |
| shuffled roll | 5.5812 | -0.0232 |
| pooled wrong rolls | 5.6021 | -0.0022 |
| correct physical roll | 5.6044 | reference |
| identity | 7.6223 | +2.0179 |

Content retrieval beats correct roll on all four physical-roll means. The
cluster-equal correct-roll gain is -0.3856 Delta-E with 95% bootstrap interval
`[-0.4956, -0.2071]`. Raw Lab outputs have zero sampled out-of-range pixels.
Correct roll strongly beats identity, but pooled and shuffled groupings are
essentially tied with it. The useful signal is therefore a shared restoration
mapping plus source-conditioned variation, not identified roll membership.

## Source-only style-matched result

| Arm | Mean target Delta-E00 |
|---|---:|
| content retrieval, wrong rolls | **5.4933** |
| correct physical roll | 5.6044 |
| pooled wrong rolls | 5.7644 |
| shuffled roll | 5.7736 |
| identity | 7.6223 |

After matching each arm to correct-roll style strength using query source
pixels only, retrieval remains better on three of four rolls. Correct-roll gain
versus retrieval is -0.1130 with interval `[-0.2303, +0.0574]`. This secondary
analysis is not safety evidence: extrapolative matching reaches a 20.63%
sampled out-of-range maximum in the worst arm. The zero-excursion raw result is
the primary interpretation and reaches the same conclusion more strongly.

The correct operator beats the mean of the three individual wrong-roll
operators by 0.075 Delta-E, but loses to the per-query best wrong-roll oracle by
0.651. That oracle is explicitly non-deployable; it confirms heterogeneity
rather than providing a product baseline.

## What retrieval is matching

The mean standardized source-descriptor distance falls from 2.458 for correct
roll supports to 1.355 for retrieved supports. Retrieval supports share the
query location only 2.3% of the time, so the result is not a simple location
lookup. They share the recorded day/night plus indoor/outdoor scene property
77.5% of the time, above the correct-roll support rate of 66.7%.

This supports a narrow mechanism: content/illumination-conditioned selection
can identify a better bounded explicit transform than physical-roll identity in
this archive. It does not identify whether the cause is restoration practice,
exposure, negative condition, scanner behaviour or genuine emulsion variation.

## Decision and propagation

- Keep `roll_information_established=false`.
- Close RF1.2 as a nuisance explanation, not a failed software experiment.
- Promote source-conditioned similar-case retrieval as one RF2 architecture challenger.
- Require RF1.1/RF2 real-film evidence to use original scans and honest source holdouts.
- Do not train a GPU router until a CPU retrieval advantage survives a real-film
  signal/content/nuisance gate and severe-artifact review.
