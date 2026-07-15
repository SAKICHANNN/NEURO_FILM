# Roll2Film BlueNeg confirmatory result

Date: 2026-07-15

Node: `ULT > U5.CT6`

## Outcome

`ambiguous_roll_information`

The narrow four-roll BlueNeg mechanism pilot does not establish that correct
physical-roll grouping provides reusable colour-operator information.

| Endpoint | Result |
|---|---:|
| Correct-roll mean hidden-query Delta-E00 | 5.2008 |
| Composite own-best-control Delta-E00 | 5.2455 |
| Cluster-equal gain | +0.0447 |
| Roll-cluster 95% interval | [-0.2201, +0.3095] |
| Correct-roll style Delta-E00 | 7.0192 |
| Sampled raw out-of-range | 0 |

The small aggregate advantage is not replicated across rolls:

| Held-out roll | Own strongest control | Correct-roll gain |
|---|---|---:|
| 19960817H | pooled development rolls | -0.2201 |
| 19970620C | development wrong roll 19960816G | +0.3095 |

The first roll is harmed by correct-roll support, while the second benefits.
With physical roll as the cluster, the confidence interval crosses zero. This
matches the preregistered ambiguous branch: one-roll-only gain cannot be
promoted.

## Controls and nuisance analysis

Correct-roll aggregate error is lower than every single frozen control when
averaged over both rolls, but that aggregate hides the sign reversal.

| Control | Delta-E00 |
|---|---:|
| pooled development rolls | 5.2740 |
| development wrong roll 19960816G | 5.3730 |
| development wrong roll 19960816H | 5.7656 |
| other confirmatory roll | 5.7160 |
| identity | 7.2307 |
| shuffled support groups | 7.6932 |

Support-location-seen and unseen queries have similarly tiny aggregate gains
(`+0.0431` and `+0.0468`), but each stratum mixes the negative and positive
physical rolls. It therefore does not rescue the failed replication. Roll,
date, location, deterioration and scan context remain inseparable.

## Decision

1. `roll_information_established=false`.
2. Do not run CT7 amortized/permutation-invariant set inference. A larger model
   cannot repair missing replicated group information and would invite
   overfitting to four rolls.
3. Do not perform a full-resolution promotion audit for this challenger; the
   statistical gate failed first. Sampled Lab output is bounded and
   style-strong, but safety cannot convert an ambiguous mechanism into a pass.
4. Reopen only with several additional independent matched-control rolls across
   multiple dates/locations and a new preregistered freeze.
5. Preserve the independent CT5 deterministic fixed recipe bank. BlueNeg does
   not invalidate its FilmSet recipe-transfer or product evidence.

## Reproducibility

- Confirmatory report:
  `outputs/roll2film/blueneg_v1/confirmatory_report.json`
- Report SHA-256:
  `4c30a9887069e4921233e9b4b4e7ecbce3e3e8a84e4aa4c74708c856be4f6af8`
- Software commit: `c79c8ddb54a1d78e8d957ab7e4b72a53506ebd3f`
- The report reruns byte-identically.
- Frozen decision:
  `configs/roll2film_blueneg_confirmatory_decision.json`
- FilmSet final-628 remains unparsed and undecoded.

## Claim boundary

This result supports neither reusable physical-roll information nor a stock
response. It is not digital-to-film evidence, calibrated authenticity,
multi-stock generalization or population preference. The scientifically correct
conclusion is ambiguous under the current data, not a weak success.
