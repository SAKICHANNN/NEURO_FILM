# U5.R2Z1 Asymmetric Query/Operator Applicability Results

Date: 2026-07-28

Decision: **development applicability fails before fresh-target access**

## Reproducibility and data boundary

- software commit:
  `a33526e8f694f029d92e432444ca8e7140cf1490`;
- config SHA-256:
  `988f079b8f0d7658af7f352397385a159f7e30fca74ec7c5b7d0f2b515aa4166`;
- two complete reports are byte-identical at
  `bda10ab453b540e05031a357e50f8d7fac4ba2055a49612827d5263546f93b70`;
- both formal stderr logs are empty;
- the NVIDIA GeForce RTX 5070 Ti Laptop GPU ran PyTorch `2.11.0+cu128`;
- exactly 24 development identities and 48 aligned input/ClassNeg payloads
  were read;
- fresh confirmatory targets, Z0 confirmatory targets, final-628 payload rows,
  Velvia, Cinema, current stock pixels and unselected payloads were not read;
- no full-raster output or visual shortlist was generated.

The automatic development gate failed in both runs. The frozen code therefore
did not decode the 16 fresh ClassNeg targets, exactly as required by the
contract.

## Decisive off-diagonal bank result

Z0's ClassNeg Evaluator Oracle allowed each query's own fitted operator to
remain in the candidate bank. Z1 asks the transferable case question: in every
development fold, it removes both the held-out query and that case's operator.

| Metric | Frozen requirement | Off-diagonal result | Pass |
|---|---:|---:|---:|
| Eligible operators | at least 12 | 24 | yes |
| Shared O0 mean linear-RGB RMSE | control | `.013227` | n/a |
| Case-bank Oracle mean RMSE | lower is better | `.012259` | n/a |
| Oracle improvement over shared | at least 10% | `7.32%` | **no** |
| Oracle win fraction | at least 75% | `62.50%` | **no** |
| Oracle bootstrap 95% lower bound, rank 2 | greater than 0 | `-.000124` | **no** |
| Oracle bootstrap 95% lower bound, rank 4 | greater than 0 | `-.000096` | **no** |

The bank itself therefore fails the prerequisite for applicability learning.
This is stronger negative evidence than a failed selector: after excluding the
same case, the remaining 23 previous-case operators do not contain enough
stable post-hoc value to justify a router.

## Frozen rank ladder

Both preregistered interaction ranks were still evaluated on development
because neither could pass the complete gate.

| Policy | Mean RMSE | Change versus shared | Shared win rate | Oracle-gap closure | OOD fallback | Decision |
|---|---:|---:|---:|---:|---:|---|
| Rank-2 hard asymmetric | `.017002` | `-28.54%` | `20.83%` | `-389.94%` | `8.33%` | fail |
| Rank-4 hard asymmetric | `.017816` | `-34.70%` | `16.67%` | `-474.01%` | `8.33%` | fail |
| Fixed operator medoid | `.016205` | control | n/a | n/a | n/a | better than both ranks |
| Spatial photometric NN | `.018383` | control | n/a | n/a | n/a | still not promotable |
| Exact random expectation | `.020379` | control | n/a | n/a | n/a | lower bar only |

Rank 2 beats the failed spatial, random and shuffled controls, but is worse
than both shared O0 and the fixed medoid. Its paired-bootstrap mean improvement
over shared has 95% interval `[-.005962, -.001814]`. Rank 4 is worse again,
fails the shuffled control as well, and has interval
`[-.006696, -.002570]`. Beating weak controls cannot override losing to the
global champion.

## Structural evidence

The shared and all 24 case operators remain numerically valid:

- minimum sampled Jacobian determinant: `.193923`;
- maximum sampled Jacobian spectral norm: `1.923287`;
- maximum inverse error: `4.34e-8`;
- maximum coefficient-vector norm: `1.134414` under the `2.0` cap;
- sampled output range: `[.049235, .964627]`;
- exact replay error: `0`.

The result is not caused by folding, out-of-cube output, coefficient explosion,
inverse failure or nondeterminism. It is an information and transfer failure.

## Interpretation and branch consequences

The Z0 ClassNeg Oracle value was materially dependent on leaving the query's
own operator in the bank. Once the literal "which previous case applies to
this new photograph?" condition is enforced, even the post-hoc Oracle misses
all three bank-value gates. A larger selector cannot manufacture missing bank
value.

- Close Z1 as `development_applicability_fail`.
- Keep all 16 fresh targets sealed; there is no confirmatory or visual stage.
- Do not add rank, kernel, neural, semantic, local or direct-RGB capacity.
- Do not average operators, tune thresholds, reopen Z0, access final 628 or
  promote a FilmCase/router product path from this software-recipe control.
- Preserve Z0 only as evidence that per-case fitting can encode paired
  self-case variation, not that previous-case retrieval transfers.
- Continue Ultimate through another evidence-authorised explicit algorithm or
  product leaf.

`ClassNeg` is a Capture One software-recipe domain. This result does not
identify a physical film stock, latent stock mode, calibrated response or
unpaired digital-to-film operator.
