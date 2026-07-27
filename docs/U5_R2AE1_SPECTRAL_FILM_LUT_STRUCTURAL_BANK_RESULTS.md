# U5.R2AE1 — spectral_film_lut structural-bank results

## Decision

**Close: every chain fails the frozen local-orientation safety gate.**

The result is not a blandness or diversity failure. All eight external chains
are strongly stylized, remain far from the bounded joint
EV/WB/contrast/saturation control, and retain within-family differences.
However, every sampled LUT contains materially too many locally
orientation-reversing cells. The source-bound bank is therefore rejected
before photographs or visual review.

## Exact replay

- external revision:
  `02ecafd78c4a97bd0708d69a9f5ec39caf492d2e`;
- evaluator commit:
  `2afa6c07c76bd5faea7bca9baac13a785e51a60c`;
- config SHA-256:
  `d868ae4a005f873e3c6b730c138701364c42558f51b07533615291bbd5f52801`;
- formal report SHA-256:
  `2ec7cd1a362facdef198daedfabaa5cd1d6b2af03c8cfea5d49d7a55e41ce6b3`;
- the independent repeated evaluation report is byte-identical;
- all cube and neutral arrays repeat exactly across two independent process
  runs;
- the duplicate Ektar+Endura invocation is exactly equal;
- the synthetic 0.75 strength control recovers strength
  `0.7499999999999999`, residual RGB RMSE `1.76e-17` and explained energy
  `1.0`.

The two manifest files differ because their run IDs and absolute output paths
are intentionally distinct; their array identities match exactly.

## Positive structural evidence

Across the eight chains:

- median style Delta E76 versus identity spans `19.07–23.96`;
- median residual after the frozen joint basic control spans `9.81–23.72`;
- all eight pass the style and non-basic floors;
- all three common-output families contain a distinct pair after the same
  basic control;
- within-family conservative pair residuals span `1.44–5.06` Delta E76;
- all outputs are finite and in range;
- interior hard clipping is zero;
- neutral-luma monotonicity passes;
- maximum local spectral norm spans `3.31–7.72`, below the frozen cap `20`.

These results show that the external method avoids the particular
average-looking/basic-adjustment collapse that motivated the search. They do
not validate the profile names or film authenticity.

## Hard failure

The frozen gate permits at most `0.5%` cells with determinant below `-1e-5`.
Observed fractions are:

| Chain | Negative-Jacobian cells |
|---|---:|
| Kodachrome 64 direct | 1.6113% |
| Ektachrome 100D direct | 4.2236% |
| VISION3 500T + 2383 | 6.5674% |
| VISION3 50D + 2383 | 7.1777% |
| VISION3 200T + 2383 | 7.3975% |
| Portra 400 + Endura | 7.5195% |
| VISION3 250D + 2383 | 7.6172% |
| Ektar 100 + Endura | 9.4971% |

Minimum determinants span `-0.2031` to `-2.3281`. This is not numerical
near-zero noise. The identity orientation test passes exactly, an intentionally
folded red-axis test is detected at 100%, and the external LUT table ordering
was independently verified to equal the evaluator's `ij` RGB cube ordering.

## Branch closure

The contract forbids clamp, smoothing, threshold relaxation, candidate
replacement, resolution search or post-result source parameter changes.
AE1 therefore closes with `close_range_fold_or_monotonicity_failure`.

No synthetic cube image or external output was visually reviewed. No AE2
photograph stage opens. The bank is not a teacher, stock response, calibrated
profile, product asset or LSM input. Its positive style/diversity evidence and
negative topology evidence remain useful for designing a future
orientation-preserving explicit parameterization, but that must be a distinct
preregistered method rather than a rescue of these LUTs.

