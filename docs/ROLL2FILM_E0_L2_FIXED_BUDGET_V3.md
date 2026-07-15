# Roll2Film E0 v3 - fixed-budget L2 nonlinear falsification

> Date: 2026-07-15
> Node: `ULT > U5.CT1/U5.CT3`
> Decision: `method_control=pass`; `roll_information=not_established`

## Contract

This experiment raises the known truth from SPD affine to an explicit L2
operator: an orientation-preserving affine map followed by three strictly
monotone rational-quadratic splines. The operator has analytic inverse and
positive Jacobian diagnostics, deterministic canonical serialization, and
33-cube/65-cube dense-LUT bake checks.

Every group-size arm receives exactly 4,096 target pixels. Group sizes are
`1/2/4/8/16/32`; only pixels per frame change. The run uses 32 independent
replicates and 5,000 paired bootstrap resamples. Passing cannot establish
physical-roll information because all groups remain synthetic.

## Result at 32 frames

| Paired control | Baseline RMSE | Candidate RMSE | Relative improvement | 95% CI for absolute improvement |
|---|---:|---:|---:|---:|
| repeated support -> independent support | 0.024197 | 0.005542 | 77.1% | [0.016821, 0.020697] |
| mixed operator -> correct single operator | 0.082791 | 0.005542 | 93.3% | [0.076517, 0.077971] |
| random boundaries -> true nuisance boundaries | 0.048832 | 0.022485 | 54.0% | [0.021608, 0.031139] |

Partitioning one unchanged pixel pool produces an operator-grid maximum
difference of exactly `0.0`. The truth Jacobian minimum on the frozen 17-cube
grid is `0.776540`. Maximum analytic-versus-LUT errors are `0.0006221` at
33-cube and `0.0001636` at 65-cube.

## Negative evidence outside the pass gate

- shifting the assumed source prior raises mean holdout error from `0.005412`
  to `0.031684`;
- with an input-side scanner/profile affine composed before the film operator,
  the estimate is closer to the composite (`0.005496`) than to the film-only
  component (`0.021536`);
- applying flexible per-frame RGB-mean normalization to a clean roll is worse
  than leaving it untouched (`0.005775` versus `0.005412`).

These are expected identifiability failures, not successes. The scanner test
is an input-side profile composition representable by the current L2 family;
it does not yet cover a post-film scanner map outside that factorization.

## Reproducibility

- implementation commit: `369ff4a53a37d40af2cbd049037a672f21775887`;
- config SHA-256: `91229d024997c29e32a08482c9ce8939b1207f03a404576d215d6b1b491be8c8`;
- ignored report: `outputs/roll2film/e0_l2_fixed_budget/report.json`;
- report SHA-256: `dc85c9e035d644f93bbeddbd444393cfadb7b902220d6e19e9cb19086826c04c`;
- a second complete run produced the same report byte for byte;
- 64 repository tests pass after the implementation.

## Decision and next gate

The L2 method controls pass. This closes the immediate nonlinear simulator
mechanics leaf but does not promote Roll2Film as a scientific method or a
product algorithm. `roll_information` remains `not_established` until real
group metadata, matched wrong-group controls and held-out transfer evidence
show information beyond fixed sample count, content support and nuisance
boundaries.

Next freeze matched-strength deterministic/classical CT5 baselines and the
internal FilmSet evaluator policy. BlueNeg acquisition remains behind the
metadata/whole-roll split gate; the official FilmSet 628 lockbox remains
sealed.
