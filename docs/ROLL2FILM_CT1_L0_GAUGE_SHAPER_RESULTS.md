# Roll2Film CT1 L0, gauge and shaper closure results

Date: 2026-07-24  
Decision: **L0/gauge/shaper analytic primitives pass; shaped 33/65 HDR bake fails**  
Node: `ULT > U5.CT1C`

## Outcome

The new analytic L0 exposure/white-balance operator, explicit roll nuisance
gauge and reversible log1p shaper all pass their frozen numerical contracts.
The complete CT1C node does not pass because shaped 33-cube and 65-cube LUTs
miss the preregistered maximum-error limits over `[0,16]` linear RGB.

This is a valid resolution/capacity result. The implementation converges at
the expected rate, but the two required standard cube sizes are not accurate
enough for the fixed HDR witness. Thresholds remain unchanged.

Formal report:
`outputs/roll2film_ct1_l0_gauge_shaper/formal/report.json`, SHA-256
`9be0fdc52105c9703fb5d56cf7be459120bcc7b138c00cf723c5246ed777a6b5`.

## Passing analytic primitives

| Primitive | Frozen metric | Result |
|---|---:|---:|
| L0 forward/inverse | max abs <=1e-12 | `1.78e-15` |
| L0 replay | exact | exact |
| L0 Jacobian determinant | >0 | `2.85765` |
| gauge WB row sum | max abs <=1e-12 | `1.39e-17` |
| gauge exposure mean | max abs <=1e-12 | `1.67e-17` |
| gauge recomposition | max abs <=1e-12 | `5.55e-17` |
| shaper roundtrip | max abs <=1e-12 | `1.11e-16` |
| shaper derivative | >0 | minimum `0.01475` |
| domain guards | both reject | pass |
| bundle JSON replay | exact | 33 and 65 exact |

The gauge moves `0.046` log exposure into the explicit shared/base term and
reconstructs every original per-channel frame log gain without inference from
pixels. These primitives are retained as isolated research infrastructure.

## Failed shaped-LUT frontier

| Cube | frozen max-error gate | max RGB error | RMSE | result |
|---:|---:|---:|---:|---|
| 33 | <=0.020 | 0.03511 | 0.00888 | fail |
| 65 | <=0.006 | 0.00899 | 0.00224 | fail |

The 65/33 maximum-error ratio is `0.256`, passing the `<=0.45` refinement
gate. Therefore the bake is deterministic and convergent; it is not exact
enough at the mandated resolutions. A larger cube could probably reduce the
error, but testing one now would be a post-result capacity expansion and is
not needed to preserve the analytic operator path.

## Branch decision

- retain `PhotometricColorOperator`, `RollNuisanceGauge`, `LogShaperSpec` and
  explicit shaped-LUT infrastructure as research primitives;
- do not claim CT1 complete HDR 33/65 bake parity;
- do not integrate shaped LUTs into production or use them for clipping;
- do not relax the gates or silently change interpolation after the result;
- retain the existing analytic L1/L2 operators as the authoritative exact
  path when shaped-bake error matters.

This numerical failure supplies no evidence about physical rolls, film stock,
style, preference or calibration. It does not reopen any closed learning/data
branch.

## Reproducibility

- config SHA-256:
  `d1c609ceaeb7c1ee7c1d2d5d81b941c98adec42993112d28f902e46ab0f0c5d7`;
- software commit: `bea64cbfb99551e26696db8d141e1b1a6664153b`;
- exact runs: 2/2;
- full CPU suite: 794 passed in 66.57 seconds.

## Claim ceiling

Analytic numerical validation of L0 photometric, roll-gauge and log1p shaper
primitives, plus a failed fixed-resolution HDR shaped-LUT parity test.

