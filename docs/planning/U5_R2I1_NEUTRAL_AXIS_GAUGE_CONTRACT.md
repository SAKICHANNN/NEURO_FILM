# U5.R2I1 Neutral-Axis Gauge Contract

**Frozen:** 2026-07-24  
**Node:** `ULT > U5 > U5.R2 > U5.R2I1`  
**Status:** numerical contract frozen before implementation/formal result

## Hypothesis

U5.R2I0's uniform green cast is caused in part by an unconstrained
display-domain neutral axis, not by a necessary chromatic interaction. A
data-independent output gauge can remove that component while leaving
non-neutral hue/luma interactions materially nonlinear.

The gauge is derived only from the immutable operator itself. It never reads a
photograph, film scan, label, content embedding or visual score.

## Fixed construction

For 1,025 uniformly spaced neutral inputs `t`, evaluate the U2.2B base as
`F(t,t,t)`. For each output channel `c`, construct one strictly monotone
rational-quadratic spline mapping `F_c(t,t,t) -> t`. The gauged operator is:

`G(x)_c = inverse_neutral_c(F(x)_c)`.

This is a coordinate gauge, not a fitted stock correction. It is deterministic
from the frozen U2.2B parameters. The knot count, spline family and all gates
are frozen. There is no per-image white balance, exposure, statistic, content
route, spatial operation or strength.

## Numerical gates

On synthetic probes only:

- document at least `0.1` maximum channel spread on the ungauged neutral axis;
- gauged dense neutral maximum absolute error from `t` at most `2e-4` and
  maximum channel spread at most `1e-5`;
- black/white endpoints within `1e-12`;
- output remains in `[0,1]` and sampled Jacobian determinant is at least `.01`;
- identity RMSE at least `.08` and best-affine residual at least `.04`, so the
  gauge does not erase the nonlinear look;
- exact partition, serialization replay, source preservation, invalid-input
  rejection and two identical formal reports.

## Branches

- **Pass:** freeze a separate U5.R2I1B real-image frontier using unchanged R2B
  style/non-basic/clipping and severe gates. A numerical pass alone is not
  visual or stock evidence.
- **Neutral failure:** close the gauge; do not raise knots or change splines
  after the result.
- **Range/Jacobian failure:** reject as unsafe; no clamping rescue.
- **Look-collapse failure:** retain the ungauged operator only as the existing
  negative control and close this route.

## Boundaries

No current film pixels may be fitted; operator/training/LSM permissions remain
false. No generative model, production renderer/profile/default integration,
stock response, calibration, authenticity or preference claim may follow from
this numerical leaf.
