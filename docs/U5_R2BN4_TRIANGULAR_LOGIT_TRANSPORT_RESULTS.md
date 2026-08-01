# U5.R2BN4 triangular logit transport development result

BN4 passes its frozen development gates. Two complete runs are byte-identical
at `0911b018...9133` (stable evidence `fc6abc02...0ec22`). The result freezes
the representation and opens only one unchanged disjoint-content confirmation.

| Population | Oracle vs identity | Adaptive vs global | Win fraction | P95 ratio | Worst ratio | Style ratio | Minimum dose |
|---|---:|---:|---:|---:|---:|---:|---:|
| AY0 camera-group OOF | 31.49% | 10.73% | 68.75% | 0.9188 | 0.9008 | 0.9279 | 0.7170 |
| AY3 fresh | 36.44% | 9.55% | 71.43% | 0.9438 | 0.8498 | 0.8735 | 0.6953 |
| AY6 confirmation | 39.13% | 8.87% | 64.06% | 0.8575 | 0.9187 | 0.9038 | 0.5967 |

Across all arms and populations there are zero new exact-boundary pixels, zero
out-of-cube pixels and zero nonpositive sampled Jacobians. Minimum sampled
determinant is `0.0200000000123`; maximum condition is `29.9999999914`; maximum
analytic inverse error is `2.998e-15`. No hard clipping or post-operator pixel
scaling is used.

This is paired MIT-Adobe FiveK digital-retouch development evidence, not film,
stock, calibration, preference or product evidence. The 14-parameter operator,
fit, source-only descriptor, rank, regularization, bounds, safety search and
thresholds are now frozen. AY2 may be used only for unchanged confirmation.
