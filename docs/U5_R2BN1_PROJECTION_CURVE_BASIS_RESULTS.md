# U5.R2BN1 projection-curve basis results

BN1 passes its frozen digital-retouch development gates. The clean-room
operator uses 16 fixed nonnegative RGB projections, nine-knot 1D curves, an
eight-component coefficient basis and the unchanged AY0 source descriptor.
It predicts parameters only; final pixels come from the deterministic explicit
operator. A fixed strict-interior envelope and one operator-level Jacobian dose
replace clipping or per-pixel gamut projection.

| population | mean gain vs global | wins | P95 ratio | worst ratio | median/min dose |
|---|---:|---:|---:|---:|---:|
| AY0 camera-group cross-fit | 7.24% | 70.31% | .9563 | .9537 | .8145 / .3429 |
| AY3 seen-fresh | 7.95% | 68.25% | .9757 | .8180 | .8274 / .7360 |
| AY6 seen-confirmation | 7.75% | 68.75% | .9122 | .9134 | .8085 / .7167 |

All three populations retain at least the global AO6 style level. Across the
adaptive operators there are zero out-of-cube pixels, zero new epsilon-boundary
pixels and zero sampled nonpositive Jacobians. Two independent reports are
byte-identical at SHA-256 `0a5c78a9...52a2d4` (stable evidence
`04798a7e...ca581f`).

This is not a reproduction of Lee, Ko and Kim's unlicensed implementation.
Only their published many-projection idea is used as prior art. The paper's
learned reconstruction and unconstrained curve path are replaced by a new
strict-interior residual formulation. BN1 uses already-seen FiveK digital
retouching controls, not film. It opens only unchanged confirmation on the
disjoint retained AY2 content rows; no visual or product review is authorized.

Primary sources: [paper](https://arxiv.org/abs/2510.02713),
[author repository](https://github.com/jbnu-vilab/pigment_enhancement).
