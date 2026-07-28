# U5.R2AL1 Analytic Chroma-Sector Curve Capacity Results

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AL1`  
Decision: **closed — relative capacity is competitive, but absolute fidelity
and invertibility fail**

## Integrity

- immutable contract hashes:
  - v1 `53fe4134ba384d0523d5b2d87c6b0029e15c0e414dcdf5501deddcef71780559`;
  - v2 `6adaba33dbcbd9ec84de81cb35b75534da8a40127244e9c0d1981d77d3aef00f`;
  - executable v3
    `e4e8e812f1bbffe8a4124066e34cb6e3dcd0d549d807eaa9297d9cdaa6f26384`;
- formal software commit:
  `21a877fb3c29b80bc38b70c1c6123fb4f1b5dcb8`;
- both independently reconstructed 17,780-byte reports are byte-identical at
  SHA-256
  `4a8bf70dd1b877cd96c63bee7a6f18f757ad17e96d1260ff9f35a0ee5ecee1c8`;
- parent repeat decision SHA-256:
  `cfb9ae14d35a860e7a5e1d71e5e8231690eae288a5c969de49932cc911e1ab5a`;
- the frozen target orders differ by RGB RMSE `0.002044972`, and both target
  maps pass cube, positive-Jacobian and bounded-Jacobian prerequisites;
- an independent post-run call to the frozen report validator reconstructs
  the same failed aggregate decision for both reports.

No photograph, film pixel, external code, checkpoint or learned colour-naming
asset was accessed. The formal run used deterministic float64 CPU fitting in
two fresh child processes.

## Result

The 96-raw/80-effective-parameter sector bank substantially beats the global
curve control and narrowly beats the stationary K3 control:

| Target | Candidate RMSE | Global curves | Gain vs global | Stationary K3 | Ratio to K3 |
|---|---:|---:|---:|---:|---:|
| A then B | 0.00625445 | 0.02444242 | 74.41% | 0.00636058 | 0.9833x |
| B then A | 0.00622158 | 0.02477797 | 74.89% | 0.00647631 | 0.9607x |

Both rows therefore pass the frozen 25% global-curve improvement and 1.20x K3
efficiency gates. They nevertheless exceed the separately frozen absolute
RMSE ceiling of `0.006`.

Every non-inverse structural check passes:

- exact identity, neutral axis, partition sum, serialization and chunk replay;
- exact `[0,1]` cube range;
- minimum finite-difference Jacobian determinants `0.52332` and `0.53046`;
- maximum Jacobian spectral norms `2.03785` and `2.07093`;
- hue-boundary output jumps below `1.75e-6`;
- monotone red-ramp and bounded collinear strength paths.

The inverse solver does not converge on either target. The report records the
frozen failure sentinel of maximum absolute error `1.0`; it does not silently
accept a partial or approximate inverse.

## Interpretation

The clean-room analytic partition adds real colour-selective capacity: it is
far stronger than three global curves and at least as parameter-efficient as
the stationary K3 control on this target. That useful mechanism evidence does
not satisfy the complete contract.

The representation selects sector weights from the input colour and then
blends independently curved outputs. Its forward map can remain smooth,
positive-orientation and cube-safe while the resulting implicit inverse is
not a contraction for the frozen solver. Missing the absolute threshold in
both orders independently closes the representation even if a different
inverse algorithm might later be imagined.

The branch is not rescued with more sectors, curves, degree, optimization,
changed targets, changed gates or photographs. A genuinely distinct next
family may instead use one globally conditioned, explicitly bounded
coordinate transform, separable monotone curves in that coordinate system,
and analytic inverse curves. That is a new representation question, not an
AL1 repair.

## Claim ceiling

`repeat-exact clean-room synthetic negative capacity and invertibility
evidence for one fixed analytic colour-selective monotone-curve operator`.

There is no photograph, film-pixel fitting, unpaired identification, stock,
calibration, preference, severe-safety or production claim.
