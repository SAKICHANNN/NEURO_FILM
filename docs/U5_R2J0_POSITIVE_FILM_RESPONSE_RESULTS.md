# U5.R2J0 positive-film-response results

Date: 2026-07-26

Node: `ULT > U5 > U5.R2 > U5.R2J0`

Decision: **representation pass; open a separate real-image frontier**

## Result

The frozen clean-room two-matrix/three-sigmoid family passes every numerical,
regularity, diversity and replay gate for all five original witnesses.

| Gate summary | Frozen result |
|---|---:|
| witnesses | 5 |
| grid points per witness | 4,913 |
| maximum endpoint error | 0 |
| output range | exactly `[0,1]` |
| minimum finite-difference Jacobian determinant | 0.0000334233 |
| minimum directional derivative | 0.00254117 |
| minimum identity RGB RMSE | 0.382954 |
| minimum pairwise witness RGB RMSE | 0.0212405 |
| minimum residual after best affine RGB fit | 0.0959595 |

Two formal runs on the same software revision are byte-identical. Strength
zero is exact identity, strength one is exact full response, serialized replay
is exact and partitioned execution is byte-identical.

## Interpretation

This is a stronger representation result than the failed I1B route: removing
the extra negative-to-print inversion does not collapse the family into a
basic global adjustment. All five witnesses preserve a substantial nonlinear
residual after the best affine RGB fit while remaining continuous, bounded and
orientation-preserving on the frozen grid.

The result does **not** establish that the witnesses look good on photographs.
Their large identity distances also make a bounded strength frontier and
severe-artifact/clipping audit mandatory. No witness is selected here.

## Reproducibility

- frozen config SHA-256:
  `3fa39f74faede3fac5767fca9f29a9ec61859d8b879a2ff95293163811a9fcf2`;
- evaluation software commit:
  `251388adeffc0844c50e3843c1e1cb23f5a363e1`;
- both formal report SHA-256 values:
  `77fb4ee9695b89735461ece2f23d6b841abd87ba0d1aaebd084f1e5e06fa7c53`;
- five focused tests pass;
- complete CPU suite: 818 tests pass.

## Claim boundary and branch

The implementation is independently written from the public mathematical
description and uses no publication parameter, measurement, image, LUT or
source code. It is a
`film-inspired/data-independent positive-film architecture witness`, not
Velvia, CineStill, a measured film response, calibrated stock response,
physical validation or preference evidence.

Open `U5.R2J1` only: freeze the five witnesses across bounded strengths on the
existing 9-gold/32-stress frontier. Reuse the inherited style, non-basic and
new-clipping gates; compare against safe-rich and the retained R2E1
cyan-shadow/warm-highlight challenger; permit blind/full-resolution visual
review only for automatic survivors.

Training, current-pixel fitting, LSM and production integration remain closed.
The Ultimate Goal remains active.
