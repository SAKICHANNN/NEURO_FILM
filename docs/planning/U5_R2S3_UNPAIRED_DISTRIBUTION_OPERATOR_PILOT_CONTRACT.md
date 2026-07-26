# U5.R2S3 Unpaired Distribution-to-Operator Pilot Contract

Date frozen: 2026-07-26

Parent: `U5.R2S2D`

Status: ready

## Question

Can a safe explicit colour flow be recovered by matching independent neutral
and styled colour distributions, rather than predicting the flow from their
histogram difference?

This is a stricter test of unpaired identification. A method can make two
distributions look alike while learning the wrong pointwise operator, so
distribution fit and oracle recovery are separate mandatory gates.

## Synthetic design

Eight unseen analytic style flows are fixed. For each style, observation sets
A and B independently generate:

- six neutral content scenes;
- six different content scenes transformed by the style;
- 256 colour samples per scene.

All scenes come from the same content meta-distribution, but none are paired or
reused across neutral/styled lanes. The optimizer sees only the two independent
sample clouds. The true style grid is hidden until evaluation.

## Fixed candidates

1. sliced Wasserstein with 24 fixed random projections and sorted quantiles;
2. random-Fourier MMD with 64 frequencies at each fixed bandwidth
   `.08/.16/.32`.

Each candidate directly optimizes one `4^3 x 3` velocity grid from identity for
200 Adam steps. Coefficients use the same vector cap `2`; final RGB comes only
from the 32-step diffeomorphic renderer. Fixed coefficient and smoothness
regularization apply.

The local RTX 5070 Ti Laptop GPU is permitted with deterministic algorithms;
CPU fallback is valid.

## Cross-fit and gates

Fit A and fit B are independent. Each fitted operator is tested on the other
observation distribution. The result must jointly:

- improve heldout distribution loss by at least 30% from identity;
- recover the hidden oracle operator at median/p90 output RMSE
  `<=.07/.10`;
- agree across A/B fits at median operator RMSE `<=.04`;
- retain oracle style strength within `[.6,1.4]`;
- stay in cube with determinant `>.005`, max norm `<=8`, inverse `<=1e-5`,
  coefficient norm `<=2`, exact replay and exact report repeat.

Distribution success with operator failure is an explicit
`unpaired operator unidentified` result, not a partial pass.

## Stop rules

No loss, projection, bandwidth, step-count, regularizer, seed or optimizer
rescue is allowed after results. No neural network, direct RGB generator,
project photograph or film pixel is permitted. Seed `28203` remains untouched.

## Claim ceiling

This is a small synthetic pilot under an idealized matched content
meta-distribution. Even a pass does not establish real digital-to-film
identification.
