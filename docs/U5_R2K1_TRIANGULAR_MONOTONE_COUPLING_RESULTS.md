# U5.R2K1 Triangular Monotone Coupling Results

## Decision

**Closed on fidelity.** The twelve-stage coupling operator achieves the
intended structural result—bounded RGB, positive orientation, finite stage
derivatives and a closed-form inverse—but it underfits both frozen nonlinear
controls. It also misses the exact partition-parity gate by one float64 ULP.

No capacity increase, optimizer change, threshold relaxation or real-image
frontier opens.

## Reproducibility

| Item | Value |
|---|---|
| Software commit | `fa96a488f1d44029c5732d86b79b7c00de29789d` |
| Config SHA-256 | `3466d2737bf3593994f3771ecdbb9f1244a49f2746dbcde3049777d4993163c4` |
| Report SHA-256 | `e37726e277b072be71e75f4ecb95021aa847f91b48833929d8ddf4a3b281cbb1` |
| Repeated reports | byte-identical |
| Fit / confirmation / Jacobian points | `512 / 1,331 / 343` |
| Parameters | `144` explicit conditioner coefficients |
| Focused tests | `4 passed` |

## Results

| Target | Confirmation RMSE | Global-affine RMSE | Relative gain | Min det(J) | Min stage derivative | Inverse max error |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0 | ~0 | — | 1.0000 | 1.0000 | 0 |
| density cyan s0.50 | 0.06232 | 0.07016 | 11.17% | 0.1677 | 0.4938 | `5.55e-16` |
| positive warm s0.35 | 0.02819 | 0.02755 | -2.34% | 0.3588 | 0.6981 | `4.44e-16` |

The frozen fidelity ceiling was `0.015` and the nonlinear-gain floor was 20%.
Both nonlinear targets fail. The positive-film fit is slightly worse than the
global affine control.

Outputs remain in `[0,1]` without clamp. Maximum Jacobian spectral norms are
`2.663` and `1.358`, coefficients remain within the `1.5` bound, and exact
serialization replay passes. Partitioned evaluation differs from full-batch
evaluation by `1.11e-16` on both nonlinear fits, failing the deliberately exact
gate.

## Interpretation

K1 separates two questions that K0 conflated:

- structural boundedness/invertibility is achievable with explicit coupling;
- that property alone does not provide enough useful colour-transform
  capacity at the tested compact budget.

The coupling primitive is retained as research evidence, not a candidate
renderer. K1 must not be enlarged or reoptimized against the same confirmation
grid. A future leaf needs an independently motivated representation or new
evidence, not a same-data capacity chase.

## Claim ceiling

This is clean-room, data-independent negative capacity evidence. It is not a
generative model, learned style latent, identified digital-to-film operator,
named stock response, calibration, authenticity, preference result or
production promotion.
