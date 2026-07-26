# U5.R2R1 Published D-LUT Asset Structural Results

Date: 2026-07-26

Decision: **close the published trajectory before image rendering**

## Reproducibility

The clean-room evaluator ran twice at software commit
`0e28a845f58a99a577dc779b9c9e810c34d7d0a3`.

- config SHA-256:
  `7677e4303d49e4395b102b1ab4e032ee5bd13f21c66204cb9e4e1e3e8ddac3e1`;
- both report SHA-256 values:
  `4250470f284a99ca45fff2148a5a3744254096e15ab8628eef28da261d2ce09c`;
- official LUT manifest:
  `41` files / `4,156,160` bytes /
  `a1263c4e547c8f2f66c44ac927a159c3e9599b73e737a35248601c59d1f2e1e3`;
- official checkpoint:
  `374,671` bytes /
  `7686a2b0c42b89b59354a35233c67ef87e298d4cfc1b1073554d4bf03cb9d4f0`.

Five focused parser/Jacobian/fail-closed tests pass.

## Gate result

| Gate | Result |
|---|---:|
| Step-0 identity error `<=1e-6` | pass, `4.29e-7` |
| Nonidentity steps | 40 |
| Node range `[0,1]` | `0/40` |
| Positive trilinear and tetrahedral orientation | `0/40` |
| Jacobian norm `<=8` | `40/40` |
| Complete nonidentity structural survivor | `0/40` |
| Byte-identical repeat | pass |

The branch is `no_nonidentity_survivor`.

## Failure starts immediately

At published step 1:

- node range is `[-0.010666, 1.019211]`;
- minimum sampled trilinear determinant is `-0.5052`;
- minimum exact tetrahedral determinant is `-0.5665`.

This is not a late-stage over-stylization problem that can be solved by merely
selecting a smaller positive step. The first available nonidentity LUT already
fails both range and orientation.

## The final mapping is strong but folded

At step 40:

- identity RMSE is `0.2080`;
- residual RMSE after the best global affine fit is `0.05240`;
- node range is `[0.037599, 1.086]`;
- sampled trilinear minimum determinant is `-0.8139`, with `48.02%`
  nonpositive samples;
- exact tetrahedral minimum determinant is `-0.5679`, with `48.85%`
  nonpositive tetrahedra;
- maximum Jacobian spectral norm is only `3.084`, below the frozen cap of `8`.

This is important evidence for the product question. D-LUT does avoid the
“only saturation/contrast” failure: its final LUT is visibly large in
operator space and materially non-affine. But the strong style is obtained
with pervasive local reversals and an out-of-range shoulder, not with a safe
bounded colour diffeomorphism.

Negative Jacobians are a structural risk, not an autonomous visual artifact
label. Because the automatic structural gate failed, the contract forbids
rendering and visual adjudication; no claim is made about which folds would
be visible on a particular photograph.

## Decision and next algorithm

The official D-LUT trajectory is closed without clipping, smoothing,
projection, identity contraction, step shopping, retraining or another seed.

One part is worth preserving as a research hypothesis: a reference image's
colour-density score can provide a strong palette direction. The unsafe part
is independently moving LUT nodes with stochastic Langevin updates.

The next distinct clean-room candidate should therefore combine:

```text
reference palette score
  -> bounded low-dimensional velocity coefficients
  -> boundary-vanishing stationary RGB flow
  -> deterministic integration
  -> explicit cube-preserving operator
```

U5.R2O0 has already established that the target representation can preserve
the RGB cube, positive orientation and an inverse. A separate contract must
test whether palette-score forcing remains reference-sensitive and
non-basic after being expressed through that safe representation. This does
not open current film pixels, stock fitting or unpaired operator claims.
