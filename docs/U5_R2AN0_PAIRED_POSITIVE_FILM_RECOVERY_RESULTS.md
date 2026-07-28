# U5.R2AN0 Paired Positive-Film Recovery Results

Date: 2026-07-28
Node: `ULT > U5 > U5.R2 > U5.R2AN0`

## Result

The frozen synthetic known-operator recovery experiment passes. Two fresh
processes produce byte-identical 21,362-byte reports at
`22fc3542...a36d`; repeat decision SHA-256 is `bc2d0656...6e55`.

The design contains 3,168 rows from 96 deterministic patch groups, three
illuminants and 11 exposure levels. Fitting uses 912 rows. The 2,256-row
confirmation set is the exact complement and contains held patch, illuminant
or exposure groups.

| Truth witness | two-matrix confirmation RMSE | one-matrix RMSE | relative gain |
|---|---:|---:|---:|
| `warm_highlight_like` | `1.0643e-16` | `0.009823` | ~100% |
| `cyan_shadow_warm_highlight_like` | `8.6472e-17` | `0.012516` | ~100% |
| `cross_bias_like` | `8.5939e-17` | `0.012310` | ~100% |

All cube, positive-Jacobian, determinant, maximum-error, source-nonmutation
and repeat gates pass. This establishes that the existing J0 two-matrix
family and fitter can recover the observable transform in a controlled,
noise-free paired design, while the one-matrix ablation cannot.

## Boundary and next experiment

This is not a real Velvia fit, real-film evidence, unpaired operator
identification, calibration or product promotion. The next discriminating
leaf tests recovery under patch-mean noise and sparse corrupted
correspondences; exact noise-free recovery alone is insufficient evidence for
a practical capture-and-fit pipeline.
