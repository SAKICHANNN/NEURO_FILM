# U5.R2K0 Bounded Gaussian Residual Results

## Decision

**Closed on structural regularity.** The candidate reproduced both frozen
nonlinear controls accurately and remained analytically inside `[0,1]`, but its
localized affine residuals folded colour space. It fails the preregistered
coefficient, positive-Jacobian and non-negative diagonal-derivative gates.

No visual frontier, production integration, capacity increase or post-hoc
clamping opens.

## Reproducibility

| Item | Value |
|---|---|
| Software commit | `5f2c8e206757204fec5c1040beef7ff0847a32db` |
| Config SHA-256 | `5637368dccce1cf0b3952fa4e8460139b784437b261aaa76d75cd26bedf6c809` |
| Report SHA-256 | `f2cf7323c32daa4808906ca6f3d657c7770a0bc2a541194f1aef5f204ae5dd2a` |
| Repeated reports | byte-identical |
| Fit / confirmation points | `2,197 / 1,728`, disjoint midpoint grids |
| Jacobian points | `729` |
| Focused tests before run | `6 passed` |

The ignored reports are:

- `outputs/u5_r2k0_bounded_gaussian_residual_v1/report_pass1.json`
- `outputs/u5_r2k0_bounded_gaussian_residual_v1/report_pass2.json`

## Paper-formula negative control

The published GLUT equation with identity local affine functions and a zero
global branch was within `2.999e-7` of identity. Giving the unspecified global
branch identity initialization instead produced a maximum of `1.99999995`
before clamp, with `46.16%` of scalar outputs outside `[0,1]`.

This confirms the source-audit concern: the paper's stated initialization is
not sufficiently specified for an exact clean-room reproduction, and its final
clamp hides rather than prevents structural overflow.

## Candidate results

| Target | Capacity | Confirmation RMSE | Gain vs global affine | Max `abs(coef)` | Min det(J) | Min diagonal derivative |
|---|---:|---:|---:|---:|---:|---:|
| density cyan s0.50 | 8 | 0.02733 | — | 1.1486 | -0.2289 | 0.1864 |
| density cyan s0.50 | 27 | 0.00990 | 85.98% | 7.0855 | -3.3097 | -1.6951 |
| positive warm s0.35 | 8 | 0.01108 | — | 0.5804 | 0.2910 | 0.5474 |
| positive warm s0.35 | 27 | 0.00344 | 87.65% | 2.8649 | -1.8684 | -0.9351 |

Identity was exact at both capacities. All outputs were in range without a
clamp, maximum sampled Jacobian spectral norm remained below `8`, and
serialization, partition and strength-zero replay were exact.

Those successes do not override the severe structural failure. Low average
reconstruction error coexists with local reversals. In a colour renderer such
folds are a plausible mechanism for posterization, false contours and unstable
hue boundaries.

## Interpretation

K0 is useful negative evidence against treating compact LUT reconstruction
metrics as safety evidence. Gaussian locality substantially improves average
fit over one global affine operator, but the unconstrained local degrees of
freedom spend that capacity on orientation-reversing regions.

The next legal algorithm leaf must make orientation preservation structural.
A data-independent triangular monotone coupling representation may be audited;
it must not reuse K0's confirmation grid for threshold tuning and must not use a
conditional generator, real images or real-film pixels.

## Claim ceiling

This result is clean-room, data-independent negative representation evidence.
It is not an exact GLUT reproduction, learned film style, identified
digital-to-film operator, named stock response, calibration, authenticity,
preference result or production promotion.
