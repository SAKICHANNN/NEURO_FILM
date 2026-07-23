# U5.R2A constrained global colour-operator results

Date: 2026-07-23

Decision: **pass the reusable numerical representation; do not infer semantic
safety or film authenticity**

## Implemented representation

`src/roll2film/constrained.py` composes a bounded affine stage, strictly
monotone per-channel splines and a versioned dense 3D LUT. The LUT supports
explicit trilinear or exact tetrahedral interpolation, deterministic
serialization and an audit over identity-relative residual amplitude,
first/second residual differences, neutral-axis error, output range and every
per-tetrahedron Jacobian determinant.

The boundary fails closed on invalid shape, dtype, finite state, domain,
headroom or constraint reports. It does not clip implicitly and it does not
mutate the caller input. No renderer, profile, recipe schema or production
default is connected to this research primitive.

## Reproducible evidence

- contract config SHA-256:
  `8b7934878655866bb97355459d9732f51416a063eb5ce39831154a31eefb6a3d`;
- implementation commit:
  `cf984cbb7dad6c68c93d9c5054a946e20cfb0300`;
- two independent audit reports are byte-identical at SHA-256:
  `481a3cb90b6f4ded0314cd1815859d165dcc3ea3245e123513881478f1aa81ca`;
- identity and serialization replay maximum absolute error: `0`;
- monotone-curve roundtrip maximum absolute error:
  `2.220446049250313e-16`;
- minimum audited curve derivative: `0.7340090296361981`;
- vectorized tetrahedral interpolation versus an independent scalar reference
  maximum absolute error: `0`;
- 15 focused property tests, 63 Roll2Film/adjacent tests and 704 complete CPU
  tests pass; compile and diff checks pass.

The ignored machine reports remain under
`outputs/u5_r2a_constrained_global_operator_v1/`.

## Required non-safety counterexample

The frozen smooth blue-to-purple counterexample is in gamut, neutral
preserving and orientation preserving:

- input `[0.1, 0.25, 0.85]`;
- output `[0.3625, 0.25, 0.64]`;
- red-channel increase `0.2625`;
- maximum LUT residual amplitude `0.35`;
- maximum residual first-axis step `0.021875`;
- maximum residual second difference `2.22e-16`;
- tetrahedron Jacobian determinant approximately `0.4225`;
- output range `[0, 1]` and neutral-axis error `0`.

It passes every frozen numerical constraint while intentionally making a large
semantic colour change. Therefore monotonicity, smoothness, gamut bounds and a
positive Jacobian are implementation guardrails, not evidence of semantic
colour safety. The independent FilmStyleSafe severe-artifact veto remains
mandatory.

## Branch decision and claim ceiling

Retain the representation as the reusable U5.R2A primitive and open U5.R2B,
which must compare eligible strong global candidates under identical
renderer/export, target-look and severe-artifact evidence. Do not fit current
closed stock pixels, train a router, start LSM, integrate this primitive into
production or claim an identified digital-to-film, stock-authentic or
calibrated transform.
