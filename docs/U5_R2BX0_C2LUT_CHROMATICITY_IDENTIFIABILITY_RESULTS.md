# U5.R2BX0 chromaticity-conditioned explicit LUT result

The newest primary mechanism screened was C2LUT (Rota et al., arXiv:2607.11681,
2026). Its official repository was still a placeholder at the frozen commit,
so this leaf is a clean-room synthetic mechanism test rather than a
reproduction.

The rank-4 convex explicit LUT passed every capacity and cube-safety gate on
held injective descriptors: maximum RMSE was `4.30e-17`, it effectively removed
the global/affine errors, and new-boundary fraction was zero. This shows that a
compact source-conditioned explicit operator can represent the frozen
illuminant-varying normalization family.

The same-chromaticity hidden-spectrum control was not strong enough for the
pre-registered identifiability test. Its target-pair RMSE was `0.018349`, below
the frozen `0.02` minimum, although the descriptor-only error was `0.009175`
and the hidden-coordinate oracle was exact. The result therefore closes as
`retain_capacity_close_underpowered_metamer_fixture`: no amplitude or threshold
rescue is allowed. A new physical spectral-integration fixture is required.

Two formal reports are byte-identical at SHA-256
`fa17cc74bde6fa7242fd7057b603cd65f171fabd709252213ad2b726c5a501ef`.
This is synthetic source-normalization evidence only, not camera calibration,
film/stock response, photographic quality or product promotion.

Primary source: <https://arxiv.org/abs/2607.11681>

