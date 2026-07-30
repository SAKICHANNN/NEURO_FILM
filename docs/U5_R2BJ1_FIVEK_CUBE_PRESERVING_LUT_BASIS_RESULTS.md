# U5.R2BJ1 — Cube-preserving adaptive LUT basis

BJ1 is closed. Two complete reports are byte-identical at
`69870b5c...083b4` with stable evidence `b52a56a1...0fc6b`.

The mathematical cube proof holds empirically: every method and population has
zero out-of-cube pixels, with no clipping or post-LUT gamut scaling. Adaptive
mean improvement over the global LUT remains 11.95--14.99%.

The stricter product gates nevertheless fail:

- new epsilon-boundary pixels reach 0.329%, 0.819% and 0.720%;
- AY3 adaptive P95 is `1.004049x` the global P95.

Zero out-of-cube is therefore not misreported as zero new near-boundary pixels.
BJ1 cannot be tuned or confirmed. One separately frozen strict-interior
operator may address the identified mathematical gap.

Claim ceiling: development-only digital-retouch mechanism evidence; no film,
stock, calibration, preference or product promotion.
