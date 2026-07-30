# P164 — unsupported-media batch atomicity evidence

## Result

All three frozen two-source cases pass twice:

| Invalid second source | Rejection | First-source encodes | Durable targets | Residue |
|---|---|---:|---|---:|
| transparent RGBA PNG | transparent hidden-RGB fallback refused | 1 / 1 | 8/8 hashes unchanged | 0 / 0 |
| two-page RGB TIFF | silent frame-zero fallback refused | 1 / 1 | 8/8 hashes unchanged | 0 / 0 |
| pinned libultrahdr MPO | existing multi-frame boundary | 1 / 1 | 8/8 hashes unchanged | 0 / 0 |

The encode observer proves that each valid first source reached the real
render/encode staging path before the invalid second source failed. Across the
six runs, all 24 pre-existing output/recipe/report hash checks remain exact.
No stage, backup or temporary file survives. Each case's normalized result
reproduces exactly.

The frozen config, runner and raw report SHA-256 identities are
`017b2abf...2a434`, `569c13d0...eeca7` and `db55efd8...d0cf2`.
The product `src/` tree is exact to candidate `f2ea6f7`.

## Interpretation

The existing transaction stages earlier results without publishing them and
cleans every stage when a later source fails media ingress. Existing
destinations are not partially replaced.

## Boundary

This is exact local consumer evidence for three fixtures. It neither extends
the decoder nor proves complete alpha, TIFF, MPO, Ultra HDR, gain-map or
ISO 21496-1 detection. It adds no HDR/media support, non-identity algorithm
promotion, target-platform runtime or product readiness claim.
