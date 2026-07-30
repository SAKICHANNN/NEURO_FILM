# Reference Color Match Empirical Prior Evidence

## Decision

`P19 = ROUTE CLOSED`.

An independently sampled neutral-photography population does not make global
RGB first/second moments identify an arbitrary reference look. The operator
remains constrained and batch-safe, but the empirical prior performs worse
than the fixed uniform-cube control. No product, commercial or visual-review
gate opens.

## Rights boundary

The official MIT-Adobe FiveK page states that the photos may be used for
research under one of two image licences. Both licence snapshots grant rights
solely for the user's own research and prohibit use directed toward commercial
advantage or monetary compensation.

Frozen 2026-07-27 byte identities:

| Official object | SHA-256 |
|---|---|
| `LicenseAdobe.txt` | `8121ecfcf743d850d6b6dbd0744cfb517c4cec189387aa010bf378077de84fcf` |
| `LicenseAdobeMIT.txt` | `69f73c481140c40d3a02a15a739da76073de35646108a70bb1dfffadb182af3d` |
| `filesAdobe.txt` | `9f4f262b25522d93f97a5b40f48f42de357ae0b14ef7098a576c912c546e8383` |
| `filesAdobeMIT.txt` | `c4cc6ef8cc9e519466ac130b977cd4292ffe9405f94d04227d0e2aee38aa5475` |

All 128 frozen source names have exactly one assignment: 59 Adobe and 69
Adobe+MIT. The experiment retains `research-only`,
`product_use_allowed=false`, and `commercial_use_allowed=false` as executable
validation fields. No interpretation of the numerical result can weaken that
boundary.

## Frozen data and ingest

- Dataset role: neutral-photo distribution control, never film evidence,
  aesthetic truth, target output or calibrated input.
- Freeze summary SHA-256: `45032415...c522b`.
- Freeze manifest SHA-256: `f820faf3...c174`.
- Selected column only: `raw_default_srgb16`.
- Rejected columns: every Expert C, filtered target, preview and gold source.
- Decode: uint16 display-sRGB to float64 display-linear linear-sRGB.
- Aggregation: select each image with equal weight, then select a pixel
  uniformly within that image.
- Inventory: 128 unique IDs, paths and byte hashes; single-page,
  orientation-1 uint16 RGB; 201,547,776 total pixels.
- No source pixels enter Git. The ignored artifact contains aggregate moments
  and source identities only.

Three independent builds are byte-identical:

- prior ID:
  `bb8238743da6ee6355c40a5414096943fc80d65a4b0fb1a1cf95b5b54bfdb0ae`;
- artifact SHA-256:
  `d590f75a4240e35aca73e0823ea0e0ad231e8d507505720f2fff8a9acb02895d`;
- mean linear RGB:
  `[0.1577828844, 0.1461626418, 0.1390916828]`;
- covariance eigenvalues:
  `[0.00089733, 0.00910803, 0.11202169]`.

## Preregistered experiment

The prior and all gates were frozen before inspecting either experimental
split. Development was diagnostic only and could not change confirmation.
Confirmation uses 24 previously unopened operators, two cross-content
observations each, across matrix-only, tone-only and combined families.

The explicit operator remains:

```text
empirical neutral-photo mean/covariance
  + uploaded reference mean/covariance
  -> unique symmetric positive-definite Gaussian transport
  -> boundary-pinned residual field
  -> constrained tetrahedral 17^3 LUT
  -> one fixed operator for every source in the batch
```

The confirmation gate required at least +3 points median gain over uniform,
at least 75% improvements, no more than 2 points worst-case loss, at most 5%
new boundary pixels, all constraints and zero identity fallbacks.

## Results

| Split/candidate | Improved | Median captured style | Worst | New boundary |
|---|---:|---:|---:|---:|
| Development uniform | 43/48 | +9.70% | -6.35% | 0% |
| Development empirical | 10/48 | -14.20% | -101.61% | 0% |
| Confirmation uniform | 31/48 | +4.63% | -30.84% | 0% |
| Confirmation empirical | 15/48 | -6.04% | -36.07% | 0% |

Confirmation decision:

- median gain over uniform: `-0.10672774857483686` — fail;
- empirical improvement rate: `0.3125` — fail;
- worst loss versus uniform: `0.052248431159932496` — fail;
- new-boundary, constraint and identity-fallback gates — pass;
- final status: `empirical-prior-route-closed`.

Two confirmation runs are byte-identical:

- report ID:
  `ca7e768dcce12deff9879410b942f44955b39e522f4cedfafcb605b8195acd0e`;
- report SHA-256:
  `2a9950966f93077fa739e2e333a8f56c81a17945127aeab2b6dd7c7fa36bf387`.

## Propagation

1. A1 does not open; default delivery remains identity.
2. FiveK pixels/statistics do not enter the product or commercial path.
3. Increasing the number of neutral photos is not an eligible rescue for the
   same first/second-moment estimator.
4. P14A and P19 together close analytic and empirical fixed global moment
   priors, not all perceptual matching.
5. The next experiment must change the information or objective: for example,
   correspondences supplied by the user, a commercially cleared learned
   canonical representation with a new falsifiable contract, or a bounded
   perceptual Look Approximation objective that does not claim hidden-operator
   recovery.
