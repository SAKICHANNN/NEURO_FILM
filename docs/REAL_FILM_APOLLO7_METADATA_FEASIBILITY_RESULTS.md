# SF2.0A Apollo 7 stock/magazine metadata feasibility results

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0A`

Decision: **closed — content confounded**

## Frozen execution

- config: `configs/real_film_apollo7_metadata_feasibility_v1.json`;
- software commit: `4a5bc0856382ac1139295bce54363005d24657b0`;
- sample: nine deterministic frames from each of seven colour magazines;
- access: 63 sequential NASA/JSC `photo.pl` HTML requests;
- image requests: zero;
- report: `outputs/real_film/apollo7_metadata_feasibility_v1/audit_report.json`,
  SHA-256 `afb8b150df3b04e99176dacc58b92d4922de0da2031178cb74b29f56b4773ee9`;
- decision: `outputs/real_film/apollo7_metadata_feasibility_v1/decision.json`,
  SHA-256 `b63e2dcbd7cef3a72070459a6399a681e941ed10c1fceb11513e3ac10561eba7`.

The first command wrapper reached its 244-second timeout, but the single child
audit process remained active and completed atomically. Process inspection
confirmed that no second audit was started. The final artifacts were written
once at 2026-07-17 11:12:53 local time.

## Integrity and support

All 63 pages returned HTTP 200 `text/html`, remained on the NASA/JSC
`photo.pl` endpoint, stayed below 22,953 HTML bytes, matched the expected NASA
photo ID and matched the expected stock code. Offline decision replay is equal
to the frozen decision.

| Evidence | SO-368 | SO-121 | Gate |
|---|---:|---:|---|
| valid pages | 18 | 45 | pass |
| independent magazines | 2 (M, N) | 5 (O, P, Q, R, S) | pass |
| filter-free rows | 18 | 9 | pass |

Every magazine contributes 9/9 valid pages. Reported exposure states are
`Normal`, `Over Exposed` and `Under Exposed`; one page has unknown exposure.
This passes support and the explicit filter-free bridge, but it does not make
exposure a supervised physical parameter.

## Decisive content gate

The frozen gate required at least two shared content tags, with each tag
supported by at least three rows and two independent magazines per stock.

| Tag | SO-368 rows / magazines | SO-121 rows / magazines | Shared gate |
|---|---:|---:|---:|
| `cloud_weather` | 3 / 1 | 6 / 4 | fail |
| `land_terrain` | 2 / 1 | 3 / 2 | fail |
| `ocean_water` | 3 / 1 | 18 / 5 | fail |
| `spacecraft_hardware` | 4 / 2 | 6 / 4 | pass |
| `sun_glint` | 1 / 1 | 1 / 1 | fail |

Only `spacecraft_hardware` passes. The second shared tag does not exist under
the preregistered sample and support rules, so the deterministic decision is
`content_confounded`. The thresholds and tags were not changed after seeing
the result.

## Interpretation and branch

Apollo 7 is valuable observed evidence that authoritative stock and magazine
labels alone are insufficient. SO-368's Earth/ocean/weather support is largely
concentrated in magazine M, so stock and content/magazine structure cannot be
separated by this source design. The filter-free bridge and multiple exposure
states do not repair that structural zero.

Binding branch:

- close SF2.0A and do not expand Apollo 7 page or pixel acquisition;
- retain Apollo as a narrow archive/stress and source-design negative control;
- do not fit an operator, train a model, cluster modes or promote `S1/S2`;
- do not treat offered image links as acquired pixels;
- continue Ultimate through another evidence-authorised stock/data source or
  the independent deterministic product path.

## Claim ceiling

Authoritative Apollo 7 stock/magazine metadata plus negative connectivity
evidence only. No pixel quality, stock response, digital-to-film operator,
latent mode, calibration, training, authenticity or product claim is allowed.
