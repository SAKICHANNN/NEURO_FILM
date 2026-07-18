# U1.6G4C Chunk-Invariant Global-Resample Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4C`

**Decision:** pass the independent numerical/field primitive; no effect integration

## Reproducibility

- implementation commit: `52dcb793e187e64d2921090b778d4dbb7d9c34ae`;
- audit commit: `a9249e52cff2461229482a52aafa5dd6448f9149`;
- operator: `shape-stable-global-resample-v2-explicit-f32`;
- config SHA-256: `157d586b9b1e32c57f791171bdca2f81d2ced0a56a402488cfb473ed6944e8a3`;
- ignored formal report: `outputs/u1_6g4c/formal_a9249e5_a.json`;
- report SHA-256: `e7b0dfa5534284f5b391419327a883be269d29ce7924b5300288fc80acd272dd`;
- contact sheet SHA-256: `2cdbca0c5f33f38de533092a484c3de893fc0784548ec2a95a395e99f672fe83`;
- visual adjudication config SHA-256: `e3e223d7240d85c3dfcfa30ccd97dbf2658b4e31abc40c46d6dda3939bed7479`;
- two complete audit executions produced byte-identical JSON and PNG artifacts.

The pre-run SHA transcription correction is recorded in the frozen contract
and agent log. It occurred before any report was written and changed no case,
algorithm, threshold or gate.

## Numerical result

All five frozen fields pass every row chunk and tile size:

- full-builder and bounded-row coarse bytes are identical;
- reconstructed full and bounded-row bytes are identical;
- full and tiled reconstruction bytes are identical;
- repeated stages and metadata are identical;
- every reader request remains below full source height;
- the frozen G1-v1 output remains SHA-256 `c3a59c15...`.

The three new-seed output hashes are:

| Case | Coarse shape | Output SHA-256 |
|---|---:|---|
| scalar `263x397`, sigma 52 | `32x49` | `8b5b033f6d7a8fb15c554559c1f3b6d33c64f5a141cef5afe79a2a99f7ff1e45` |
| scalar `197x281`, sigma `(41,69)` | `17x25` | `d9fb76716070cf66408597621bb340091dea52c43e278e7860fb38805f10e3d5` |
| RGB `89x113`, sigma 39 | `14x18` | `1252c485eb56238ea407a10b4d204332e05a9b49ee60411664577ba22ada359b` |

## Real-field compatibility

| Case | v1 status | v2-v1 max / mean | Rounded uint8 change | v2 row/full/tiled |
|---|---|---:|---:|---|
| `u41_01_luma`, `1204x1600` | available | `2.38e-7` / `8.65e-9` | 3 pixels (`1.56e-6`), max 1 code | byte-identical |
| `u41_11_luma`, `900x1600` | frozen last-cell failure | not defined | not defined | byte-identical |

The first field passes the frozen `5e-7` max, `5e-8` mean, `1e-5` changed
fraction and one-code ceilings. The second reproduces v1's documented
`shape-mismatch for sum`; v2 completes and remains chunk/tile invariant.

## Autonomous visual field review

The two fixed v2 luma fields are continuous and smooth with no confirmed seam,
band, block or boundary failure. Normalized row-minus-full panels are black
because their numerical difference is zero. The available normalized
v1-minus-v2 panel shows sparse, unstructured least-significant-bit noise with
no tile or boundary alignment.

This is an intermediate-field review, not final-RGB or physical-effect visual
validation. It cannot promote physical/density halation by itself.

## Verification

- 25 focused v1/v2 tests pass;
- 505 complete CPU tests pass in 22.11 seconds;
- invalid version/geometry/shape/chunk/reader data and injected failure close;
- existing v1 functions, current effects, renderer, profiles and CLI remain
  unconnected to v2.

## Decision and next branch

G4C passes as an independent versioned primitive. Bind the real v2 capability
to the G3 static readiness audit, then freeze the smallest density-family
staged executor as a separate leaf. That executor requires its own effect
parity, full-resolution visual/severe-artifact and resource gates before any
renderer integration.

## Claim ceiling

This proves chunk-invariant explicit-float32 global staging and limited
real-field compatibility. It does not prove effect parity, physical accuracy,
renderer integration, streaming decode, bounded total memory, stock/calibrated
response or 100MP readiness.
