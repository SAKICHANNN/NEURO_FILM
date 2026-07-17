# SF2.0B0 NASA/JSC STS098 exact-stock metadata snapshot results

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0B0`

Decision: **closed — insufficient independent-roll support**

## Frozen execution

- config: `configs/real_film_nasa_sts098_stock_snapshot_v1.json`;
- software commit: `e973be41f280d0c2b1534ad47fde707b4d3d5efb`;
- requests: exactly 3 NASA/JSC `Technical.pl` POSTs plus 3 generated
  `ShowQueryResults-TextTable.pl` GETs;
- photo-page/image/API requests: zero;
- report: `outputs/real_film/nasa_sts098_stock_snapshot_v1/snapshot_report.json`,
  SHA-256 `8c373de39d0c0a9ab03e991608d93c3f071b781b2910ee20ab6c1c126934ff43`;
- decision: `outputs/real_film/nasa_sts098_stock_snapshot_v1/decision.json`,
  SHA-256 `759867b9c73c15078089049a6d44ff81b720fb01a2ef03c990de6d51bef52f8a`.

All six responses remain on the frozen NASA/JSC query/result endpoints. The
nine required columns parse for all queries, there are no query errors, no
cross-stock photo-ID overlaps and offline decision replay is exact.

## Support result

| Film code | Canonical stock | All-mission rows | STS098 rows | STS098 rolls | Frozen minimum | Result |
|---|---|---:|---:|---:|---:|---|
| `VELVI` | Fujifilm Velvia 50 | 13,255 | 168 | 2 (`701`, `720A`) | 150 rows / 4 rolls | fail |
| `5775` | Kodak Portra 400NC | 397 | 329 | 12 | 300 rows / 4 rolls | pass |
| `5776` | Kodak Portra 400VC | 122 | 10 | 1 (`373`) | 8 rows / 1 roll, auxiliary | pass |

The frame count made the source look large, but the primary Velvia arm has only
two independent film rolls. The frozen source-feasibility contract requires at
least four rolls for each primary stock. Therefore the deterministic decision
is `insufficient_stock_roll_support`.

## Interpretation and branch

STS098 is a useful negative example of why frames are not independent evidence.
The same-mission connection is real, and Portra 400NC has substantial roll
support, but Velvia's 168 frames cannot supply the preregistered group structure
needed for a two-stock nuisance audit. Portra 400VC remains a one-roll auxiliary
and cannot repair the primary edge.

Binding branch:

- close the STS098 VELVI/5775/5776 edge and do not open SF2.0B1;
- do not lower the four-roll minimum or substitute another film code after
  seeing this result;
- do not request photo pages or image payloads for this branch;
- do not fit operators, train models, cluster latent modes or promote `S1/S2`;
- retain the exact snapshot as authoritative source-support negative evidence.

## Claim ceiling

Exact NASA/JSC media-code, mission, roll and table-metadata support plus a
negative independent-roll decision. No stock identifiability, pixel quality,
scan interpretation, operator, latent mode, calibration, authenticity or
product claim.
