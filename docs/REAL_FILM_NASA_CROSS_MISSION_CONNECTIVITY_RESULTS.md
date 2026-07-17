# SF2.0C0 NASA/JSC cross-mission exact-stock connectivity results

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0C0`

Decision: **closed - no candidate mission**

## Frozen execution

- config: `configs/real_film_nasa_cross_mission_connectivity_v1.json`;
- config SHA-256: `fcefeb5ed5fcb7ebed80076e0e1cc9b191630441efa6bc3d0f66c0ee34612ce6`;
- software commit: `6a9d78444eacfcc16f554d5986b4d9dd29f45fc1`;
- requests: exactly three NASA/JSC `Technical.pl` POSTs plus three generated
  `ShowQueryResults-TextTable.pl` GETs;
- photo-page/image/API/mask/ZIP/KML requests: zero;
- report: `outputs/real_film/nasa_cross_mission_connectivity_v1/census_report.json`,
  SHA-256 `8e33c3961aca5857d4b29f7683258ec148fddb3544fe7638fbf881ddb20b08b5`;
- decision: `outputs/real_film/nasa_cross_mission_connectivity_v1/decision.json`,
  SHA-256 `379b97718ee7e23642f5be0346c4c91bb47d505cd02443c03ecfbf33c733f7e1`.

All six responses remained on the frozen NASA/JSC query/result endpoints and
passed status, content-type and size checks. There were no query errors or
cross-stock photo-ID overlaps. Raw HTML and frame rows were not retained; the
aggregate report reproduces the frozen decision exactly offline.

## Connectivity result

| Film code | Canonical stock | All-mission rows | Role |
|---|---|---:|---|
| `VELVI` | Fujifilm Velvia 50 | 13,255 | primary |
| `5775` | Kodak Portra 400NC | 397 | primary |
| `5776` | Kodak Portra 400VC | 122 | auxiliary only |

The census contains 25 missions, but only **STS098** contains rows for both
primary stocks. Its frozen aggregate remains:

| Stock | STS098 rows | Rolls | Supported rolls (at least 8 rows) | Required |
|---|---:|---:|---:|---:|
| Velvia 50 | 168 | 2 | 2 | 4 |
| Portra 400NC | 329 | 12 | 10 | 4 |

No other mission creates a Velvia50/Portra400NC edge. The auxiliary Portra
400VC code cannot rescue primary connectivity. Therefore the deterministic
decision is `no_candidate_mission`; the wider census confirms rather than
reopens the prior STS098 independent-roll failure.

## Binding branch

- close the NASA/JSC exact-code cross-mission expansion;
- do not lower the four-roll gate, substitute post-result film codes or reopen
  STS098;
- do not request photo pages or image payloads under this branch;
- do not fit operators, train models, cluster latent modes or promote `S1/S2`;
- retain the aggregate census as exact source-support negative evidence;
- continue Ultimate through another independently evidence-gated named-stock
  source or deterministic product leaf.

## Claim ceiling

NASA/JSC exact-film-code mission-by-roll metadata connectivity feasibility and
a negative candidate-mission decision only. No pixel quality, content balance,
stock identifiability, scan interpretation, operator, latent mode, calibration,
authenticity or product claim.
