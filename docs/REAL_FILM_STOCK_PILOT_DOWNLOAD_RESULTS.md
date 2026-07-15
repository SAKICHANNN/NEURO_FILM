# Real-film stock pilot download results

Date: 2026-07-16

Node: `ULT > RF0.4 > SF0.2`

Decision: exact frozen four-stock download verified

## Evidence

At software commit `873376443c3cf6be6f5db292fcf062f1bacf523f`, two complete
download executions produced byte-identical reports:

| Check | Result |
|---|---:|
| Exact revision | `b038a1ae68f42067ff12b5e79ddbe62919b7af23` |
| Frozen files | 189 |
| Verified files | 189 |
| Frozen bytes | 227,287,697 |
| Verified bytes | 227,287,697 |
| LFS SHA-256 failures | 0 |
| Manifest-external lane files | 0 |
| Image payloads decoded | no |
| 16-bit archive downloaded | no |

Deterministic report:

- path: `outputs/real_film/stock_pilots_v1/download_report.json`
- SHA-256: `de286950dd78970281b592ffd1975f26db9b2a56565d92d84043bf3eafb348b3`
- acquisition manifest SHA-256:
  `1ac2dfce6060d0094d221e8776f4318ebdd144a86194c534daca01ae3749324e`
- metadata report SHA-256:
  `9ba954cffd5f47839fcb14fe7b025bb8be27ca48256baf7b579dce4c21e756d3`

Isolated download root: `data/raw/blueneg_stock_pilots_v1` (gitignored).
The previous BlueNeg pilot under `data/raw/blueneg` remained untouched
(213 local files counted after verification). FSA/OWI partial cache was not
modified.

## Claim ceiling

This leaf establishes only that the frozen 189 preview/proxy objects exist
locally with matching sizes and LFS SHA-256 hashes. It does not establish
stock signal, display-colour expertise, `S2` transfer, calibration,
authenticity or release clearance. Pixels remain undecoded for scientific
work until `SF0.3`.

Machine-readable decision:
`configs/real_film_stock_pilot_download_decision.json`.

## Next leaf

`SF0.3`: decode integrity, exact/perceptual duplicates, border/content and
stock × roll × content × proxy-availability support audits before any RF1.4
identifiability fit. No colour expert training and no GPU job are justified.
