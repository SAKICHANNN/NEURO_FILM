# Real-film stock pilot integrity and support audit

Date: 2026-07-16

Node: `ULT > RF0.4 > SF0.3`

Decision: integrity/support audit passed; no colour fit performed

## Evidence

At software commit `c952309bbee4da51f3849e155c49ddca2706dac3`, two complete
audit executions produced byte-identical reports:

| Check | Result |
|---|---:|
| Decoded files | 189 / 189 |
| Decode failures | 0 |
| Exact SHA duplicate groups | 0 |
| Cross-frame dHash≤4 pairs | 0 |
| Same-frame preview/proxy dHash≤4 pairs | 3 |
| Official-test roll overlap | 0 |
| Colour fit performed | no |
| Border/interior masks | not implemented |
| Contact sheets | 8 (review-only) |

Report:

- path: `outputs/real_film/stock_pilots_v1/integrity_audit/report.json`
- SHA-256: `0e09c90106824425ceaa57dead73155229c35997ae933db55f88420c70e93f0a`
- all payloads decode as RGB PNG; ICC profile bytes are zero on every file

The three near-duplicate pairs are same-frame negative-preview versus display
proxy siblings. No cross-frame perceptual near-duplicate was found at dHash≤4.

## Per-stock eligibility before RF1.4

| Stock | Rolls | Previews | Proxies | Structural content diversity | Decision |
|---|---:|---:|---:|---|---|
| `kodak_gold_100_gen5` | 7 | 50 | 47 | yes | display-operator research candidate |
| `fujifilm_nph_400` | 4 | 53 | 0 | yes | density/metadata identifiability only |
| `konica_super_xg_100` | 7 | 22 | 1 | weak (one dominant day/outdoor cell) | density/metadata identifiability only |
| `kodak_ga_100_5095` | 3 | 16 | 0 | weak (one supported content cell) | density/metadata identifiability only |

Display-operator eligibility follows the frozen SF0.1
`display_operator_candidate` metadata, not raw proxy counts. Konica's single
proxy therefore does not open a display lane.

Contact sheets live under
`outputs/real_film/stock_pilots_v1/integrity_audit/contact_sheets/` and are
marked `pending Codex vision adjudication`. Cursor does not claim visual
pass/fail on borders, base, captions or severe artifacts.

## Claim ceiling

This leaf only decides decode integrity and provisional density/display
research eligibility. It does not establish stock signal, `S2` transfer,
calibration, authenticity or release clearance. Deterministic border masks
remain unimplemented.

Machine-readable decision:
`configs/real_film_stock_pilot_integrity_decision.json`.

## Next leaf

`RF1.4`: freeze and run per-stock leave-one-roll-out identifiability against
pooled, wrong-stock, retrieval, historical, shuffled-label and simple
enhancement controls. Keep density-domain and display-proxy domains separate.
No GPU training.
