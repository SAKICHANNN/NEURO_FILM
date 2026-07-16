# Real-film stock pilot integrity and support audit

Date: 2026-07-16

Node: `ULT > RF0.4 > SF0.3`

Decision: integrity/support audit passed; no colour fit performed

## Evidence

The original Cursor report was accepted only after hardening manifest/metadata
cross-checks and correcting the preview-domain claim. At software commit
`3c52a31a2632a18e3fff03ee5da8c11eef07444e`, two complete audit executions
produced byte-identical reports:

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
- schema: v2
- SHA-256: `a93257ee45e14ac0519dc1ce76a840ebe4f517a8d5c413e33b17c915160d1caa`
- all payloads decode as RGB PNG; ICC profile bytes are zero on every file

The three near-duplicate pairs are same-frame negative-preview versus display
proxy siblings. No cross-frame perceptual near-duplicate was found at dHash≤4.

## Per-stock eligibility before RF1.4

| Stock | Rolls | Previews | Proxies | Structural content diversity | Decision |
|---|---:|---:|---:|---|---|
| `kodak_gold_100_gen5` | 7 | 50 | 47 | yes | display-operator research candidate |
| `fujifilm_nph_400` | 4 | 53 | 0 | nominally yes, visually confounded | post-negation preview identifiability only |
| `konica_super_xg_100` | 7 | 22 | 1 | weak (one dominant day/outdoor cell) | post-negation preview identifiability only |
| `kodak_ga_100_5095` | 3 | 16 | 0 | weak; 13/1/2 frames by roll | post-negation preview identifiability only |

Display-operator eligibility follows the frozen SF0.1
`display_operator_candidate` metadata, not raw proxy counts. Konica's single
proxy therefore does not open a display lane.

BlueNeg's local README explicitly defines `negative-preview-8bit` as a
"Negative preview (after negation)". These positive-looking 8-bit previews are
not verified physical density. They may support a fail-closed label/shortcut
diagnostic, but cannot establish a density response or train a display-colour
operator for the three stocks without a valid proxy lane.

All eight contact sheets under
`outputs/real_film/stock_pilots_v1/integrity_audit/contact_sheets/` are now
vision-reviewed. No severe glitch was confirmed at contact-sheet scale, but
date imprints, dark borders/crops, exposure/scan variation and strong
roll-to-location/content coupling are visible. Full-resolution artifact review
was not performed. See
`configs/real_film_stock_pilot_visual_decision.json`.

## Claim ceiling

This leaf only decides decode integrity and provisional preview/display
research eligibility. It does not establish stock signal, `S2` transfer,
calibration, authenticity or release clearance. Deterministic border masks
remain unimplemented.

Machine-readable decision:
`configs/real_film_stock_pilot_integrity_decision.json`.

## Next leaf

`RF1.4`: first freeze shortcut/null and crop/date/border sensitivity gates,
then run per-stock leave-one-roll-out identifiability against
pooled, wrong-stock, retrieval, historical, shuffled-label and simple
enhancement controls. Keep post-negation preview descriptors and display-proxy
targets separate. A preview-only result cannot promote a colour expert.
No GPU training.
