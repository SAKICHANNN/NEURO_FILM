# SF0.4 Commons named-stock source audit results

## Decision

The immutable metadata snapshot passes the frozen source gate for three exact
community-category stock labels. A separately frozen bounded pixel pilot may
proceed. No image payload was downloaded or decoded by SF0.4.

Velvia also has strong source volume but remains a family-only control; it does
not count as Velvia 50 or as a fourth exact stock.

## Reproducibility

- snapshot fetcher commit: `0d3bad1d9bb888be426a5377036ce6d7f4049f63`;
- 4 categories / 658 current file rows;
- snapshot SHA-256 `f7ee9db16b198f3ce59c2cea887792e82f6dce8109ae6a79cb17c4ef7f10e5cd`;
- two offline audits are byte-identical;
- report SHA-256 `5b51fada9c6cfc01a32ecf32cbdb130135a4ed027e734c30b954319e0fc23a2b`;
- `image_payloads_downloaded_or_decoded=false`.

| Stock/category | Files | Uploaders | Largest share | Permissive rows | Decision |
|---|---:|---:|---:|---:|---|
| Fujifilm Superia X-TRA 400 | 133 | 13 | 35.34% | 105 | exact-stock metadata pass |
| Kodak Ektar 100 | 163 | 19 | 47.85% | 45 | exact-stock metadata pass |
| Kodak Gold 200 | 21 | 8 | 38.10% | 11 | exact-stock metadata pass |
| Fujifilm Velvia family | 341 | 59 | 54.25% | 75 | family control only |

All rows have an accepted free licence, file/original URL, current Commons SHA1
and dimensions. All exact-stock rows have minimum dimension at least 512 and
no frozen non-scene title pattern fired. Author metadata is present on every
exact-stock row. CC BY-SA rows remain internal-research-only pending legal
review; the next pilot can avoid that uncertainty by using only permissive
CC0/public-domain/CC BY rows with an explicit licence URL. The 18 Ektar rows
labelled only `Attribution` have no licence URL and are excluded from SF0.5;
the one Gold public-domain row may proceed with its explicit usage terms and
file page. This pixel-stage filter is stricter than the SF0.4 metadata gate.

## Evidence boundary and next branch

These are community-maintained categories, not manufacturer certificates,
edge-code reads or owned roll logs. Uploader and author/source controls remain
mandatory, and process/lab/scanner are unknown. Metadata pass is therefore
`S0 candidate` evidence, not `S1` or a stock response.

`SF0.5` may freeze uploader-capped 1600px derivative rows for the three exact
stocks only. It must preserve author/uploader/licence/source attribution,
verify downloaded hashes/decode, remove exact/perceptual duplicates, audit
content support and hold out whole uploader/source groups before any training.

Machine decision: `configs/real_film_commons_stock_source_decision_v1.json`.
