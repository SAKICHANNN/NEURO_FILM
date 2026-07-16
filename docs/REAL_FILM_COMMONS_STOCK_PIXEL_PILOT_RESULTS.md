# SF0.5 Commons named-stock pixel pilot results

## Decision

The derivative-only pilot is decode/duplicate/visual clean, but only Ektar100
passes the frozen source-group gate. One stock cannot identify a stock-specific
signal against content, author and scanner nuisance, so no learning starts.
Superia and Gold stop; `SF0.6` must find at least two additional independent
exact-stock sources before a comparative identifiability experiment.

## Reproducibility and integrity

- implementation/report commit: `4754978d45db8d8ae344009f5e0ac5e0fa3a078c`;
- corrected selection: 36 rows, SHA-256 `174aee331a07a5d206fd6586b57b723789eacfbd7864dd2df939e197642451f4`;
- download: 36 1600px derivatives / 27,812,816 bytes, manifest SHA-256 `3b4bdb465b0b990177bdd36197d6aeeeee73460ee01e4dcf3adffa7091ef9614`;
- two byte-identical offline audits, report SHA-256 `93d6a218c4b9342bb6a1bd0e08cbb0ca41b941a19fbdc6ad9b9ef380e3cf48b0`;
- 36/36 decode and hash checks pass; zero exact duplicates, zero dHash<=4
  pairs and zero cross-stock collisions;
- four contact sheets plus the maximum-white, maximum-black and Gold content
  risk cases were reviewed at full resolution.

| Stock | Files | Author groups | Largest author | Source gate | Decision |
|---|---:|---:|---:|---|---|
| Kodak Ektar 100 | 26 | 8 | 30.77% | pass | provisional unpaired positive reference only |
| Fujifilm Superia X-TRA 400 | 2 | 2 | 50.00% | fail | insufficient derivative and author support |
| Kodak Gold 200 | 8 | 4 | 62.50% | fail | author dominance plus repeated content cluster |

## Visual adjudication

No image shows confirmed red-speckle blocks, posterization, seams, geometry
damage, repeated synthetic texture or destructive scan corruption. Ektar has
real clipped skies, deep silhouettes, vignetting and varied grain; these are
source properties rather than renderer artifacts. Gold includes several
same-author photographs of damaged objects on patterned flooring, making the
content/source shortcut visually obvious. Clean pixels therefore do not cure
the identifiability failure.

The earlier interrupted attempt downloaded five files before the corrected
derivative-only rule was applied. They are preserved under ignored quarantine,
not included in any manifest, audit or learning set. Three had API thumbnail
URLs equal to original URLs; the corrected selector forbids that equality.

## Claim boundary and next branch

Ektar remains `S0` community-category evidence. It is not a measured stock
response, physical-roll result, calibrated target or proof that the look will
transfer to digital photographs. With only one passing stock, even a visually
successful model could simply learn author/content/scanner averages.

`SF0.6` must obtain at least two further exact stocks with explicit derivative
rights, at least five independent author/source groups, no >60% group, and
usable content diversity. Until then, training is forbidden.

Machine decision: `configs/real_film_commons_stock_pixel_decision_v1.json`.
