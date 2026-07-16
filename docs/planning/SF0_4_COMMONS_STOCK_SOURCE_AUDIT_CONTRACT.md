# SF0.4 Wikimedia Commons named-stock source audit contract

## Why this leaf exists

BlueNeg identifies a repeatable archive display chain but not a useful digital
film look. The next test needs stock-labelled positive scans from multiple
independent sources. Wikimedia Commons exposes candidate categories plus
per-file author, uploader, source URL, dimensions, SHA1 and licence metadata
through its public API.

This stage fetches metadata only. It does not download or decode image pixels.

## Frozen candidates

| Candidate | Current files | Uploaders | Label ceiling |
|---|---:|---:|---|
| Kodak Ektar 100 | 163 | 19 | exact stock, community category |
| Fujifilm Superia X-TRA 400 | 133 | 13 | exact stock, community category |
| Kodak Gold 200 | 21 | 8 | exact stock, community category |
| Fujifilm Velvia | 341 | 59 | family only; cannot count as Velvia 50 |

Counts are discovery observations dated 2026-07-16, not hashes. The fetcher
must snapshot category revision IDs/timestamps and every returned row, then all
scientific audits rerun offline from that immutable snapshot. Each category
uses at most two API requests: one category revision/count request and one
single-page file/imageinfo request. A future 1600px derivative URL may be
recorded, but no image bytes are fetched here.

Primary pages:

- `https://commons.wikimedia.org/wiki/Category:Photographs_taken_on_Kodak_Ektar_100_film`
- `https://commons.wikimedia.org/wiki/Category:Taken_on_Fuji_Superia_X-TRA_400`
- `https://commons.wikimedia.org/wiki/Category:Photographs_taken_on_Kodak_Gold_200_film`
- `https://commons.wikimedia.org/wiki/Category:Taken_on_Fuji_Velvia`
- licence/reuse guidance: `https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia`

## Rights and group policy

Every row retains the file page, original URL, author/credit, uploader, licence
name/URL, SHA1 and dimensions. CC BY-SA rows may support internal research but
remain outside any commercial/released-weight lane until legal review. A
separate permissive subset includes CC0, public-domain and attribution-only
licences.

Uploader is a leakage group, not automatically the author. Bots never become
author groups. Author/source normalization is deferred until the raw snapshot
exists and must preserve both raw and normalized strings.

## Gates before any pixel pilot

Each exact stock needs at least 20 files, five uploaders, no uploader above 60%,
eight permissive rows, 95% at minimum dimension 512, no more than 10% obvious
packaging/strip/grain-test titles and complete free-licence/source/SHA1/size
metadata. Velvia-family is reported separately and cannot satisfy the
three-exact-stock condition.

Only if all three exact stocks pass may a later contract download at most 192
1600px derivatives/512MiB, capped at 12 rows per author/uploader group/stock.
Pixels still require decode, duplicate, content, scanner/processing nuisance
and group-held-out audits before training.

## Claim boundary

Passing this audit means metadata eligibility for a bounded unpaired research
pilot. Community category labels are weaker than edge-code/manufacturer or
owned roll records. There is no physical-roll, process, lab, scanner, stock
response, `S1/S2`, authenticity, training or release claim.

Machine contract: `configs/real_film_commons_stock_source_audit_v1.json`.
