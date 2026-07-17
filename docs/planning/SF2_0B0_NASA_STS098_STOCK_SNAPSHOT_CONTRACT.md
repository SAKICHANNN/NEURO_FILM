# SF2.0B0 NASA/JSC STS098 exact-stock metadata snapshot contract

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0B0`

Status: frozen before bounded query execution

## Question

Can the public, keyless NASA/JSC astronaut-photography search surface produce
a reproducible STS098 metadata snapshot for Velvia 50, Portra 400NC and Portra
400VC with enough independent film-roll support to design a separate
stock/content/source nuisance audit?

This is discovery and source-integrity work. It does not test whether stock is
identifiable and cannot authorise pixels or learning.

## Parent evidence and rationale

- SF1.3B closes current community pixels because nuisance controls beat RGB;
- SF2.0A closes Apollo 7 because its authoritative stock labels still have
  content/magazine structural zeros;
- NASA/JSC's official database defines exact media codes, mission-roll-frame
  identity, film roll, camera, geographic/features and focal-length metadata;
- bounded reconnaissance found STS098 support for codes `VELVI`, `5775` and
  `5776`, but those exploratory counts are not frozen evidence;
- the public HTML search needs no API key or external contact;
- NASA/JSC permits use subject to source credit, non-endorsement and third-party
  caveats. SF2.0B0 accesses metadata only, not photographs.

## Frozen access

Run one exact-code query for each:

| Code | Canonical ID | Role | Reconnaissance-informed minimum |
|---|---|---|---:|
| `VELVI` | `fujifilm_velvia_50` | primary | 150 STS098 rows / 4 rolls |
| `5775` | `kodak_portra_400nc` | primary | 300 STS098 rows / 4 rolls |
| `5776` | `kodak_portra_400vc` | auxiliary | 8 STS098 rows / 1 roll |

Each query is one POST to `Technical.pl` followed only by its generated NASA/JSC
result-table GET. Maximum: three POSTs, three GETs, six requests total. Parse
only the frozen table columns and retain target-mission rows plus response
hash/size/time and aggregate counts. Do not retain raw HTML.

Do not request any `photo.pl` link, image, cloud mask, ZIP, KML or API endpoint.
One exact code per query is mandatory so the result table's rows inherit an
unambiguous query label. Cross-stock photo-ID overlap must be zero.

## Branches

- **Snapshot complete:** all three queries meet table/integrity, target-row,
  roll-support and zero-overlap gates. Freeze the snapshot and separately
  preregister SF2.0B1 content/date/focal/roll nuisance connectivity. Still no
  pixels, training, operator fitting or LSM.
- **Insufficient support:** close the unsupported code/edge; do not compensate
  by adding a different code after reading results.
- **Cross-stock overlap:** mark query semantics or source metadata ambiguous and
  close before connectivity analysis.
- **Contract mismatch/source unavailable:** preserve response hashes/errors and
  stop. Do not fall back to API-key requests or external contact.

## DoD and evidence bundle

- committed config and implementation commit/hash;
- exactly six or fewer request records with UTC, status, final URL, bytes and
  bounded-response SHA-256;
- exact code-specific all-mission count and STS098 row/roll manifest;
- frozen table headers and zero cross-code photo-ID overlap;
- deterministic decision and report hashes;
- explicit no-photo-page/no-image/no-fit/no-train/no-LSM booleans;
- targeted tests, full CPU suite, propagation, scoped commit and push.

## Claim ceiling

At most: exact-code NASA/JSC STS098 metadata support suitable for designing a
later nuisance audit. No stock signal, scan interpretation, digital-to-film
operator, latent mode, `S1/S2`, calibration, authenticity or product claim.
