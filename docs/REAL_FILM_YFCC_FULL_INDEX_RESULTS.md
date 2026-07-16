# SF1.1 full-YFCC shared-author metadata results

Date: 2026-07-16

Node: `ULT > RF0.4 > SF1.1`

Decision: `open_bounded_live_rights_preflight`

## Evidence identity

- source: public YFCC100M metadata SQLite only;
- bytes: `65,644,027,904`;
- SQLite SHA-256: `dc3739758ce7a73f09c57cfae3c4d97a31a00a4695e525b1b82fd4fa37f83a1c`;
- expected S3 multipart ETag: verified with 7,826 parts of 8 MiB except the final part;
- download manifest SHA-256: `39563843b13aab0132584d5f80baf3954796c16a23ff11e5f07a49c2cf8b560c`;
- config SHA-256: `2bf377aaa8ef34deeefbef7e301fa275033c5f19b0d6d0fe21857dd28e402985`;
- audit A/B SHA-256: `19fd20b30b6724c1033f90ad2c3dafaabfeaf1dad528df12e2de77deaa3723c2`, byte-identical;
- decision SHA-256: `34254d047c56820286aec03dd08b178f9a30afa1dc2e479776ce3bf167549e85`;
- software commits: audit A used the pre-lock process image at `80f80bf`; audit B and the decision used `7fa5ed4`. The scan/filter implementation is identical; `7fa5ed4` adds only the process lock around it.

No image URL was requested and no image payload was downloaded or decoded.

## Results

The exact-text, single-stock-only and permissive-metadata-rights scan retains
2,441 rows. Three multi-stock comparison/listing rows are preserved as
ambiguous and excluded; zero exact-stock rows have a missing UID.

| Stock | Rows | UIDs | Largest UID share |
|---|---:|---:|---:|
| Kodak Ektar 100 | 780 | 122 | 7.18% |
| Fujifilm Velvia 50 | 240 | 62 | 11.25% |
| Kodak UltraMax 400 | 166 | 27 | 30.12% |
| Kodak Portra 160 | 368 | 59 | 15.49% |
| Kodak Portra 400 | 575 | 89 | 11.48% |
| Kodak Ektachrome E100VS | 287 | 41 | 14.29% |
| Kodak Gold 200 | 18 | 5 | 50.00% |
| Fujifilm Pro 400H | 7 | 5 | 42.86% |

`Ektar100–Velvia50` passes every frozen metadata gate: each has at least 100
rows and 30 UIDs, with 16 shared UIDs. The 16 shared authors contribute 154
exclusive candidate rows across the two stocks.

`Ektar100–UltraMax400` fails. It has 13 shared UIDs and enough rows, but
UltraMax has only 27 total UIDs versus the frozen minimum of 30. The threshold
is not changed after seeing the result.

## Decision and claim boundary

SF1.1 closes as a metadata feasibility pass for one edge. It opens only
`SF1.2`, a newly frozen bounded live-page rights preflight. SF1.2 must project
at least five authors with a currently rights-confirmed page for both stocks
before any later pixel scope may even be proposed.

This result does not authorise pixels, training, operator fitting, LSM mode
discovery, `S1/S2`, stock response, calibration, authenticity, released weights
or commercial claims. Metadata connectivity is necessary evidence, not stock
identifiability.

## Integrity recovery note

The first complete local byte stream failed the independent S3 multipart gate
after two downloader processes overlapped. Remote probes localized the affected
upper-middle region; a separately downloaded, exact-Range-validated `[47 GiB,
60 GiB)` repair restored the file. The entire 65.64GB object then passed SHA,
multipart ETag, SQLite schema and two independent scans. Download, repair and
audit CLIs now share a fail-closed atomic process lock.
