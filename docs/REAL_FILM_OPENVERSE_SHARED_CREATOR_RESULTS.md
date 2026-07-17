# SF2.1A Openverse exact-stock shared-creator results

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.1A`

Decision: **closed - query contract mismatch**

## Frozen execution

- config: `configs/real_film_openverse_shared_creator_v1.json`;
- config SHA-256: `5778fb7fb205b5062f0d1f60819d5a50e08bfa296c9d5f22d1dfc80497f929b5`;
- software commit: `2a608a5046ed3d58d33c8e764d8f90ef921ca427`;
- requests: 48 anonymous Openverse `/v1/images/` metadata searches, 12 pages
  of 20 results for each frozen exact phrase;
- thumbnail/detail/related/landing-page/image requests: zero;
- snapshot: `outputs/real_film/openverse_shared_creator_v1/metadata_snapshot.json`,
  SHA-256 `64b680825aea22835f2c60c94b145f37ea01a0175786859b0fcad2852363af95`;
- report: `outputs/real_film/openverse_shared_creator_v1/source_audit_report.json`,
  SHA-256 `0a8240e911c6fdd6bb2e47119dbe79235ad8a0f5e6542250e9253ccaf43100fc`;
- decision: `outputs/real_film/openverse_shared_creator_v1/decision.json`,
  SHA-256 `36bd2da1a43866349a6ac9e00de29af3a9a209342f426e9a009062eab9924b56`.

All responses passed endpoint, status, content-type and size checks. The
snapshot retains no raw JSON, image URL or thumbnail URL, and the offline audit
is repeat-identical.

## Decisive integrity failure

The Ektar100 pagination contains 19 repeated Openverse IDs, each with the same
repeated upstream landing URL. Its 240 returned rows therefore represent only
221 unique works. The other three queries have no within-stock repeat.

The frozen implementation treats within-stock identity repetition as a query
contract mismatch because otherwise unstable relevance pagination can inflate
row, creator and edge support. This check was committed and tested before the
formal query. The deterministic decision is therefore
`query_contract_mismatch`; post-result deduplication cannot reopen this leaf.

## Secondary support diagnostic

These values do not override the integrity failure:

| Stock | Returned rows | Strict rows | Strict creators | Largest creator share | Per-stock gate |
|---|---:|---:|---:|---:|---|
| Ektar 100 | 240 | 41 | 10 | 51.22% | fail dominance |
| Velvia 50 | 240 | 1 | 1 | 100.00% | fail support/diversity |
| Portra 400 | 240 | 43 | 11 | 62.79% | fail dominance |
| UltraMax 400 | 240 | 30 | 8 | 36.67% | pass |

Only one stock passes the independent per-stock gate, so there is no eligible
three-stock shared-creator component even before any live upstream label or
rights verification. Openverse remains useful for individual discovery, but
this relevance-search surface is not a stable comparative census and its
indexed licence claims are not upstream rights truth.

## Binding branch

- close SF2.1A and do not deduplicate after the result, loosen the licence set,
  lower creator/connectivity gates or add replacement queries;
- do not open an upstream landing-page preflight or request pixels;
- do not fit operators, train models, cluster latent modes or promote `S1/S2`;
- retain the snapshot as negative source/integrity evidence;
- continue Ultimate through another independently gated named-stock source or
  deterministic product leaf.

## Claim ceiling

Openverse search-pagination integrity, weak exact-stock text and strict licence
metadata diagnostics only. No upstream rights, physical-stock label, pixel
eligibility, content balance, stock identifiability, operator, latent mode,
calibration, authenticity or product claim.
