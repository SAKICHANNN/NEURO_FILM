# SF1.2 YFCC shared-author live-rights feasibility contract

Date: 2026-07-16

Node: `ULT > RF0.4 > SF1.2`

Status: frozen before live-page access

## Question

Do at least five of the 16 metadata-connected Ektar100/Velvia50 Flickr UIDs
still expose at least one live CC BY 2.0 page for each stock?

This tests current rights feasibility only. It does not test stock signal,
content balance, pixel quality, operator identifiability or latent modes.

## DoR

- SF1.1 source SHA and S3 multipart ETag pass;
- audit A/B are byte-identical at SHA-256 `19fd20b3...3723c2`;
- deterministic SF1.1 decision opens only bounded live-rights preflight;
- 16 shared UIDs and 154 exclusive rows are frozen;
- no image request, pixel scope, operator fit or model work is allowed.

## Selection and access

For each shared UID, visit stocks in fixed order Ektar100 then Velvia50. Within
each UID/stock arm, order by numeric `photoid`, apply the prospective process /
HDR / multiple-exposure / B&W contamination exclusions, and request at most
four pages. Stop that arm at the first page that returns HTTP 200, `text/html`
within 2 MiB, and contains the CC BY 2.0 licence URL. Page URL, final URL,
status, content type, bounded body hash/bytes and decision are retained.

Maximum requests are 16 authors × 2 stocks × 4 pages = 128. Requests are
sequential with a one-second interval and bounded retry/backoff. No image or
download URL may be requested, and no HTML body is retained.

## Gate and branches

A usable shared author has at least one live-rights-confirmed page in both
stocks.

- **Pass:** at least five usable shared authors. Freeze exact retained page IDs,
  rights timestamps and a separate pixel preflight proposal. This pass itself
  still leaves pixel download and operator fitting false.
- **Fail:** fewer than five usable shared authors. Close this public
  shared-author expansion; retain SF1.1 as metadata evidence and continue a
  different evidence-authorised data/product leaf.

Missing, changed, non-200, non-HTML, oversize or ambiguous pages fail that
candidate. Do not substitute a new author, loosen the minimum or request image
payloads after seeing the result.

## DoD and evidence bundle

- input report/decision hashes and config hash;
- software commit and request timestamp;
- every bounded page attempt and final URL/status/licence result;
- per-author/per-stock pass matrix;
- usable shared-author count and frozen pass/fail decision;
- explicit `image_payload_download_allowed=false` and
  `operator_fitting_allowed=false`;
- targeted tests, full CPU suite, report SHA, propagation commit and push.
