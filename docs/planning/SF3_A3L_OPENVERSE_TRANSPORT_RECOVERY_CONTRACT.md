# SF3.A3L Openverse transport-recovery contract

## Question

The frozen SF3.A2 client received HTTP 401 on 2026-08-21, while a read-only
2026-08-27 probe showed that the same anonymous Openverse endpoint is reachable
through PowerShell/curl and Python's standard-library HTTP stack. Does a
prospectively frozen standard-library transport expose enough *new* exact-stock
metadata to open a bounded live-rights/label preflight for Velvia 50, Portra
400 and Ektar 100?

## Frozen execution

- Reuse the exact SF3.A2 queries, aliases, page/request/response limits,
  retained fields, prior-identity exclusions and connectivity gates through
  the hash-bound base contract.
- Replace only the HTTP transport with `urllib`; require HTTPS,
  `api.openverse.org`, exact `/v1/images/`, JSON content type, no redirects and
  the same 2 MiB response ceiling.
- Run two complete forward/reverse stock-order acquisitions. Remove only
  request timestamps and rate-limit headers before scientific-payload replay
  comparison.
- Retain no raw response, image URL, thumbnail, image body, landing page or
  related object. Do not fit, train, score pixels or consume a candidate slot.

## Decision

Open only a separately frozen bounded upstream live-rights/label preflight if
the unchanged three-stock connectivity gate passes on rows absent from the
prior 941-identity snapshot. Otherwise close this refresh without changing
queries, aliases, pagination, thresholds, source roles or transport.

This leaf can supersede the old SF3.A2 **transport availability fact only**.
It cannot reinterpret the earlier source results or establish film-stock
response, calibration, authenticity, product readiness or multi-stock success.
