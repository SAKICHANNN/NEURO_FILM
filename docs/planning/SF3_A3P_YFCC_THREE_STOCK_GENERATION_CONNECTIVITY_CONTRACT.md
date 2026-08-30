# SF3.A3P — YFCC three-stock generation-safe connectivity close

## Role and prior observation

This is a retrospective source-admission adjudication, not a prospective
scientific preregistration. A read-only inspection of the already frozen SF1.1
full-YFCC report found seven author UIDs with rows labelled Ektar 100, Velvia
50 and Portra 400. The formal leaf binds that exact report and applies the
previously established five-independent-author minimum only after excluding
explicit legacy or generation-ambiguous Portra rows.

The source report is the two-run byte-identical SF1.1 metadata artifact. No
SQLite rescan, Flickr request, image request or pixel read is permitted.

## Question

Does the existing CC BY 2.0 YFCC metadata pool contain at least five
independent authors who each have:

1. at least one exact Kodak Ektar 100 row;
2. at least one exact Fujifilm Velvia 50 row; and
3. at least one row that explicitly identifies the post-2010/current Kodak
   Portra 400 formulation?

This is only a connectivity gate. It cannot establish same-scene pairing,
stock response, stock identifiability, scanner/process control or learning
eligibility.

## Frozen classification

- Normalize title, description and user tags with URL decoding plus lowercase.
- A Portra row is `current_explicit` when it contains `new portra`,
  `new kodak portra`, or `portra 400 (new)`.
- A Portra row is `legacy_explicit` when it contains `400 NC`, `400 VC`,
  `400-NC`, `400-VC`, `400NC`, or `400VC`, unless the same row also explicitly
  states a current indicator. A comparison sentence mentioning both old VC/NC
  and the new formulation remains current-explicit.
- All remaining Portra 400 rows are `generation_ambiguous`.
- Only authors with at least one `current_explicit` row count toward admission.
  Legacy and ambiguous rows remain reported diagnostics and never backfill the
  current stock.

## Frozen gates and execution order

1. Hash and size of the exact SF1.1 report must match the contract.
2. The report must identify itself as metadata-only and record zero image
   payload download/decode.
3. Reconstruct per-stock UID sets from exact rows, independent of input order.
4. Report all pairwise and all-three shared-UID counts.
5. Apply the current-Portra classification above.
6. Require at least five all-three authors with a current-explicit Portra row.
7. If the author gate fails, stop with zero page, image, pixel, fit, render,
   score or database reads.

Forward and reverse stock-order runs must serialize to identical bytes.

## Stop rules

- Do not count Portra 400 NC/VC as the target stock.
- Do not infer generation from upload date alone.
- Do not use ambiguous plain `Portra 400` rows to reach the threshold.
- Do not request Flickr pages, downloads or image URLs after a metadata failure.
- Do not add authors, aliases, stocks, cues or thresholds after reading the
  result.
- Do not reopen the failed SF1.3B Ektar/Velvia identifiability route.

## Claim ceiling

Pass would open only a separately frozen live-rights and nuisance preflight for
the qualifying authors. Failure closes this exact YFCC three-stock/current-
Portra connectivity route. Neither outcome is stock calibration, a learnable
dataset, a product profile, a multi-stock system result or candidate-three
admission.
