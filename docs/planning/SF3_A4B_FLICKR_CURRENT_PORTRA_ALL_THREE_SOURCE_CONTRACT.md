# SF3.A4B — live Flickr current-Portra all-three source admission

## Role

This is a prospective, metadata-only source-admission audit for one Flickr
author absent from the seven-author SF3.A3Q all-three cohort. Public metadata
for `sprklg` (`59731892@N00`) contains Kodak Ektar 100, Fujifilm Velvia 50 and
rows that explicitly say `New Portra 400` or `Portra 400 (new)`. The leaf asks
whether this new live source closes the frozen A3Q current-generation author
connectivity gate and, separately, whether its own roles are sufficient to
authorize a pixel experiment.

It does not reopen the failed A3Q cohort, SF1.3B or R1HI natural-label
association experiments. A connectivity pass is not pixel, stock-response,
identifiability, fitting, learning, calibration or product evidence.

## Frozen source and operations

1. Read the exact public author HTML only to obtain Flickr's ephemeral public
   site key; do not retain or report the key.
2. Call official `flickr.people.getPublicPhotos` for the seven required metadata
   pages with extras limited to license, description, tags, capture date,
   owner name and media state. Do not request any media URL extra.
3. Require the exact owner, public row count, page count, unique photo IDs,
   target-row counts and canonical target-manifest SHA-256.
4. Count Portra only when the same row identifies both Portra 400 and one of
   the frozen A3Q current-generation phrases: `new portra`,
   `new kodak portra` or `portra 400 (new)`; compact `newportra` tags are
   retained only when the row also names Portra 400.
5. Record explicit `desaturated` / `de-saturated` disclosures as alterations.
   Do not infer an unedited scan when either phrase is present.
6. Bind the immutable SF3.A3Q evidence bytes and add this owner only after all
   three stock labels and CC BY 2.0 metadata gates pass.
7. Canonicalize by photo ID so forward and reverse API-page order must produce
   identical reports.

## Frozen gates

The connectivity gate requires the exact A3Q baseline of four qualifying
current-Portra all-three authors plus this distinct fifth author.

Pixel admission is independent and stricter. Every stock requires at least
three non-explicitly-altered rows; the source must also expose explicit
cross-stock camera control, roll, process and scanner identities, and a sealed
confirmation group. The single Velvia row discovered before freeze is
explicitly marked `desaturated`, but the formal report, not this observation,
decides the frozen gates.

## Stop rules

- Do not request `url_*`, original, thumbnail or EXIF extras, photo media,
  HEAD/range/body bytes, or pixels.
- Do not infer Portra generation from capture or upload date.
- Do not infer roll, process, scanner or same-scene pairing from order,
  location, author or camera name.
- Do not remove the explicit-alteration gate, add a second Velvia row, change
  the author, relax support minima or merge another cohort after the result.
- Do not use a connectivity pass to rescue a closed natural-label experiment,
  fit an operator or open candidate three.

## Claim ceiling

At most this leaf can establish one additional live CC BY 2.0 author with
explicit current Portra 400, Ektar 100 and Velvia 50 metadata. Pixel admission
requires every separately frozen support and grouping gate. No media body,
pixel, stock response, source-independent identity, operator, training,
calibration, candidate-three or product claim is permitted.
