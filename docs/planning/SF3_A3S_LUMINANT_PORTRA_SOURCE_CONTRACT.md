# SF3.A3S — Luminant Halide Portra 400 source admission

## Role

This is a prospective, zero-pixel source-admission audit for the public
Luminant Halide gallery. It is not a film-response experiment. The source was
found after the owner-supplied physical-capture recovery condition was retired.

The site states that its images were captured by a human operator through a
photosensitive chemical process, transferred to the digital domain through
optical reimaging, and published under CC BY 4.0. A read-only discovery pass
found 191 exact Portra 400 entries across the first 74 fixed gallery pages.

## Frozen source and operations

1. Read only the public `about` page and gallery pages `0001` through `0074`.
2. Parse only image IDs, month, film label, place and deterministic full-image
   URL from HTML. Do not request thumbnails or image bodies.
3. Retain only exact `Kodak Portra 400`, `Portra 400`, and
   `Kodak Portra 400 +2` labels.
4. Require the frozen inventory count, label counts and canonical inventory
   SHA-256.
5. Rank IDs by `SHA256(id)` and issue HEAD requests for the frozen first 24.
6. Require exact status, JPEG media type, byte ranges, aggregate bytes and
   canonical HEAD-metadata SHA-256.
7. Forward and reverse page/HEAD orders must serialize to identical reports.

Allowed network operations are 75 HTML GET requests and 24 image HEAD
requests per run. Image GET/range, pixel decode, fit, render and score are zero.

## Admission gates

The audit gates verify a real labelled public source. Stock-evidence admission
additionally requires independent author/source groups, explicit roll, process
and scanner groups, same-scene neutral/film observation and a sealed independent
confirmation role. A place or month is not a roll, process or scanner identity.

## Stop rules

- Do not download any gallery image in this leaf.
- Do not infer roll, process, scanner, camera or neutral input from adjacency,
  month, place or visual content.
- Do not combine `+2` and standard exposure as an identified stock-response
  curve.
- Do not use site-level CC BY 4.0 as evidence for Kodak manufacturer data.
- Do not backfill missing groups with YFCC, Commons, FilmMatch or NegICC.
- Do not fit an operator, tune a threshold or open candidate three.

## Claim ceiling

At most this can establish a rights-clear, publicly addressable, single-site
Portra 400 appearance source with a bounded original-file preflight. It cannot
establish a calibrated Portra response, source-independent stock identity,
digital-to-film mapping, three-stock evidence, product profile or learning
eligibility.
