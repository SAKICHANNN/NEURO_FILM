# SF3.A3T — DADDY PLEASE Ektar 100 source admission

## Role

This is a prospective zero-pixel source-admission audit for Parker Day's
`DADDY PLEASE` Ordinals collection. The parent record declares Canon EOS-1V,
Kodak Ektar 100, one fixed lens/exposure, 100 models, 1,000 photographs and
CC0. The child index exposes 1,000 unique inscription identities whose compact
metadata can supply explicit model/content groups.

This leaf determines whether those declarations, child identities, metadata
groups and parent provenance are mechanically complete. It is not a colour,
grain, stock-response or product experiment.

## Frozen source and operations

1. Read the exact parent inscription HTML once.
2. Read the exact ten `/r/children/...` JSON pages and require 1,000 unique
   IDs plus the frozen ordered-ID SHA-256.
3. Read only the compact `/r/metadata/<id>` JSON response for every child.
   Decode only the five-string CBOR map `MODEL`, `MOOD`, `SIGN`, `PROP`,
   `BACKGROUND`; require 100 model groups with ten photographs each.
4. Rank all child IDs by `SHA256(id)` and read the HTML metadata pages for the
   frozen first 24. Require the exact parent link, AVIF content type, positive
   declared byte length, aggregate length and frozen sample identity.
5. Do not request `/content/`, `/preview/`, AVIF bodies, ranges or pixels.
6. Forward and reverse index/metadata/sample request orders must serialize to
   identical reports.

The complete metadata pass is allowed because it is approximately tens of
kilobytes of structured provenance, not media. Network concurrency is bounded
and does not change canonical ordering.

## Admission gates

Source-structure gates require exact Ektar/camera/project/CC0 declarations,
1,000 child IDs, complete child metadata, explicit 100-by-10 model grouping,
and exact sampled child-to-parent provenance.

Stock-evidence admission separately requires independent authors/sources,
explicit roll/process/scanner identities, at least one same-source cross-stock
control, a same-scene neutral/film pair and a sealed confirmation role. Model
identity is a content group, not a roll, process, scanner or independent
source.

## Stop rules

- Do not request or decode any image body in this leaf.
- Do not infer roll, process or scanner from the shared camera, lens, exposure,
  background, inscription adjacency or Bitcoin transaction structure.
- Do not infer that 100 models create 100 independent film workflows.
- Do not combine this Ektar-only project with Portra-only Luminant or
  Velvia-only YFCC as if stock and source were separable.
- Do not fit an operator, tune a threshold or open candidate three.

## Claim ceiling

At most this can establish a publicly addressable, CC0-declared, explicitly
model-grouped single-project Ektar 100 appearance source. It cannot establish
source-independent Ektar identity, an Ektar response, digital-to-film mapping,
three-stock evidence, calibration, learning eligibility or a product profile.
