# SF0.5 Commons named-stock pixel pilot contract

## Question and stop rule

Can a small, rights-complete sample from the three SF0.4 exact-stock Commons
categories survive pixel integrity, duplicate, content and true source-group
checks strongly enough to justify a later unpaired explicit-operator test?

This node does not train a model. Any stock with fewer than eight retained
images, fewer than five normalized author groups, more than 60% from its
largest normalized author, a cross-stock exact/dHash<=4 collision, or confirmed
severe source corruption is stopped before learning. Uploader names are not a
substitute for authors.

## Immutable input and rights filter

- use only snapshot SHA-256 `f7ee9db16b198f3ce59c2cea887792e82f6dce8109ae6a79cb17c4ef7f10e5cd`;
- exact Ektar100, Superia X-TRA400 and Gold200 categories only; no Velvia-family rows;
- CC0/CC BY rows require an explicit licence URL;
- public-domain rows require explicit usage terms and a Commons file page;
- exclude bare `Attribution`, all share-alike rows, missing author/source URLs,
  and missing 1600px derivative URLs;
- preserve raw author, uploader, credit, licence, file-page, original URL,
  Commons SHA1 and derivative URL in every manifest row.

## Deterministic selection and acquisition

Selection is stable uploader round-robin, title-sorted within each uploader,
while enforcing both a 12-row normalized-author cap and a 12-row uploader cap
per stock. The per-stock cap is 64; global ceilings are 192 files and 512 MiB.
Only the 1600px derivative may be fetched, with a 32 MiB per-file fail-closed
limit and atomic `.part` replacement. Existing files are resumed only after
their recorded SHA-256 and decode checks pass.

The frozen preflight expectation is 27 Ektar, 14 Superia and 11 Gold rows.
Superia's 105 nominally permissive rows reduce to 14 because 103 share one
author. This is evidence of source confounding, not a reason to reinterpret
nine uploader accounts as nine independent photographers.

## Evidence bundle and promotion boundary

The implementation must produce a byte-stable selection manifest, resumable
download manifest, per-file SHA-256/decode/dimensions, exact and dHash<=4
duplicate audit, author/uploader distributions, content/contact sheets and a
machine-readable decision. A repeated offline audit must be byte-identical.

Passing this node raises only bounded community-category pixel evidence. It
does not establish stock response, physical-roll generalization, scanner or
process control, `S1/S2`, authenticity, training rights, released weights or
commercial suitability.

Machine contract: `configs/real_film_commons_stock_pixel_pilot_v1.json`.
