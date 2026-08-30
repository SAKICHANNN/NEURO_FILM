# SF3.A3P FILM2PAINT reversal-target source contract

Date frozen: 2026-08-31

Node: `ULT > SF3 > SF3.A3P`

Status: prospective zero-pixel source gate

## Question

Does the official FILM2PAINT / Chromatic Divide lineage expose a reusable,
rights-clear, independently grouped physical-film target dataset that can open
a stock-first Velvia 50 observation leaf?

This is materially different from community labels, presets and manufacturer
curves. The official paper describes physical reversal films of known colour
targets, including Fujichrome Velvia 50, captured under documented illumination
and scanned with a hyperspectral camera against known reference colours.

## Frozen sources

- IS&T Archiving 2024 article DOI
  `10.2352/issn.2168-3204.2024.21.1.4`, exact official PDF URL, 2,002,643
  bytes, SHA-256
  `bad7ee283b3d8352faf45bdc567cb8166d9aaf4f5a5dc05d206300165643f0f7`.
- University of Basel edoc item UUID
  `0485fd09-fdcd-4dfe-97ee-95b133ea8b99`, DOI
  `10.5451/unibas-006820282`.
- Exact DSpace bundle and bitstream inventory frozen in
  `configs/sf3_a3p_film2paint_reversal_target_source_v1.json`.

The audit may read only the article document and official repository metadata.
It must not request the thesis PDF bitstream, article figures separately,
film-target scans, image derivatives, account/request-copy workflows, pixels,
fits, renders or scores.

## Admission gates

All gates are conjunctive:

1. The official article PDF matches its frozen byte identity and explicitly
   documents Velvia 50, known colour targets, controlled illumination and
   hyperspectral scanning.
2. The Basel item, bundles and every bitstream row match the frozen official
   inventory.
3. A public film-target dataset payload exists independently of the thesis and
   article documents.
4. The dataset has an explicit licence permitting research fitting and
   distribution of derived parameters.
5. The dataset publishes exact member identities/checksums and usable
   roll/process/scanner/source groups with an independent holdout.
6. All forbidden media/pixel/model/product operations remain zero.

Failure closes only this exact public-source route. It is not a claim that the
authors' private dataset is invalid, that Velvia 50 has no measurable signal,
or that a separately authorised export could not qualify.

## Claim ceiling

Even a source pass would open only a separately preregistered physical-target
data audit. It would not establish a stock operator, natural-scene transfer,
unseen-roll generalisation, product profile, candidate 3 or calibrated output.

