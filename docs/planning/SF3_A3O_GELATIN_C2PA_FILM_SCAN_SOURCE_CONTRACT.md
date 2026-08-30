# SF3.A3O — Gelatin Labs C2PA film-scan source admission

## Question

Does Gelatin Labs' official 2026 Content Credentials launch expose a
rights-usable, machine-verifiable physical-film source for the stock-first
Velvia 50 / Portra 400 / Ektar 100 programme?

## Frozen sources

- official Content Credentials page: `https://cr.gelatinlabs.com/`
- official Terms of Service: `https://gelatinlabs.com/terms-of-service/`

The launch is materially different from weak community labels. It says every
delivered scan is signed with C2PA 2.2 and records physical-film origin, roll
number, stock, format, scanner, process and push/pull state. The same page also
places four CAWG training-and-mining assertions at `not allowed`, while the
Terms state that customers retain all rights and Gelatin Labs acquires no usage
rights over their photographs.

## Execution order and gates

1. Request only the two official HTML documents, once per role.
2. Verify the exact provenance fields, C2PA version and physical-film origin
   claims from the official launch page.
3. Verify the no-training/no-inference/no-mining assertions and customer-rights
   boundary from the official pages.
4. Require publicly listed exact assets for Velvia 50, Portra 400 and Ektar
   100, an asset/roll/process/scanner manifest with cryptographic hashes,
   independent scene and roll groups, and explicit permission for research
   fitting and distribution of derived parameters.
5. Any failed gate closes before image, verifier-upload, fingerprint lookup,
   account, order, pixel, fit, render or score.

## Stop rules

- Do not download or request sample images, TIFFs, JPEGs, PNGs or customer
  scans.
- Do not upload anything to the verifier or query the private fingerprint
  registry.
- Do not create an account, place an order, contact the lab or seek customer
  files.
- Do not interpret a signed stock label as permission to train, fit, mine,
  redistribute or ship derived stock parameters.
- Do not substitute Kodak Ultramax 400 or a folder-name illustration for any
  target stock or for a downloadable dataset.

## Claim ceiling

At most this leaf can establish that signed lab-origin C2PA metadata is a
promising provenance mechanism for future independently licensed film scans.
It cannot establish a reusable dataset, target-three-stock coverage, stock
response, fitting rights, independent holdouts, product profile or release
permission.
