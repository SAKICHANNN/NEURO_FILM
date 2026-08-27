# RF3.D13 — Darkroom three-stock explicit-baseline source contract

## Question

Does the exact official `ComfyUI-Darkroom` revision provide a rights-clear,
source-traceable and headlessly reproducible explicit operator for each of
Velvia 50, Portra 400 and Ektar 100, suitable only as a common-input baseline?

## Frozen source

- repository: `https://github.com/jeremieLouvaert/ComfyUI-Darkroom`
- branch observed before audit: `main`
- commit: `eb8f3d86f49a57e54ff426b3f517f8747248ec99`
- README claims: 161 stocks, MIT, Capture One curve data and published
  manufacturer technical data

The audit reads only Git objects and source metadata. It requests no image,
film scan, preset marketplace payload, model or user content.

## Admission gates

All gates must pass without correction after source inspection:

1. the commit and audited blobs are exact and replayable;
2. a root licence file explicitly covers the repository code and bundled
   operator data;
3. each of the three exact stocks has one machine-readable explicit operator;
4. every admitted stock row identifies its primary source, extraction method
   and rights basis rather than relying only on a repository-wide claim;
5. the operator can run headlessly and deterministically without ComfyUI,
   network access, proprietary applications or unavailable external assets;
6. no bundled Capture One-derived or other third-party data has an unresolved
   redistribution/derivation boundary;
7. the source is materially distinct from the already closed
   `spectral_film_lut` structural-bank route.

## Stop rules

- Missing licence coverage, per-row provenance, an exact stock, or a headless
  path closes RF3.D13 before pixel execution.
- Do not infer that a repository-level MIT statement licenses proprietary
  source data.
- Do not replace a failed stock, scrape Capture One, reconstruct missing data,
  normalize a rights defect, or fall back to the already consumed spectral
  bank.
- A pass opens only a separately frozen common-input baseline. It does not
  establish real-film evidence, stock accuracy, calibration, preference or
  product promotion.

## Claim ceiling

Source and callable admission for a three-stock explicit Look Approximation
baseline only. No pixels are evaluated in RF3.D13.
