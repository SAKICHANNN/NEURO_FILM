# SF2.1A Openverse exact-stock shared-creator contract

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.1A`

Status: frozen before implementation or formal metadata execution

## Question

Does the official Openverse search API expose a bounded, openly licensed
metadata graph in which the same creators contribute exact-text candidates for
multiple named colour stocks, with enough strict derivative-rights rows to
justify a later upstream live-rights and label audit?

This is source discovery, not stock evidence. Openverse indexes upstream
licence claims but explicitly does not guarantee their accuracy. A result title
or tag is weak `S0_claimed_text_only` evidence and may describe a product,
simulation, miscategorizaton or scan workflow rather than the physical stock.

## Parent evidence

- Commons and YFCC pixels are closed because content, scene colour and source
  fingerprints dominate stock labels;
- SF2.0A/B0/C0 close the NASA/JSC branches without weakening their gates;
- the full YFCC shared-author pilot still fails stock identifiability;
- bounded reconnaissance found 240-result exact-phrase caps for Ektar100,
  Velvia50, Portra400 and UltraMax400, with cross-stock creator names and some
  CC BY rows. Those observations design this gate but are not formal evidence.

## Frozen access and retention

Use only anonymous `GET /v1/images/` relevance searches for the four frozen
quoted phrases. Request at most 12 pages of 20 results per stock and at most 48
requests total, respecting response rate limits and a one-second interval.
Each response must be HTTP 200 JSON and at most 2 MiB; there is one attempt per
page so network retries cannot exceed the frozen request ceiling.

Never request an Openverse thumbnail/detail/related endpoint, upstream landing
page or image URL. Do not download pixels. Do not retain raw responses, image
URLs, thumbnail URLs, attribution prose or unbounded tags; retain only the
versioned metadata fields in the config and at most 64 normalized tags of 128
characters each per row.

## Frozen eligibility and connectivity gate

A row is strict only when:

- an exact normalized stock alias occurs in the title or tags and Openverse
  reports title/tag matching;
- creator, creator URL, upstream landing URL, source/provider, licence code,
  licence version and licence URL are present;
- licence is CC BY, CC0 or PDM under the frozen version list; NC, ND and SA are
  excluded from this strict lane;
- mature content is false and dimensions are positive;
- its title is not automatically deleted when a product/test pattern is seen;
  it is retained but flagged for the later human/vision label audit.

An eligible stock needs at least 20 strict rows, five normalized creators and
no creator above 40%. The graph must contain a same-source connected component
of at least three stocks, at least five shared creators overall and at least two
shared creators on every retained edge. Openverse IDs and upstream landing URLs
must not overlap across stock labels.

## Branches

- **Pass:** open only a separately frozen bounded upstream landing-page
  live-rights/label preflight for the best connected component. It does not
  open images or pixels.
- **Insufficient graph:** close this source as another metadata/data-gap result;
  do not loosen licence, creator or graph gates.
- **Identity overlap, contract mismatch or source failure:** fail closed and
  preserve bounded request evidence.

Every branch keeps pixels, fitting, training, LSM and stock claims false.

## DoD and claim ceiling

Commit this contract/config before client code or formal execution. Separate
network acquisition from a pure offline audit, preserve exact provenance and
response hashes, test pagination/rate-limit/no-image behavior and all
branches, run from committed code, repeat the offline audit byte-identically,
propagate the result and run the full CPU suite.

At most this node can establish Openverse-indexed weak stock text, strict
licence metadata and same-source shared-creator connectivity suitable for a
later live verification. It cannot establish upstream rights, physical stock,
pixel eligibility, content balance, stock identifiability, an operator, latent
mode, `S1/S2`, calibration, authenticity or product value.
