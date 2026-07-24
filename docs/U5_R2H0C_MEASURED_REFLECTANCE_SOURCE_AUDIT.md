# U5.R2H0C measured-reflectance source audit

Date: 2026-07-24
Decision: **CAVE research-only acquisition eligible; open U5.R2H0C1**
Parent evidence: `U5.R2H0A` numerical identifiability failure

## Research question

Do measured reflectances of real materials that are close under D65 CIE colour
produce materially less variation through the frozen Velvia datasheet witness
than the adversarial feasible metamers in H0A?

This is a new empirical-prior question. It cannot reverse the theoretical H0A
non-identifiability result, identify a digital-to-film operator, or turn the
datasheet witness into a calibrated Velvia response.

## Source matrix

| Source | Primary evidence | Visible support | Access / rights boundary | Decision |
|---|---|---|---|---|
| Columbia CAVE Multispectral Image Database | official CAVE database page and Yasuma et al. technical report/paper | 32 scenes, 512x512, 31 reflectance bands at 400--700 nm / 10 nm; real materials, skin/hair, paints, food/drinks, real/fake objects | official page says the database is available to the research community; no reusable licence grant was found. Internal research and derived aggregate metrics only; do not redistribute source pixels or use for commercial/released weights | **first pilot** |
| USGS Spectral Library v7 | USGS Data Series 1035 and ScienceBase release `10.5066/F7RR1WDJ` | thousands of laboratory/field/airborne spectra, broad material coverage and ASCII metadata | official public data release; current complete archive is 5,479,324,354 bytes and the download endpoint does not expose a convenient bounded member index | reserve replication/source-diversity lane |
| Bristol hyperspectral natural images | official University of Bristol/UAB-hosted database description | 29 natural scenes, 31 bands, 400--700 nm | research-use/acknowledgement language is reported by the source, but the current page is intermittently unavailable | reserve after live rights snapshot |
| Harvard hyperspectral database | official Harvard vision page | 50 daylight and 27 indoor hyperspectral images | non-commercial research use; 7.5 GB total | reserve; larger and more restrictive than the first pilot |

## Why CAVE is the first source

The Columbia-hosted full archive is currently reachable at
`https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip`.
The server reports 405,988,465 bytes, `application/zip`, byte ranges and ETag
`"1832e471-4630073df3380"`. The newer repository route currently returns HTTP
502, but the official legacy host is live; this is an endpoint discrepancy,
not a mirror substitution.

CAVE is small enough for exact local hashing, has dense pixel-level conditional
support, and covers several material categories in a single acquisition system.
The official source warns that its reflectances are calibrated estimates and
close approximations rather than exact physical measurements. Scene and spatial
sibling structure therefore remain mandatory grouping/nuisance variables.

## Frozen source boundary

U5.R2H0C1 may:

- download the one official 405,988,465-byte ZIP;
- record URL, response headers, SHA-256, member inventory and decode integrity;
- retain the source archive under ignored data storage for internal research;
- compute non-identifying aggregate statistics and frozen manifests;
- use only the 400--700 nm overlap of the H0A witness.

It may not:

- fit on current film pixels, anchors, owner preferences or H0A outputs;
- train an RGB-to-spectrum network before empirical feasibility is established;
- treat neighbouring pixels as independent observations;
- publish or redistribute CAVE pixels, weights learned from them, or a source
  derivative without a separate licence decision;
- extrapolate 400--700 nm values to 380--720 nm and call the result measured;
- claim that a narrow empirical conditional distribution resolves metamerism.

## U5.R2H0C1 acquisition and feasibility DoR/DoD

DoR is satisfied by an official reachable archive, bounded size, research-only
use, exact lineage and a question answerable without film pixels or training.

DoD requires:

1. exact size and SHA-256 plus a deterministic member inventory;
2. all required 16-bit PNG bands decode with consistent dimensions and naming;
3. no archive path traversal or unexpected executable content;
4. scene/material grouping retained in the manifest;
5. a preregistered sampling and conditional-pair contract written before any
   witness-output result is examined.

Failure closes CAVE only. It does not reopen H0A or justify a larger model; the
next legal source is a bounded USGS ASCII subset or another separately audited
measured-reflectance collection.

## Claim ceiling

At most: empirical conditional variability of one approximate measured-
reflectance database under one frozen datasheet-prior witness. Any useful
result remains `film-inspired/datasheet-prior`, source-limited and
theoretically non-identifying.
