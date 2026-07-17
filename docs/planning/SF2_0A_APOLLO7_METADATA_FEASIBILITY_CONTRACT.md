# SF2.0A Apollo 7 stock/magazine metadata feasibility contract

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0A`

Status: frozen before bounded live-page access

## Question

Does the authoritative Apollo 7 record expose enough same-mission,
multi-magazine metadata connectivity between Kodak Ektachrome SO-368 and
SO-121 to justify a larger metadata-only stock/content/nuisance audit?

This is a source-feasibility question, not a stock-identifiability experiment.
Apollo uses unusually narrow orbital content, non-consumer emulsions, different
filters and difficult exposure conditions. A pass cannot establish a reusable
stock look or authorise pixels.

## Parent evidence and DoR

- SF1.3B closed the current community pixels because held-out-UID RGB signal
  loses to nuisance controls;
- LSM1 therefore remains ineligible and no model-capacity fallback is legal;
- NASA/JSC documents Apollo 7 magazine, film, filter and frame ranges;
- NASA's photo pages expose mission-roll-frame, film code, exposure assessment,
  geographic/features/caption metadata and offered-file metadata;
- raw Apollo flight-film scans are public domain with required credit, while
  ASU processed derivatives have separate restrictions and are out of scope;
- no API key, external contact, image request or paid resource is needed.

## Frozen scope

Use the seven colour magazines in the authoritative Apollo 7 report:

| Physical magazine | Photo roll | Frames | Stock | Filter |
|---|---:|---:|---|---|
| M | 3 | 1511-1557 | SO-368 | none |
| N | 4 | 1558-1612 | SO-368 | none |
| Q | 5 | 1613-1671 | SO-121 | Wratten 2A |
| O | 6 | 1672-1737 | SO-121 | Wratten 2A |
| S | 7 | 1738-1879 | SO-121 | Wratten 2A |
| R | 8 | 1880-1943 | SO-121 | Wratten 2A |
| P | 11 | 1979-2043 | SO-121 | none |

Select nine deterministic, evenly spaced frames per magazine using the exact
integer algorithm in the config: 63 sequential HTML requests maximum. Retain
only bounded page metadata, request UTC and page SHA-256. Do not retain HTML
bodies and do not request any offered JPG, PNG, TIFF, ZIP, KML or cloud mask.

## Nuisance-aware feasibility gate

The audit must verify the expected NASA photo ID and stock code on every usable
page. A `metadata_connectivity_candidate` additionally requires:

- at least 12 valid pages per stock and four per magazine;
- at least two independent magazines per stock;
- at least two preregistered shared content tags, each supported by at least
  three rows and two magazines per stock;
- at least four filter-free rows per stock, preserving an explicit cross-stock
  bridge rather than hiding the SO-121/Wratten-2A association;
- at least two distinct reported exposure states in the complete audit.

Content tags are deterministic keyword observations over geographic name,
features and caption. They are nuisance/connectivity descriptors only and may
not be treated as stock labels or learned embeddings.

## Branches

- **Candidate:** freeze the exact report and design a separate, more complete
  metadata-only audit with mission/location/exposure/filter controls. Pixels,
  fitting, training and LSM remain false.
- **Insufficient connectivity/content-confounded:** retain Apollo as a narrow
  archive/stress candidate; do not add more pages or model capacity to rescue
  the result.
- **Stock/filter-confounded:** do not interpret SO-121 versus SO-368 appearance
  as stock signal; seek a different source or a genuinely balanced bridge.
- **Source unavailable/metadata mismatch:** stop the source leaf and preserve
  the discrepancy. Do not substitute unregistered frames after seeing results.

## DoD and evidence bundle

- frozen config hash and software commit;
- source URLs, access timestamps, page hashes and exact deterministic sample;
- page/stock/magazine/filter/exposure/content support matrices;
- source availability and metadata mismatch accounting;
- deterministic decision from the frozen gates;
- explicit `image_payload_download_allowed=false`,
  `operator_fitting_allowed=false`, `training_allowed=false` and
  `latent_mode_study_allowed=false`;
- targeted tests, complete CPU suite, report/decision hashes, propagation,
  scoped commit and push.

## Claim ceiling

At most: authoritative Apollo 7 stock/magazine metadata feasibility. No result
from SF2.0A is a real digital-to-film operator, a stock response, a latent
mode, a calibrated profile, an `S1/S2` promotion or a product-ready look.
