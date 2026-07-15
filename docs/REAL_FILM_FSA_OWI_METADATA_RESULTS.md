# RF0.3 FSA/OWI metadata-gate results

**Decision:** the raw Commons category is rejected as a corpus;
`canonical_pilot_allowed` for a strictly deduplicated subset. No image pixels
were downloaded or decoded in this phase.

## Reproducible result

- 1,031 unique Commons pages were enumerated in 21 API responses.
- 983 pages (95.34%) exposed a LOC `fsac.*` identifier.
- The raw category contained only 559 unique identifiers and therefore failed
  the frozen one-record-per-identifier rule.
- Public-domain category coverage was 99.42%, below the frozen 100% raw gate.
- LOC-source category coverage was 95.64%.
- Strict filtering removed 48 records without LOC identifiers, five without
  the LOC-source category, and one without the required public-domain category.
- Selecting one deterministic representation per remaining LOC identifier
  produced 558 canonical records and removed 419 duplicate representations.
- Curated photographer-category coverage is 81.90%. The canonical subset has
  nine named photographer groups; seven have at least eight records.

The complete acquisition was executed twice at commit `0dbb52a`; the raw
manifest, canonical manifest and report were byte-identical on the second run.

| Artifact | SHA-256 |
|---|---|
| raw manifest | `dcdea06dcef1d51720d3ef02eb1b3e917a1be5a706cb2a30a141860d73943352` |
| canonical manifest | `914a659e5c88795f352aaf36a1bc08dfe5774bd7e1f8c8c75b7c318d4c39a12e` |
| metadata report | `67db03a94de75a47cf2f7504ae82739b812e3fcc5340dd7945a6aec758a75416` |

## Interpretation

The first apparent duplicate explosion was not a parsing artefact: Commons
really contains multiple crops, adjusted files, format conversions and renamed
representations of some LOC scans. Treating all 1,031 pages as independent
training samples would create severe leakage and overweight popular images.
The raw negative result is therefore preserved rather than averaged away.

The canonical subset passes only a data-access gate. Creator labels remain
missing for 18.10% of the subset, creator is entangled with shooting assignment
and content, and the collection does not expose physical-roll IDs. The source
still cannot identify an emulsion response or train a stock-accurate transform.

## Next gate

Phase B may download at most 64 deterministic creator-balanced 1,280 px
derivatives and at most 64 MiB total. It must verify decoding, hashes,
dimensions, duplicate content, borders/crops/retouching, colour-profile
variation, caption alignment and coarse content balance. Failure stops the
source before any learning experiment.
