# SF2.0C0 NASA/JSC cross-mission exact-stock connectivity contract

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.0C0`

Status: frozen before implementation or formal query execution

## Question

Does the keyless NASA/JSC astronaut-photography metadata contain any single
mission where both exact primary media codes `VELVI` (Velvia 50) and `5775`
(Portra 400NC) have enough independently supported film rolls to justify a
separate content/date/geography/focal-length nuisance audit?

This is a cross-mission source census. It does not reopen the failed STS098
snapshot, change its roll gate or request photographs.

## Parent evidence

- SF2.0B0 retained only STS098 frame rows and closed because Velvia had two
  rolls below the frozen four-roll primary minimum;
- the result tables contain all missions for an exact code, but the B0 report
  retained only aggregate all-mission counts plus STS098 records, so no other
  mission can be inferred from frozen evidence;
- a different mission is a new source-design candidate, not a post-result
  substitute inside the closed STS098 contract;
- stock learning remains forbidden, so only mission/roll aggregation is legal.

## Frozen access and retention

Issue one keyless `Technical.pl` POST and generated result-table GET for each
code `VELVI`, `5775` and auxiliary `5776`: at most six requests. Parse the same
nine columns transiently and retain only response evidence, per-code total rows,
mission×stock row/roll counts, photo-ID overlap, ranking and decision.

Do not retain frame rows, raw HTML, captions, geography, dates, feature text or
focal lengths. Never request `photo.pl`, an image, API, mask, ZIP or KML.

## Frozen support gate

A mission passes only when both primary stocks independently have at least four
rolls with at least eight rows each and at least 32 total rows. Photo-ID overlap
must be zero. Auxiliary Portra 400VC cannot rescue either primary.

Rank passing missions by minimum primary supported-roll count, minimum primary
row count, then mission ID. Ranking selects a later metadata-audit target; it
is not evidence of content balance or stock signal.

## Branches

- **Candidate found:** open only a separately frozen SF2.0C1 metadata-only
  content/date/geography/focal/record-type nuisance audit for the top mission.
- **No candidate:** close this NASA/JSC cross-mission exact-code expansion; do
  not lower support, add codes after results or request pixels.
- **Overlap/contract/source failure:** close and preserve request evidence.

Every branch keeps photo pages, pixels, fitting, training and LSM false.

## DoD and claim ceiling

Commit config/contract before code/query; reuse the source client without B0
regression; test aggregation, no-frame retention and branches; run formally
from committed code; replay offline; propagate and run the full CPU suite.

At most this can establish mission-level roll connectivity suitable for a later
nuisance audit. It cannot establish pixel rights, content balance, stock
identifiability, an operator, latent mode, `S1/S2`, calibration or authenticity.
