# SF3.A3V Noritsu Portra 400 physical-profile source contract

Date frozen: 2026-08-31

Node: `ULT > SF3 > SF3.A3V`

Status: prospective zero-pixel source gate

## Question

Does the exact public `noritsu-tool` Portra 400 calibration profile expose a
reproducible, rights-cleared physical observation with enough independent
groups to open a stock-first operator experiment?

This is materially stronger than a preset-name audit. The official repository
states that the profile was fitted from real LS-600 raw scans and matching
machine-rendered TIFFs. The paired-data audit, however, also states that those
owner captures and native evidence are private and are not redistributed.

## Frozen sources and reads

The exact repository commits, trees, Git blobs, sizes, decoded-text SHA-256
identities and required statements are frozen in
`configs/sf3_a3v_noritsu_portra_physical_profile_source_v1.json`.

The audit may read only commit-pinned GitHub raw responses for the three frozen
text blobs (`README.md`, `LICENSE`, and the paired-data audit source). It may
make only a body-free `HEAD` request for the profile. The exact commits, trees
and Git blobs were frozen before execution; commit-pinned raw URLs plus decoded
text hashes bind the executed sources without depending on mutable branch heads
or GitHub API quotas. It may inspect only the Git blob identity and byte size of
`portra400_calib.npz`. It must not request or decode that profile body, owner
RAW/TIFF/PRM/native files, sample imagery, pixels, fits, renders or scores.

## Admission gates

All gates are conjunctive:

1. Exact official repository, commit, tree, commit-pinned text and
   profile-metadata identities remain frozen.
2. The real Portra 400 LS-600 pair and three-frame calibration statements are
   present.
3. The underlying owner RAW/TIFF observations are publicly addressable.
4. Their owner grants fitting and derived-parameter use compatible with the
   project; the code repository's MIT licence alone is insufficient.
5. Exact roll, process, scanner-unit/source and sealed independent
   confirmation roles are published.
6. The pair payload has a public member manifest and checksums.
7. All forbidden profile-body, media, pixel, fit, render and score operations
   remain zero.

Failure closes only this exact public derived-profile route. It does not
dispute the private measurements or the tool's scanner-compatibility result.
A newly published, independently licensed pair corpus with group roles would
require a fresh prospective leaf.

## Claim ceiling

Even a pass would open only a separately frozen pair-integrity audit. It would
not establish a digital-to-film operator, source-independent Portra response,
calibration, product profile, three-stock completion or candidate 3.
