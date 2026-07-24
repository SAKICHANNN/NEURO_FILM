# U5.R2H0C3 external measured-spectrum source audit

Date: 2026-07-24  
Decision: **USGS Spectral Library Version 7 measured ASCII pass**  
Parent: `U5.R2H0C2`

## Purpose

Find one rights-eligible reflectance source that is independent of Columbia
CAVE and can test the already frozen H0C2 policy. This audit does not inspect
the synthetic witness target errors and does not change the retained hard
Top-1 radius of Delta E76 `1.0`.

## Selected source

The selected source is the official USGS Spectral Library Version 7 data
release, DOI `10.5066/F7RR1WDJ`, ScienceBase parent item
`5807a2a2e4b0841e59e3a18d`. USGS describes the library as laboratory, field
and airborne spectra spanning natural and man-made materials. The current USGS
catalogue marks access as public and links the US public-domain label; the USGS
2025 download instructions also state `Public Domain`.

The 5,479,324,354-byte aggregate archive is **not required**. ScienceBase child
item `586e8c88e4b0f5ce109fccae` publishes the original measured `splib07a`
spectra separately:

- file: `ASCIIdata_splib07a.zip`;
- bytes: `21,812,828`;
- official MD5: `bfe74068d85811e52e5e07d017720a17`;
- acquired SHA-256:
  `d232645740869a82aafcad5839448c50b1dc72965ce042d1374f29b7a798a91c`;
- ZIP members: `3,156`;
- uncompressed bytes: `110,402,184`.

The local MD5 is byte-identical to the ScienceBase record. The archive remains
under ignored `data/real_film/usgs_splib07/`; project code and reports may
record hashes and aggregates but do not redistribute source spectra.

## Frozen eligibility boundary

Only original measured reflectance records whose header ends in `AREF` are
eligible. The evaluator may use ASD and Beckman records with valid measured
coverage across 400--700 nm. It must:

- use the official wavelength record matching the exact sample length;
- treat USGS sentinel values such as `-1.23e34` as missing;
- linearly interpolate valid measured samples to the already frozen 31-point
  400--700 nm / 10 nm overlap grid;
- reject rather than clip any interpolated reflectance outside `[0, 1]`;
- retain only the H0C1 relative-Y range `[0.005, 0.95]`;
- exclude `errorbars`, `RREF`, `TRAN`, `RTGC`, NIC4 and any spectrum without
  full visible overlap;
- preserve archive member, record header, chapter, instrument family and a
  filename-derived sample-record group in every query row.

A source-only feasibility scan, performed before any target rendering, found
`1,732` eligible records across seven chapters. With the frozen CAVE bank and
D65 Lab feature, `184` records (`10.62%`) are within Delta E76 `1.0`, spanning
all seven chapters; the largest chapter supplies `48.91%` of selected records.
These figures may set support gates but cannot set performance gates.

## Confirmatory design boundary

The external USGS spectra are queries only. The retrieval bank remains the
official-CRC-verified CAVE population used by H0C2. USGS spectra may not enter
the bank, train a model, tune the threshold, select a feature or alter the
smooth fallback. This makes the test genuinely cross-source.

The evaluator will compare, at the fixed `1.0` radius:

1. the existing bounded smooth spectrum reconstruction;
2. the frozen hard Top-1 CAVE spectrum, otherwise the same smooth fallback.

Both are scored against the known USGS spectrum under the same synthetic
datasheet-prior witness. This remains mechanism evidence, not real-film,
Velvia, photographic-image, calibrated-stock or product evidence.

## Rejected/unused alternatives

- the 5.48GB aggregate ScienceBase bundle: unnecessary and its current manager
  endpoint serves a request page rather than a range-readable archive;
- CAVE again: not independent;
- Bristol and Harvard hyperspectral sets: useful reserves, but weaker current
  access/size terms than the official public-domain 20.8MB USGS measured ASCII
  package;
- convolved `splib07b` and sensor-resampled products: derived rather than the
  requested original measurements.

## Branch decision

The source, rights, integrity, overlap and pre-target support audit passes.
Open the preregistered external replication evaluator only. Failure closes the
broad external empirical-prior claim; no threshold retuning, soft blending,
neural rescue or same-source retrieval is allowed.

## Claim ceiling

Source eligibility and pre-target support for a cross-source test of one fixed
hard measured-spectrum canonicalizer under a synthetic datasheet-prior witness.

