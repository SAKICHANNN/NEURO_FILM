# SF2.7R ColorReference scanner-nuisance source contract

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.7R`

Status: **frozen before retained download or archive inspection**

## Question

Can a small public test corpus provide a same-physical-film-slide,
different-scanner/software control that quantifies one of the dominant nuisance
variables before future stock or latent-mode work?

This does not seek a digital-to-film operator.

## Source and rights boundary

The official ColorReference test-data page states that user scans are made
available free to anybody interested for scanner-profiler development,
comparison and related purposes. It does not provide a standard reusable data
licence or explicit redistribution grant.

Therefore:

- acquisition is bounded to `71,068,957` exact bytes;
- use is internal scanner-nuisance research only;
- archives, extracted pixels and derived weights are not redistributed;
- no commercial-training or public-release right is inferred;
- no external message or purchase is required.

## Frozen acquisition

Acquire exactly:

- five scaled recorder-device-space source TIFFs;
- Set 3 IT8 and CGATS.5 measured reference archives;
- Nikon LS 50 ED / NikonScan;
- Nikon LS 50 ED / VueScan;
- a second Nikon LS 50 ED / NikonScan capture;
- Nikon LS 9000 / NikonScan.

All assets, byte counts and server ETags are fixed in
`configs/sf2_7r_colorreference_scanner_nuisance_v1.json`. The hard transfer cap
is 72 MiB. No other set, scanner or full-resolution contact-only file may be
added after results.

## Feasibility gates

- every request returns the exact expected byte count;
- every ZIP passes CRC testing;
- every image decodes and reports dimensions/mode/bit depth;
- all four scan pipelines contain all five Set 3 slide identities;
- archive naming/readme evidence supports the same physical target set;
- no exact duplicate exists across scan pipelines;
- manifest records URL, ETag, SHA-256, bytes, role, archive member and lineage.

Passing establishes only a scanner-nuisance control corpus and may open a
separately frozen patch-alignment/separability audit.

## Prohibited interpretations

- recorder-space TIFF is not neutral scene-linear RGB;
- a scan is not a stock-independent film truth;
- same target set is not independent-roll support;
- differences between pipelines are scanner/software nuisance, not latent film
  modes;
- the one Velvia 100F target set cannot identify a Velvia stock operator;
- no fit, training, LSM, calibration or production integration opens.

## DoD

- contract/config committed before retained transfer;
- bounded resumable download with exact bytes/hash;
- archive CRC, decode, inventory and duplicate audit;
- two deterministic manifests/reports;
- evidence registry/tracker/log propagation;
- scoped commit/push; Ultimate Goal remains active.
