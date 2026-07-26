# SF2.7R ColorReference scanner-nuisance source results

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.7R`

Decision: **source pass; open separate scanner-nuisance quantification**

## Result

The exact bounded lane is locally complete:

| Evidence | Result |
|---|---:|
| retained assets | 11 / 11 |
| retained bytes | 71,068,957 |
| source TIFFs | 5 |
| measured reference archives | 2 |
| complete scan pipelines | 4 |
| slide identities per pipeline | 5 / 5 |
| ZIP CRC failures | 0 |
| image decode failures | 0 |
| cross-pipeline exact duplicate pairs | 0 |

The four pipelines cover one Nikon LS 50 ED with NikonScan, the same scanner
model with VueScan, a second LS 50 ED/NikonScan capture, and a Nikon LS
9000/NikonScan capture. Every archive contains the same Set 3 slide identities
1–5. Image sizes and colours differ across pipelines, so the files are not
repeated representations.

## Reproducibility

- frozen config SHA-256:
  `ed44375415b7471aa1b0256d87e086a66c83f55c0cab3d2a25ec33a90e037fa0`;
- acquisition/audit software commit:
  `cec5cc7355bb764d9e64182dabda57e40ceeeb97`;
- two independent local audit reports:
  `b88da41c129b58087c68194f21545c1f7b708be91ba7d1b868ce50579d410fb6`;
- all server byte counts match the frozen values;
- all retained bytes and archive members have SHA-256 lineage in the ignored
  audit report.

## Evidence meaning

This is unusually useful nuisance-control structure: identical physical
Velvia 100F target slides are observed through scanner-hardware,
scanner-device and software changes, while measured IT8/CGATS references and
the recorder-device-space source grids are also present.

It is not a digital scene/film pair. The source TIFFs are explicitly in the
film recorder's device space, not sRGB or neutral scene-linear RGB. Set 3 is
one physical target set, not independent rolls. Therefore this corpus cannot
identify a Velvia operator, prove stock distinction, train a stock expert or
support a calibrated output claim.

## Rights boundary

The source page permits free scanner-profiler testing/development use but does
not provide a standard redistribution licence. Local use remains internal
research. Raw archives, extracted images and derived weights are not
redistributed, and no commercial-training right is inferred.

## Branch

Open `SF2.7A` only: preregister a same-slide patch/alignment audit comparing raw
pipeline separation with bounded global canonicalizers and leave-one-slide-out
validation. Its purpose is to quantify how easily scanner/software nuisance can
create stable colour clusters and which nuisance controls future stock/mode
studies must beat.

No stock fitting, training, LSM, production integration or named-stock
authenticity claim opens. The Ultimate Goal remains active.
