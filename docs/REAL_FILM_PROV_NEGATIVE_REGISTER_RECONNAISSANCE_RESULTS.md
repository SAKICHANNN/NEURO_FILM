# SF2.5R PROV negative-register reconnaissance results

Date: 2026-07-23

Decision: **close current source expansion — the film-stock register is
catalogued as physical-only**

## Why the source mattered

Public Record Office Victoria describes VPRS 17690 as the register to VPRS
17684 and says it was used to track film stock. VPRS 17684 is a large
institutional collection of Country Roads Board master negatives and digitised
images. An accessible register could therefore provide a rare same-agency,
negative-number-linked stock/data design instead of uncontrolled community
labels.

## Reproducible metadata result

Two bounded official-API passes were canonicalized by removing only Solr
`QTime`. They are byte-identical at SHA-256
`dc2f7a35ac52b8e5b922ce2a214587449d9f1bb0966bff20465b497f8d03c954`.

| Series | Result |
|---|---:|
| VPRS 17684 collection items | 6,832 |
| VPRS 17684 digital items | 6,716 |
| VPRS 17684 physical items | 116 |
| VPRS 17690 register items | 30 |
| VPRS 17690 physical items | 30 |
| VPRS 17690 digital/IIIF/VEO records | 0 |

The 30 catalogue records span six consignments: P0001 15, P0002 1, P0003 1,
P0004 5, P0005 4 and P0006 4. Their titles/ranges describe physical annual
volumes and negative-number ranges. The catalogue exposes open access status,
but not the register pages or per-negative film-stock entries.

The audit report SHA-256 is
`0f7a514397dc836b60a21db2e7633f872107c4fde984da3981b7c6f8ce5b792a`
at implementation commit
`162eedc791de72a9636ca67bcf7ef7e90c3b8850`.
No image, TIFF, VEO or IIIF payload was requested.

## Branch decision

Close SF2.5R at the metadata gate. Do not download the 6,716 digitised images:
without the physical register contents they supply no machine-readable stock
labels, and catalogue openness does not establish image reuse/training rights.

This is a high-value data gap, not evidence that stock signal or latent modes
do not exist. A future official transcription, digitisation, licensed export
or separately approved physical-copy workflow could reopen only a bounded
register-content/join audit.

## Claim ceiling

The result establishes catalogue structure and current machine
inaccessibility only. It does not establish per-negative stock identity,
photographer/process/scanner groups, pixel rights, stock signal, an identified
operator, training eligibility, LSM eligibility, calibration or authenticity.
