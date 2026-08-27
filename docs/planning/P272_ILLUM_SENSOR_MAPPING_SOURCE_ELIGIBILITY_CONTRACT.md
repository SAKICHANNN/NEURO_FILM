# P272 — controlled illumination/sensor RAW source eligibility

Status: **frozen before dataset-object request**  
Date: 2026-08-27

## Question

Can the official SamsungLabs illumination/sensor-mapping release open the final
paired/capture-time candidate slot with exact reproducible assets and rights
compatible with the product research cycle?

## Frozen source

- official repository head `f649f85cddaafdcdfd92bc32cea2415d3b72fd1e`;
- exact `README.md` and `LICENSE.md` Git objects and SHA-256 identities;
- README-declared 390 illuminants, four cameras, 18 scenes and the official
  Sync dataset link;
- zero dataset-object, pixel, model, training or inference requests.

## Gates

The source must simultaneously provide: exact official identity; a genuinely
new controlled physical observation; explicit scene/illumination/camera groups;
an exact dataset manifest with asset sizes/checksums; anonymous reproducible
access; and rights that allow the commercial product research cycle.  Failure
of either manifest reproducibility or commercial rights closes the source
before data access.  A repository code/data licence is not silently broadened
beyond its exact text.

## Stop rule and ceiling

Failure closes P272 as a source-operational/rights result, not as a scientific
negative.  Do not request the Sync payload, accept additional terms, use a
mirror, train, infer, fit an operator or consume candidate 3.  A pass would open
only a separately preregistered bounded member-integrity audit.
