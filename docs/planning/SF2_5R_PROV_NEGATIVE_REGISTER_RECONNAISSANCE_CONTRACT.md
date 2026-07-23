# SF2.5R PROV negative-register reconnaissance contract

Date frozen: 2026-07-23

Node: `ULT > SF2 > SF2.5R`

Status: **frozen after bounded source discovery and before the reproducible
two-pass metadata audit**

## Question

Can the Public Record Office Victoria catalogue provide a machine-accessible
join between:

- VPRS 17684 digitised Country Roads Board master negatives; and
- VPRS 17690, the linked photographic negative register described as tracking
  film stock?

If the register contents are accessible, this could offer unusually valuable
same-agency, repeated-photographer/content-domain connectivity. The catalogue
description alone is not a stock label and no image may be acquired at this
stage.

## Discovery evidence

The official PROV API is explicitly offered for raw-data research and permits
non-commercial API use under CC BY-NC. Bounded discovery found:

- VPRS 17684 has 6,832 item records, 6,716 catalogued as digital;
- VPRS 17690 has 30 item records;
- sample register items are catalogued as physical volumes;
- the external Research Data Australia record states that VPRS 17690 is the
  register to VPRS 17684 and was used to track film stock.

These are source-discovery facts, not a completed audit or evidence that
per-negative stock entries are online.

## Frozen audit

Run the exact official API queries twice:

1. all VPRS 17690 item metadata, maximum 100 rows;
2. aggregate VPRS 17684 item counts and `format` facet, zero rows.

Remove only response timing (`QTime`) before canonical JSON hashing. Require
two byte-identical normalized passes and record:

- total item counts;
- format counts;
- VPRS 17690 consignment counts;
- exact item identifiers/titles/negative-number ranges;
- presence of any IIIF manifest or digital-object field;
- rights/access statements;
- source URLs and observation date.

No image, TIFF, VEO payload, physical-copy order, account login or external
message is permitted.

## Branches

- If VPRS 17690 exposes digital register pages or structured per-negative
  stock fields, open only a separately frozen bounded register-content and
  join audit.
- If all register items are physical/catalogue-only, close SF2.5R as a
  promising but currently machine-inaccessible stock-label source.
- API instability or non-identical normalized passes produces
  `unresolved_source_instability`; retry is bounded and no pixels open.

Either outcome leaves training, operator fitting, LSM, stock claims and VPRS
17684 pixel acquisition forbidden.

## Definition of done

Two-pass normalized evidence, branch decision, claim ceiling, tracker/registry
propagation, targeted tests and a scoped commit.
