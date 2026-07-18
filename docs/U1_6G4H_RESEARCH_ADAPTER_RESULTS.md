# U1.6G4H Research Adapter Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4H`

**Decision:** isolated adapter pass; production integration remains closed

## Reproducibility

- frozen config SHA-256: `f2b1829360767ee0bccbbca37ee608e5d2fda97f164937a415c5732413f48ea7`;
- adapter implementation commit: `002c7bf`;
- audit implementation commit: `70b0a52eb3d11ee1750f31eb04ac0ca096df4769`;
- ignored reports: `outputs/u1_6g4h/formal_70b0a52_{a,b}.json`;
- both reports are byte-identical at SHA-256
  `72a55c52644ea0d68602576ec411aaae4efb7f272dd9834b06edbae9aa2ff44a`.

## Implementation boundary

`src/filmfx/staged_density_adapter.py` is a direct-module research API. It:

- calls the existing staged-density executor;
- composites bounded rows by calling the existing `composite_layers` rather
  than duplicating the screen equation;
- returns one float32 RGB ndarray and frozen metadata;
- does not return or retain the internal RGB+alpha layer;
- uses no scratch files;
- is absent from `src.filmfx.__all__`, renderer, inference, profile, recipe and
  CLI code.

## Frozen matrix result

All 36 policies pass across two seeded non-divisible synthetic cases and the
hashed `u41-12` real case:

- two executor tile sizes per case;
- composite row chunks 17, 64 and 113;
- output margins 0 and 4.

For every policy:

- float32 output bytes equal the materialized current-compositor reference;
- rounded sRGB8 bytes equal the reference;
- input bytes and writeability state remain unchanged;
- repeated output and metadata match;
- recursively inspected metadata contains no ndarray/list/dict/set;
- public return contains exactly one ndarray;
- scratch bytes are zero;
- adapter and executor version identities match the frozen config.

There are eight unique output hashes because tile size and composite row chunk
do not alter pixels, while case and margin do. All 36 metadata hashes are
distinct because metadata intentionally records tile/row/margin policy.

## Failure and isolation result

- injected executor failure returns no partial result;
- injected second-row compositor failure occurs after one private row was
  written but returns no partial result;
- nonfinite compositor results fail closed in focused tests;
- current production references to the adapter: 0;
- sensitive renderer/inference/package-export/schema/profile changes since the
  pre-contract head: 0.

No separate visual promotion is needed: the adapter is byte-identical to the
G4F-reviewed compositor result. This does not extend G4F's visual claim.

## Verification

- 33 focused adapter/audit/executor tests pass before the formal run;
- 553 complete CPU tests pass in 25.51 seconds;
- current renderer, effect defaults, recipe/profile schemas and CLI remain
  unchanged.

## Branch decision

Retain the isolated direct-module adapter. A pass does not authorize production
wiring. The next legal leaf may freeze a 100MP process-tree resource and
determinism audit of this adapter. Only a later, separate complete-path contract
may consider decode/color/effects/encode integration.

## Claim ceiling

G4H proves exact current-compositor parity and private layer ownership for one
isolated Python research adapter. It does not prove production integration,
streaming decode/encode, complete renderer memory, 100MP, cross-platform
behavior, default promotion, physical calibration or stock style.
