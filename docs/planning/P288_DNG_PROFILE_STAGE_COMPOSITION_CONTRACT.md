# P288 — DNG Profile Stage Composition Contract

**Status:** frozen before P288 implementation, producer fixture deserialization,
callable import or composition output.
**Date:** 2026-08-27
**Node:** `ULT > mature RAW/DNG explicit mechanisms > P288`
**DRPT:** L2 / Mode C; consumer owns only the P288 paths named below.

## Question

Can the four already qualified, source-locked producer callables be composed
in the Adobe DNG SDK stage order without copying producer implementation into
consumer `src`, mutating caller state, changing a default loader or silently
crossing the XYZ D50 / linear-ROMM domain boundary?

The stages are fixed as:

1. `ProfileHueSatMap` (P254/R1EC), XYZ D50 in/out;
2. `ProfileGainTableMap v1` (P263/R1EJ), linear ROMM in/out;
3. one explicit scalar exposure multiplier;
4. `ProfileLookTable` (P274/R1ER), XYZ D50 in/out;
5. `ProfileToneCurve` (P271/R1EN), linear ROMM component-wise.

This is an interoperability/mechanical leaf, not a DNG render or quality
experiment. It does not reopen any real-file profile-chain failure.

## Immutable parents

- P254 evidence and exact R1EC callable/core/schema/fixture Git objects.
- P263 evidence and exact R1EJ callable/core/schema/fixture Git objects.
- P274 evidence and exact R1ER callable/core/schema/fixture Git objects.
- P271 evidence and exact R1EN callable/core/schema/fixture Git objects.
- Producer repository is read only. Every byte is obtained by exact
  `git show <commit>:<path>` into one owned system-temporary import root.
- No producer worktree import and no producer source copy into consumer `src`.

## Frozen composition fixture

- Shape: `2x2x3`.
- Initial linear-ROMM values, row major:
  `[[[.1,.2,.3],[.25,.5,.7]],[[.4,.3,.2],[.7,.6,.5]]]`.
- Initial XYZ D50 is computed only with the exact source-locked
  `ROMM_RGB_TO_XYZ_D50` matrix from the common R1EC/R1ER arithmetic core.
- HueSat payload: exact P254 producer fixture payload.
- PGTM payload: exact P263 producer fixture payload. Its `2x2` `image_area`
  and gain lattice are used unchanged.
- Exposure multiplier: exact float64 `0.75`.
- LookTable payload: exact P274 producer fixture payload.
- ToneCurve payload: exact P271 producer fixture payload.
- XYZ D50 to ROMM and ROMM to XYZ D50 conversions use only the exact common
  source-locked matrices. No clipping, gamut mapping, tone substitution or
  tolerance rescue is allowed.

## Independent parity path

The primary path calls the four versioned wrappers. The control path calls the
source-locked arithmetic functions directly with each wrapper's validated
parser output. Both paths use the same frozen matrix conversions and exposure.
Primary and direct outputs must be byte-identical float64. This establishes
wrapper interoperability only; it is not an independent DNG SDK pixel oracle.

## Gates

All gates must pass:

1. every bound parent evidence and producer Git object is exact;
2. all four Draft 2020-12 schemas validate their unchanged fixture payloads;
3. all four callables import only from the owned temporary source tree;
4. both common arithmetic-core objects are byte-identical;
5. the frozen initial ROMM and derived XYZ hashes are exact;
6. every intermediate stage is finite and its declared domain is respected;
7. wrapper-chain output is byte-identical to the direct-core chain;
8. output is owned, writable, C-contiguous float64 with exact `2x2x3` shape;
9. source arrays and all payload objects are unchanged;
10. wrong-rank/nonfinite input, nonpositive/nonfinite exposure, swapped stage
    payload and out-of-domain LookTable input all reject without output;
11. network reads, new RAW/DNG/pixel/target reads, persistent artifacts and
    temporary residue are zero;
12. fresh forward/reverse reports are byte-identical.

## Stop rule

Any identity, schema, matrix, domain, order, parity, ownership, invalid-control,
replay or cleanup failure closes P288. Do not copy or edit producer code, alter
fixture payloads, clip an intermediate, relax equality, replace exposure, add
stages, touch the default loader or rerun a real-file cohort as rescue.

## Claim ceiling

At most: private exact interoperability of four already qualified DNG profile
callables on one synthetic `2x2` composition fixture. No real-file stage
composition, complete/arbitrary DNG renderer, photographic or colorimetric
quality, Adobe renderer equivalence, default-loader integration, public
dependency/API/package/schema/capability, stock evidence, automatic matching,
product admission or candidate-3 change.

## Owned paths

- `docs/planning/P288_DNG_PROFILE_STAGE_COMPOSITION_CONTRACT.md`
- `configs/p288_dng_profile_stage_composition_v1.json`
- `scripts/audit_p288_dng_profile_stage_composition.py`
- `tests/test_p288_dng_profile_stage_composition.py`
- `docs/evidence/P288_DNG_PROFILE_STAGE_COMPOSITION_RESULT.json`
- later narrow tracker and agent-log propagation only.
