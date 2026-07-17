# U5.R1B FilmStyleSafe failure-suite design

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B`

Decision: **design/membership contract frozen; generators not yet executed**

## Scope

R1B freezes a leakage-safe membership contract for a future synthetic/real
failure suite that supports FilmStyleSafe evaluation. It does not recruit
participants, train SCIS, populate hidden splits, or claim risk bounds.

Authoritative machine contract: `configs/filmstylesafe_r1b_contract_v1.json`  
Validator: `src/eval/filmstylesafe_r1b.py`

## Inventory (design)

Transform families retained for A0 development include identity, safe-lab /
safe-rich globals, owner-anchor strength and containment paths, basic
WB/contrast/saturation, bounded CCM, monotone+LUT, SepLUT/NILUT challengers,
the closed archive-matrix transplant negative, and the external spektrafilm
Look Approximation control.

Failure families cover the R1A chromatic severe ontology plus an explicit
`id11_red_speckle_regression` case. Legitimate-local hard negatives remain
separate from automatic severe labels.

Owner schemes `53/55/56/09/01` are full anchors; `33/03/02` are smoke cues.
`53/55/56` are a strength-path negative control: splitting them into three
stable “modes” disqualifies a method. They are not stock truth.

All currently seen U4 / owner-anchor / RF2.C0 / ID11 origins remain A0-only.

## Membership validation

Suite cards must declare role, split, parent-scene grouping keys, transform and
failure families, origin and hashes. Synthetic failures require an explicit
non-generative operator card with frozen parameter/input hashes. A0↔A1 leakage
audits reject shared parent/source/hash keys and reused transform/failure
family pairs.

## Authorization boundary

Allowed now: design, schema, local inventory cards, future synthetic operator
cards under this contract.

Forbidden now: external recruitment, paid resources, model training, hidden
split population, risk/population claims, generative RGB synthesis as
Style-safe evidence.

## Verification

Targeted R1B tests plus the repository CPU suite must pass before this design
leaf is marked complete. Generator corpus execution is a later leaf after this
contract commit.

## Next

Implement a bounded A0 inventory pack and/or explicit synthetic operator
prototypes under this contract, still without hidden splits or participants.
U5.R1C remains blocked until an executed leakage-safe suite exists.
