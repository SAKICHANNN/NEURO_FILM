# U1.6G4J Staged-Density Recipe Integration Contract

**Date:** 2026-08-26

**Node:** `ULT > U1.6 > U1.6G4J`

## Question

Can the retained `staged-density-halation-v1-defaults` operator enter the
existing deterministic renderer as an explicit, non-default research effect,
with a new strict recipe version that replays exact output bytes, while every
legacy render and recipe version remains unchanged?

This is an integration and replay experiment. It does not re-score G4E/G4F,
change the operator, or promote physical, stock, calibrated, or default
halation claims.

## Frozen boundary

- existing render-profile v1 bytes and semantics remain unchanged;
- recipe v1/v2/v3 validators and schemas remain unchanged;
- the existing `simple` and `physical` halation routes remain unchanged;
- the new route is selected only by `--halation-model staged-density-research`;
- it requires `--use-render-profile`, `--halation 1.0`, locked controls, no
  halation preset, and an explicit positive tile size;
- the operator version is exactly `staged-density-halation-v1-defaults`;
- source/coarse/composite row chunks are fixed at `64/7/64`;
- the generated recipe uses `kmcfm.render-recipe.v4` and binds the operator
  version, tile size and fixed row chunks;
- replay rejects any version, model, strength, control, preset, parameter or
  tile-size mutation before returning pixels;
- output remains an sRGB Look Approximation and never becomes calibrated.

## Formal workload

The audit creates one deterministic 192x256 RGB8 fixture in owned scratch and
runs four fresh processes in `baseline/candidate/candidate/baseline` order.
Candidate runs use the versioned render profile, 16-bit PNG, tile size 64,
the exact staged-density operator, recipe writing, and independent recipe
replay to a second file.

## Gates

1. legacy baseline output and recipe bytes are exact across the two baseline
   processes and keep their existing schema/model semantics;
2. both candidate outputs and both candidate recipes are byte exact;
3. each candidate recipe validates as v4 and independently replays to the
   exact encoded output bytes;
4. direct CLI output equals direct invocation of the unchanged staged-density
   executor/compositor on the same safe-Lab base;
5. all outputs are finite, bounded, non-empty and carry the standard sRGB ICC;
6. disabled/default renders remain unchanged;
7. wrong model version, non-unit strength, expert control, preset, missing or
   changed tile size, changed chunk size and unknown fields fail closed;
8. owned scratch is empty after success and after an injected replay failure.

## Decisions

- **Pass:** retain the new route as a private, explicit opt-in research effect
  with exact recipe/replay support. Default and calibrated claims remain
  unchanged.
- **Fail:** remove the route and v4 schema, preserve G4E-I as isolated research
  evidence, and do not relax gates or alter the operator on this cohort.

## Claim ceiling

At most this leaf proves private Windows/Python deterministic integration and
exact replay of the already-retained staged-density Look Approximation effect.
It does not establish physical film accuracy, named-stock response, scanner or
process calibration, user preference, arbitrary platform parity, public
package readiness, or a new default.
