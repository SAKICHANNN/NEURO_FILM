# U1.6G4D Halation Capability-Binding Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4D`

**Status:** frozen / implementation ready

**Config SHA-256:** `b641d9d7373a4e7b1556622f79a092184a3ff305be0d062d21ae3d84504ec74b`

## Purpose

G3 accepts a set of capability names so it can model missing prerequisites,
but a caller can currently supply the required strings without proving that a
provider exists. G4A and G4C now pass independently. G4D binds those concrete,
versioned exported implementations to the G3 vocabulary.

This is a static readiness proof. It does not execute a halation graph.

## Frozen binding

Add immutable binding records with exactly these providers:

| G3 capability | Implementation version | Module | Symbol |
|---|---|---|---|
| `coordinate_exact_gradient_window` | `coordinate-gradient-window-v1` | `src.filmfx.gradient_window` | `coordinate_gradient_window` |
| `row_chunked_global_stage_builder` | `shape-stable-global-resample-v2-explicit-f32` | `src.filmfx.global_resample` | `build_chunk_invariant_global_stage_from_rows` |

The resolver must verify each imported provider is callable and its actual
`__module__`/`__name__` and version constant match the binding record. It then
returns the exact capability names for `build_halation_resource_plan`.

Existing caller-supplied capability semantics and the default empty set remain
unchanged so prior G3 fingerprints/evidence are not rewritten.

## Scope

Modify only `src/filmfx/halation_dag.py`, established package exports and
focused tests. Do not modify effects, renderers, G4A/G4C providers, profiles,
recipes or CLI paths.

## DoR

- G3 static inventory/lifetime result passes;
- G4A coordinate gradient passes;
- G4C chunk-invariant row staging passes with 505 full tests;
- clean pre-contract HEAD is `c707e28`;
- no effect/renderer/training/download process is active.

## DoD and gates

1. resolver returns exactly the two frozen immutable binding records;
2. module/symbol identity, callability and version match are verified;
3. resolved capability names equal `REQUIRED_INTEGRATION_CAPABILITIES`;
4. both physical-colour and density plans become statically
   `integration_ready=true`, with no missing/unresolved capability nodes;
5. graph inventories, topology, blur/percentile counts, finite/global
   classification and resource arithmetic remain unchanged from unbound G3;
6. default and gradient-only G3 behavior remain frozen/closed;
7. forged/unknown/incomplete bindings fail or remain missing;
8. focused tests and the complete CPU suite pass.

## Branches

- **Pass:** record G3 static readiness and freeze U1.6G4E, the simplest staged
  density-family executor contract.
- **Provider/version mismatch:** remain integration-closed; fix the binding,
  never bypass it with strings.
- **Graph/resource drift:** close and investigate before any executor contract.

## Claim ceiling

A pass proves only that G3's two prerequisite names resolve to the passed G4A
and G4C providers and that its static plan has no missing capability. It does
not prove graph execution, effect parity, visual quality, physical accuracy,
renderer integration, total memory, streaming decode or 100MP readiness.
