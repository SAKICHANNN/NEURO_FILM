# U1.6G3 Halation Field-DAG and Lifetime Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G3`

**Status:** frozen / implementation ready

## Purpose

U1.6G1 proves one original-coordinate coarse grid and U1.6G2 proves exact
two-pass float32 percentiles. They are necessary but not sufficient to stage
physical or density halation. U1.6G3 freezes the actual effect-family DAGs,
classifies every blur as finite-halo or global-grid for the requested geometry,
and computes explicit lifetimes and resource categories before execution code
is allowed.

This is a planning primitive, not an effect adapter.

## Current-code inventory

At default `local_diffusion=1.0`:

| Family | Percentiles | Blur fields | Finite-halo | Global-grid |
|---|---:|---:|---:|---:|
| colour physical | 2 | 8 | `local_abs`, `red_near`, `red_mid`, `green_near`, `green_mid` | `local_mean`, `red_tail`, `red_glare` |
| density | 1 | 6 | `local_abs`, `near`, `mid` | `local_mean`, `tail`, `glare` |

Classification is not hard-coded by name. It uses the same direct-radius,
target-coarse-sigma and image-geometry rule as U1.6G1. A parameter change may
move a field between classes and must produce a different plan fingerprint.

## Graph model

Add one module under `src/filmfx/` with:

- immutable node, graph and resource-plan dataclasses;
- strict topological validation and deterministic graph fingerprints;
- concrete builders for `physical-colour-v1` and `density-v1` matching the
  current `effects.py` dependencies;
- liveness analysis from producer creation through final consumer;
- separate accounting for:
  - caller-owned external input;
  - required returned output;
  - persistent in-memory context/coarse stages;
  - per-window transient workspace;
  - temporary disk scratch;
  - scalar/reduction metadata.

Node operators are restricted to declared vocabulary: external, pointwise
stream, gradient, exact percentile, finite-halo blur, global-grid blur, combine
and output. Unknown operator or storage classes fail closed.

## Required graph semantics

The colour graph must preserve separate source and high-source percentile
branches, background mean/deviation, four red and two green blur paths,
visibility/skin control, energy/colour and alpha output. The density graph must
preserve its independent source, background, four blur paths, visibility and
tinted screen/alpha output. The two graphs may share planner code but never
silently share effect-family semantics.

Pointwise fields are streams/windows unless a downstream global operator makes
materialisation unavoidable. A finite-halo blur remains a window operator.
Only a U1.6G1 global-grid blur may retain a coarse field. Percentile histograms
are transient G2 workspace and the resolved percentile is scalar context.

## Hard integration readiness rules

The plan must report `integration_ready=false` while any required capability is
absent. Current known missing capabilities:

- U1.6G1 can stage from a full ndarray but cannot yet consume a repeatable
  row-chunk/window factory. Derived source, deviation and weighted-source fields
  therefore cannot be global-staged without materialising a hidden full field.
- weighted source fields depend on `np.gradient(y)`, but there is not yet a
  coordinate-exact original-boundary/chunk-boundary gradient-window primitive.

The planner must name these as `row_chunked_global_stage_builder` and
`coordinate_exact_gradient_window`. It is forbidden to hide a full field,
mislabel it as bounded context, accept chunk-edge gradients, or integrate an
effect before both capabilities pass later child contracts.

## Resource rules

- Source shape, channels, float dtype, tile size and diffusion parameters are
  explicit plan inputs.
- Every allocation has exact bytes, creation step, final consumer and release
  step.
- Peak context, workspace, scratch, external and output bytes are reported
  separately; they must not be collapsed into a misleading single bounded-RAM
  claim.
- Integer overflow, impossible shapes, duplicate nodes, unknown dependencies,
  cycles, use-before-produce and release-before-last-use fail closed.
- A 100MP dry plan is allowed as arithmetic evidence only. It cannot establish
  executable 100MP readiness.

## DoR

- U1.6G0 is closed with an explicit DAG prerequisite;
- U1.6G1/G2 pass independently;
- current `physical_halation_layer` and `density_halation_layer` are unchanged;
- clean pre-contract HEAD is `f9360a6`;
- no project renderer, download or training process is active.

## DoD and frozen gates

1. concrete graph inventories match exactly 8/2 colour and 6/1 density
   blur/percentile calls in current code;
2. default medium-image classification matches the table above;
3. changing geometry/diffusion deterministically reclassifies nodes and changes
   the fingerprint where appropriate;
4. all dependencies are topologically valid and every retained allocation is
   released only after its final consumer;
5. peak resource accounting is independently recomputed by tests;
6. 24MP and 100MP arithmetic plans are deterministic, overflow-safe and keep
   external/output/context/workspace/scratch categories separate;
7. both plans remain `integration_ready=false` with both exact missing
   row-chunked-stage and coordinate-gradient capabilities;
8. malformed graphs, storage classes, cycles, duplicate/unknown nodes and
   invalid shapes/parameters fail closed;
9. focused tests and the complete CPU suite pass;
10. no existing effect, renderer, profile, recipe or CLI output changes.

## Branches

- **Pass:** retain the planner and open the narrowly specified
  row-chunked-global-stage builder before effect execution.
- **Inventory mismatch:** repair the graph from current code; do not reinterpret
  or simplify effect semantics.
- **Hidden full field or false readiness:** reject the implementation even if
  resource totals look acceptable.
- **Resource arithmetic failure:** close the plan; do not weaken overflow or
  category separation.

## Commit and rollback

Contract, implementation and evidence use separate scoped commits and pushes.
The implementation is additive under `src/filmfx/` with focused tests. Reverting
it leaves G1/G2 and all current renderer/effect paths unchanged.

## Claim ceiling

A pass proves a faithful, validated and resource-audited static execution plan
for the two current halation families. It does not execute either effect, prove
pixel parity, physical realism, renderer integration, streaming decode,
calibrated/stock response, bounded total memory or 100MP performance.
