# U1.6G4G 24MP Measured Resource and Orchestration Audit Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4G`

**Status:** frozen / implementation ready

**Config SHA-256:** `33ecdfa3efa4444448529183d8f0ac5327706e0685d293d15e3d280e57458392`

## Purpose

G4F retains the staged density operator as a safe, non-empty opt-in research
candidate. G4G asks whether one fixed 24MP executor-plus-composite job has a
bounded, reproducible local resource envelope worth carrying toward a later
integration design.

This is not renderer integration. It does not exercise decoding, colour
management, file encoding, the complete effect stack, M5 hardware or 100MP.

## Frozen workload

- shape: float32 `4000x6000x3` (24,000,000 pixels);
- input: deterministic analytic highlight/gradient field version
  `u1-6g4g-analytic-highlight-field-v1`;
- input is allocated once and filled in 64-row chunks; no full coordinate grid;
- executor: fixed G4F operator, tile 512, source rows 64, coarse rows 7;
- output: required full `FilmLayer` RGB+alpha;
- composite: a separate full float32 RGB output, filled in 64-row chunks with
  the current screen-layer equation;
- no pixel-sized disk spill, image encoding or visual decision;
- two independent child-process runs.

Before the 24MP workload, a small committed test must prove the row composite
is byte/equation compatible with `composite_layers` within frozen float32
rounding and that source generation is chunk-invariant.

## Measurement method

A parent harness launches a fresh Python worker for each run and samples that
child's RSS every 20ms with `psutil`. The child reports phase timings, hashes,
executor metadata and boundedness. Parent wall time and peak child RSS include
input construction, executor output and row-stream composite lifetimes.

The worker report is written atomically only after success. A timeout or
injected failure kills/waits the child, removes partial public reports and
leaves zero orphan workers. Small JSON files are allowed; scratch pixel bytes
must remain zero.

RSS is an observed local process measure, not an allocator-perfect proof. The
report must also retain the executor's declared categories and compare observed
peak with known input/output byte floors.

## Frozen gates

Both independent runs must satisfy:

- peak child RSS `<=4 GiB`;
- total worker wall time `<=90s`;
- executor time `<=75s`;
- row-stream composite time `<=15s`;
- source, layer RGB, alpha, composite and metadata hashes repeat exactly;
- source/output finite and in `[0,1]`; alpha remains in `[0,0.26]`;
- executor reports zero persistent derived full-frame scalar bytes and zero
  scratch bytes;
- every maximum source window remains below the 4000-row full height;
- scratch pixel bytes zero and no partial/orphan worker after success/failure;
- focused tests and the complete CPU suite pass.

The 4GiB gate is deliberately far below the 32GB target-system capacity but is
not a claim that the full renderer fits in 4GiB. The timing gates are local
engineering screens on the current Ryzen host, not M5 or cross-platform SLAs.

## Failure probe

One separate worker receives a frozen `before_executor` injected failure. It
must exit nonzero, publish no successful result, leave no partial temporary
report and have zero surviving child process. The probe does not allocate the
executor outputs.

## DoR / DoD

DoR: G4F is frozen; current clean head is
`3d529617219f10f50f5bd01d4e4df8d25f86f0c1`; `psutil 7.2.2` is locally
available; no download, training, GPU or renderer change is opened.

DoD: committed worker/harness and tests; successful failure probe; two complete
24MP runs; repeat/resource decision; complete CPU suite; result propagation;
scoped commit/push.

## Branches

- **Pass:** retain measured local 24MP evidence and design a separate renderer
  adapter/integration contract; integration is not authorized by this pass.
- **RSS/time failure:** keep G4F effect evidence but close current orchestration;
  profile allocations before changing the operator.
- **Hash mismatch:** diagnose nondeterminism; no integration.
- **Failure cleanup failure:** repair harness ownership/atomicity; no large rerun
  until focused cleanup tests pass.
- **System pressure/preflight failure:** do not run; this is a transient blocked
  leaf, not evidence against the operator or the Ultimate Goal.

## Claim ceiling

A pass proves only repeatable local 24MP executor-plus-row-composite behavior
under the frozen synthetic workload and observed RSS/timing gates. It does not
prove complete renderer memory, file I/O, M5/RTX behavior, cross-platform
parity, 100MP readiness, physical calibration, stock response or product
promotion.
