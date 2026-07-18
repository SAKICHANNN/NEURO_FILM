# U1.6G4I 100MP Isolated-Adapter Resource Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4I`

**Status:** frozen / harness implementation ready

**Config SHA-256:** `70564a4dfce6e9e2c8f91ca54322787338be1fee99b9e0e7101cb61c80e43126`

## Purpose and boundary

G4H proved that the isolated direct-module adapter exactly reproduces the
current compositor while keeping its intermediate layer private. G4I asks one
narrow next question: can that exact adapter complete a deterministic 100MP
synthetic job within a bounded resource envelope on the current local Windows
host?

This is effect-only evidence. It excludes decode, colour management, safe-Lab,
other effects, encoding, the production renderer, CLI and recipe/profile
schemas. A pass does not authorize production wiring and is not an M5,
cross-platform or complete-renderer memory claim.

## Frozen workload

- exact shape: float32 `10000x10000x3`, or 100,000,000 pixels;
- deterministic analytic input generator reused unchanged from G4G, filled in
  64-row chunks without a full coordinate grid;
- G4H adapter version `staged-density-halation-research-adapter-v1`;
- executor tile 512, source rows 64 and coarse rows 7;
- compositor rows 64 and output margin 0;
- two independent fresh-worker runs;
- no pixel-sized disk spill, image encoding or new visual adjudication.

The known simultaneous live-array floor is 4,000,000,000 bytes: 1.2GB caller
input, 1.6GB private RGB+alpha layer and 1.2GB returned composite. This is a
sanity lower bound for process-tree RSS, not an allocator-perfect prediction.

## Measurement and preflight

The parent must reuse the corrected G4G process-tree RSS method and include the
Windows launcher and Python child. It samples every 20ms, records all observed
PIDs, kills/waits on timeout and accepts a report only after an atomic worker
write. The worker calls the G4H adapter directly and hashes the source, returned
RGB and recursively serialized metadata.

Before either large run, the harness must fail closed without launching it if:

- available physical memory is below 12GiB;
- any unrelated process has RSS above 4GiB;
- output volume free space is below 1GiB.

Preflight failure is transient system pressure, not negative operator evidence.
It must not publish a passing report or trigger an automatic retry loop.

## Frozen gates

Both independent runs must satisfy:

- peak complete process-tree RSS is at least the 4.0GB live-array floor and at
  most 8GiB;
- total worker time `<=240s`, adapter call `<=210s`, input construction
  `<=30s`;
- source, output and metadata hashes are repeat-identical;
- source/output are finite float32 RGB in `[0,1]`;
- metadata reports exactly one public ndarray and zero scratch bytes;
- no temporary report or orphan worker survives success, failure or timeout;
- focused tests and the complete CPU suite pass.

These local timing gates are engineering screens, not product SLAs. The 8GiB
ceiling is deliberately below the 32GB target-system capacity but does not
reserve memory for a complete renderer.

## Failure probe

A separate small worker receives `before_input_allocation` failure injection.
It must exit nonzero before allocating the 100MP input, publish no successful
or temporary result and leave no process-tree survivor. Adapter-internal failure
ownership is already covered by G4H and is not reinterpreted here.

## DoR / DoD

DoR: G4H is committed and pushed at `9c40be6`; the adapter remains isolated;
`psutil` is present; the current host reports about 44.6GB free physical memory
with no process above 1GiB; no download, GPU, training or production change is
opened.

DoD: committed versioned harness and focused tests; passing failure/preflight
checks; two complete 100MP runs; repeat/resource decision; full CPU suite;
result propagation; scoped commit and push.

## Branches

- **Pass:** retain local effect-only 100MP evidence; complete-renderer design
  remains a separate contract.
- **RSS/time failure:** retain G4H parity but close current 100MP orchestration;
  profile allocations before changing architecture.
- **Hash mismatch:** diagnose nondeterminism; no renderer integration.
- **Cleanup failure:** repair ownership and atomicity before rerun.
- **Preflight failure:** defer the run until local pressure clears while the
  Ultimate Goal continues on another ready leaf.

## Claim ceiling

A pass proves only repeatable local execution of the isolated G4H effect
adapter at exactly 100MP under this analytic workload. It does not prove the
complete render path, streaming I/O, physical halation, stock authenticity,
cross-platform parity, target-M5 behavior, production safety or default
promotion.
