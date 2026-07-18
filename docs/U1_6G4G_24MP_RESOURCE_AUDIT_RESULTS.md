# U1.6G4G 24MP Resource and Orchestration Audit Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4G`

**Decision:** measured local 24MP pass; renderer integration and 100MP remain closed

## Reproducibility

- frozen config SHA-256: `33ecdfa3efa4444448529183d8f0ac5327706e0685d293d15e3d280e57458392`;
- initial harness commit: `0a4f854`;
- process-tree RSS fix commit: `a60fd88501e372e8d336f0a34b839c9310bba4f0`;
- authoritative ignored report: `outputs/u1_6g4g/formal_a60fd88.json`;
- report SHA-256: `29a998b4268427c147aaeb73436ca7ca9755a803796e0ef70c2f8290816eef2a`.

The earlier `formal_0a4f854.json` is rejected as memory evidence. It sampled
only the Windows virtual-environment launcher and reported an impossible 4.9MB
peak. The corrected harness samples the full launcher/worker process tree and
requires observed peak RSS to exceed the 960,000,000-byte known-live-array
floor. No experimental gate or operator changed.

## Failure and ownership result

The injected `before_executor` worker:

- exits nonzero without timeout;
- publishes no result or temporary report;
- leaves zero observed process-tree members alive;
- records a plausible 342,384,640-byte construction-phase peak.

Both successful workers also leave zero orphan or temporary files. Pixel
scratch remains zero.

## 24MP result

| Metric | Frozen gate | Run 1 | Run 2 |
|---|---:|---:|---:|
| peak process-tree RSS | `<=4 GiB` | 1,033,719,808 B | 1,055,174,656 B |
| known live-array floor | sanity requirement | 960,000,000 B | 960,000,000 B |
| worker total | `<=90s` | 29.69s | 28.64s |
| executor | `<=75s` | 27.66s | 26.73s |
| row composite | `<=15s` | 0.73s | 0.67s |
| orphan workers | `0` | 0 | 0 |

The peak is about 0.963-0.983 GiB. This includes the 288MB input, 384MB layer
output, 288MB composite and measured Python/operator overhead on the current
Ryzen host.

Executor metadata reports:

- input bytes: 288,000,000;
- RGB+alpha layer bytes: 384,000,000;
- global contexts: 12,791,340;
- declared tile workspace: 83,759,104;
- maximum source window: `81x6000x3`, below full height;
- persistent derived full scalar bytes: 0;
- scratch bytes: 0.

## Determinism

Both independent child processes repeat exactly:

- source: `d80a3169...`;
- layer RGB: `77a2195c...`;
- layer alpha: `215794e6...`;
- composite: `d9b62156...`;
- metadata: `0010ef76...`.

The small prerequisite tests also prove analytic source chunk invariance and
row-screen composite byte identity with the current full compositor.

## Verification and branch

- 20 focused executor/harness tests pass;
- 536 complete CPU tests pass in 24.44 seconds;
- no renderer, effect entry point, profile, recipe or CLI file changed.

G4G passes only its measured local engineering claim. It opens design of a
separate research adapter/orchestration contract. It does not authorize direct
renderer wiring. A later adapter must prove existing-renderer parity, ownership
of input/layer/composite lifetimes, deterministic failure cleanup and opt-in
isolation before any 100MP total-path audit.

## Claim ceiling

This is local 24MP synthetic executor-plus-row-composite evidence. It is not a
complete-renderer, file-I/O, M5/RTX, cross-platform, 100MP, physical, stock,
default or production-integration result.
