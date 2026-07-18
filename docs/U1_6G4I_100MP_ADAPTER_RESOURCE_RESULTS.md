# U1.6G4I 100MP Isolated-Adapter Resource Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4I`

**Decision:** local 100MP effect-only pass; complete renderer and production integration remain closed

## Reproducibility

- frozen config SHA-256: `70564a4dfce6e9e2c8f91ca54322787338be1fee99b9e0e7101cb61c80e43126`;
- frozen contract commit: `222c6a9`;
- audit implementation commit: `6d325df`;
- ignored report: `outputs/u1_6g4i/formal_6d325df.json`;
- report SHA-256: `25248cdf570f7207960ace1c38c3409e8c0e5985db5dc80157a70dc27ce8dfa7`.

## Frozen-workload result

The preflight passes with sufficient physical memory and disk, and no process
above the frozen competing-RSS threshold. The injected
`before_input_allocation` worker exits nonzero, publishes no report or
temporary file and leaves no orphan process.

Both fresh `10000x10000x3` float32 runs pass:

| Measure | Run 1 | Run 2 | Gate |
|---|---:|---:|---:|
| Process-tree peak RSS | 4,072,169,472 B | 4,065,918,976 B | 4.0GB floor; <=8GiB |
| Worker total | 131.759 s | 123.847 s | <=240 s |
| Adapter | 126.877 s | 119.498 s | <=210 s |
| Input construction | 3.564 s | 3.052 s | <=30 s |
| Parent wall | 132.187 s | 124.208 s | <=360 s timeout |

The peak is about 3.79GiB and lies just above the analytically declared 4.0GB
decimal simultaneous-array floor. It includes the observed Windows launcher
and Python child process tree rather than the invalid launcher-only measure
previously rejected in G4G.

## Determinism and ownership

The two workers repeat exactly:

- source SHA-256: `f15598be388f5efeb37faf82bca5b64c4165a1405d41259d1d0aa1d657016e6d`;
- returned-output SHA-256: `4819c8285b256d43b181cf610375fd44e13cf517f2efb6d301b63883c0ee89dd`;
- metadata SHA-256: `43b9f31cd262a32b12c0800b0a74d72b475fbfed7c39a4691375d90c225a597b`.

Both outputs are finite, bounded float32 RGB. Metadata records one public
ndarray and zero scratch bytes. All worker JSON writes are atomic; success and
failure leave zero temporary files and zero observed process survivors.

## Verification

- 23 focused G4G/G4H/G4I tests pass before the large run;
- 558 complete CPU tests pass in 33.12 seconds after the formal run;
- no renderer, CLI, inference, package export, profile or recipe file changed.

## Branch decision

Retain G4I as measured local 100MP evidence for the isolated density-halation
adapter. Do not wire it into production on this result. The next product leaf
returns to the remaining U1.2/U1.4/U1.5 colour-state and format gaps; any future
complete-renderer 100MP claim requires a separate contract including decode,
colour, the selected effect stack and encode lifetimes.

## Claim ceiling

G4I proves repeatable local execution of one isolated effect adapter at exactly
100MP under one analytic workload. It does not prove complete-renderer memory,
streaming file I/O, target-M5 behavior, cross-platform parity, production
integration, physical halation, stock authenticity or default promotion.
