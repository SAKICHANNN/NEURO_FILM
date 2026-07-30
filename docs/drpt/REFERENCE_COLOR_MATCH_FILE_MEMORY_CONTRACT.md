# P154 Reference-Match File Memory Contract

## Question

Do the P151-P153 lifetime changes reduce measured peak process-tree RSS on the
real `match_reference_files` path without changing durable output, recipe or
semantic report content?

This is a local Windows/Python engineering comparison. It is not an algorithm
quality result, a platform-runtime result, a RAW/HDR/video claim, or evidence
that a D-PCT candidate is product-admissible.

## Frozen comparison

- Baseline: `34259086fbbd6e479753a9f72bf77f2e0d463e93`, the P150 payload before
  the three lifetime changes.
- Candidate: `de57918e23568ad6321262903addfac7f2349d3c`, containing only the
  P151 reference release, P152 source release and P153 rejected-candidate
  release in the functional diff.
- Workload: one deterministic 3000 x 2000 PNG8 reference and one deterministic
  3000 x 2000 PNG8 source, one 16-bit PNG output, one recipe and one report.
- Guard: the default fail-closed policy, which must produce
  `identity-fallback` and therefore exercises the P153 branch.
- Order: baseline, candidate, candidate, baseline. Each execution is a fresh
  process from a detached worktree at the exact commit.
- Measurement: sample the complete worker process tree every 20 ms using the
  corrected G4G/G4I method. The report records process identities, wall time,
  peak RSS, exact artifact hashes and cleanup facts.

The generated inputs are shared read-only bytes across both revisions. Output
directories are revision/run-specific. Report paths are normalized to logical
roles before semantic hashing because absolute run directories differ by
construction.

## Preflight and failure behavior

The parent fails closed without launching workers when available physical
memory is below 12 GiB, free output-volume space is below 2 GiB, or an unrelated
process exceeds 8 GiB RSS. Each worker has a 300 second timeout. Timeout or
failure terminates its observed process tree, publishes no successful run, and
cannot be retried automatically under a changed contract.

The harness owns only its exact ignored output directory and detached
worktrees. It must remove its worktrees in `finally`, leave no observed worker
alive, and report any staging temporary left by the product transaction.

## Frozen gates

All four executions must:

1. exit successfully within 300 seconds with no orphan worker and no staging
   temporary;
2. produce byte-identical output PNG and recipe JSON within and across
   revisions;
3. produce an identical normalized semantic report within and across
   revisions;
4. record the default `identity-fallback` action;
5. yield candidate median peak RSS at least 64 MiB below baseline and no more
   than 92% of baseline median peak RSS.

Failure of either memory threshold is a negative result; the threshold will not
be adjusted after observing the run.

## Interpretation boundary

A pass measures this exact 6 MP relative-SDR file workload on this host. It does
not establish allocator-independent savings, 100 MP readiness, mobile/Apple
memory behavior, production latency, arbitrary batch scaling, or any change to
the current A1/A4/A5 rejection. No production module, public contract, producer
schema, FilmFX path or main-project file is modified by this experiment.
