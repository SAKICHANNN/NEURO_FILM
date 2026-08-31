# U7.8B Resumable Three-Stock Batch Contract

## Question

Can the existing U7.8A multi-input transaction recover from an application-level
interruption without rerendering already completed inputs, while retaining full
hash validation and one create-only final publication?

This is a product recovery question. It is not a new colour algorithm, stock
calibration experiment, installer, release or adjacent recipe-bundle format.

## Parent and non-goals

U7.8B is a prospective successor to the frozen U7.8A result. U7.8A remains
unchanged and authoritative for an uninterrupted 100-small-image transaction.
U7.8C independently owns the six-Canon real-scale resource audit. U7.8B does not
read its inputs, reuse its thresholds or modify its files.

The leaf must not modify the single-input renderer, product look catalog, colour
parameters, recipe schema, U7.7 recovery formats, U7.8A implementation or any
RAW/HDR/OpenEXR path. Velvia 50, Portra 400 and Ektar 100 continue to denote
three user-selected `film-inspired / Look Approximation` outputs only.

## Workspace contract

The caller provides a dedicated workspace path and a final destination path.
They must be distinct siblings under one existing parent on Windows. This makes
the terminal `os.rename(workspace, destination)` a same-parent, no-replace
publication and avoids copying a completed batch.

A sibling lease file is held through a nonblocking Windows process-lifetime
byte-range lock. The OS releases the lock on process exit, so a crash leaves no
unsafe stale-owner decision. A concurrent invocation rejects before workspace
mutation or rendering. The open lease handle is outside the workspace and
therefore cannot block the terminal workspace rename.

The workspace contains:

- one static `resume.json` identity with no machine-local paths;
- one atomically updated `checkpoint.json` completed-job ledger;
- zero or more complete child directories, one per canonical job ID;
- no other persistent members before completion;
- one final aggregate `batch.json` only when every job is complete.

The state binds the canonical job/hash set and a one-way input-path binding per
job, destination through a one-way path hash, exact
profile/statistics/guardrail hashes, render parameters, software commit and
core-file hashes. Raw manifest row order is intentionally non-semantic; the
same rows in reverse order produce the same state, while any job, path or input
identity change rejects. A resume under different code or inputs is rejected;
completed work is never silently reinterpreted.

Each completed job directory contains exactly the existing child manifest,
three RGB16 PNG outputs and three strict recipes. The job is first built under a
reserved sibling transient, fully validated, then published by no-replace rename
to its canonical job directory. Only after that publication may the atomic
checkpoint ledger advance. Before a job can be skipped, the implementation
revalidates every member, hash, recipe, input, output path, profile and software
identity. A complete canonical child omitted by a stale ledger is reconciled
only after full validation; a ledger claiming a missing, truncated or tampered
child fails closed without deletion or rerender. Counts and status are never
trusted as artifact evidence.

Names beginning with `.u7-8b-` are reserved transient children of this dedicated
workspace. Only after the static workspace identity and exclusive lease both
validate may the runner remove a stale reserved transient from an interrupted
current-job render. Symlinks, reparse points, multiply linked files and
unexpected members are rejected.

## Pause, resume and publication

An optional `maximum_new_jobs` value in `1..100` limits the amount of new work in
one invocation. Reaching the limit returns a portable progress receipt, keeps the
validated workspace and publishes no destination. Omitting it completes all
remaining jobs.

When all jobs are complete, the runner writes the aggregate relative-path
receipt inside the workspace, revalidates all inputs/configuration/checkpoints,
then renames the whole workspace to the absent final destination. A destination
that appears late is preserved and the unchanged workspace remains resumable.

## Formal test

The committed-head formal fixture uses the unchanged U7.8A deterministic 100
small images. The first invocation stops after 37 new jobs. The second invocation
must reuse exactly those 37 and render only the remaining 63. Its complete
output/recipe/manifest hash set and path-independent scientific identity must
match one uninterrupted execution under the same source, parameters and input
hashes. Forward and reverse job enumeration must reproduce the same report.

Negative controls cover a concurrent writer, a complete child with stale ledger,
a ledger claiming a missing child, a tampered completed child, software/core
drift, a stale reserved transient, a crash after the last child but before the
final receipt, late foreign final destination, input/config drift, hardlinks,
links/reparse points and unexpected workspace members.

## Claim ceiling

Passing establishes only private Windows/Python resumable mechanics for three
deterministic film-inspired Look Approximations. It does not prove hard-power-loss
durability during an executing child render, 24MP throughput, installer or public
API readiness, stock distinguishability, calibrated/physical stock response,
release readiness or product value.
