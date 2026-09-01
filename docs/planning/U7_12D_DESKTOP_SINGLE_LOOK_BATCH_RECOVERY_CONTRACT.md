# U7.12D Desktop Single-Look Batch Recovery Contract

## Product question

Can the private desktop recover an interrupted two-to-100-photo export of one
explicitly selected film-inspired Look Approximation without rerendering
already completed photos, while preserving the current strict recipes and one
create-only final directory publication?

This is a restart-recovery feature for the existing U7.11A desktop batch. It is
not a colour algorithm, a calibrated or physical stock-response claim, another
three-look input-batch experiment, an installer, or a public API.

## Parent state, dependency, and ownership

- Frozen prescore HEAD: `f59b4e2fb9e87e1bcc7eaaef4b49b313474ab0f1`.
- U7.11A remains authoritative for uninterrupted single-look batch semantics.
- U7.8B remains authoritative for its distinct three-look-per-input resumable
  workspace; this leaf may reuse verified transaction primitives but must not
  reinterpret a U7.8B workspace or emit three looks per input.
- U7.12C is a hard implementation dependency: no U7.12D source, test, pixel, or
  formal run may begin until U7.12C reaches a committed terminal result and its
  parent regressions are stable.
- This preregistration owns only this contract and its matching config. It does
  not claim U7.12C source, audit, evidence, tracker, or agent-log paths.
- Existing `.codex/` and `tmp/` are foreign/untracked. Formal scratch must use
  an owned repo-relative subtree and remove it after each complete run.

## Existing defect and compatibility boundary

`ProductDesktopWorkflow.export_batch` currently creates a fresh UUID stage for
every attempt. On cancellation or a child failure it correctly removes every
owned completed image/recipe pair. That exact U7.11A cancellation-and-cleanup
behavior remains unchanged.

U7.12D adds a separate explicit resumable entry point. The caller supplies a
dedicated workspace and an absent final destination that are distinct siblings
under one existing parent on Windows. The resumable operation must never be
selected implicitly by the existing `export_batch` method. The desktop UI may
call the new entry point deliberately and derive one reserved sibling workspace
from the chosen destination; selecting the same inputs, look, amount, and
destination after restart resumes only if the complete workspace identity
validates.

## Frozen workspace identity

The static workspace state binds:

- canonical input order, each resolved path binding, complete-file SHA-256,
  byte size, filesystem identity, and basename;
- the one selected style ID, look amount, output PNG16 format, recipe schema,
  product profile/statistics/guardrail hashes, software commit, and the exact
  implementation-core identities;
- the representative preview state and source hash that authorized the batch;
- a one-way final-destination path binding without storing an absolute output
  path in the portable aggregate receipt;
- the claim label `film-inspired / Look Approximation` and explicit false
  calibrated/physical-stock flags.

Changing any bound fact rejects before rendering, deleting, reconciling, or
publishing. Reverse caller enumeration must canonicalize to the same identity.

The workspace is caller-dedicated and contains exactly a static state file, an
atomic checkpoint ledger, zero or more complete child directories, and a final
aggregate receipt only after every child validates. Reserved transient names
are private to this workspace. Links, reparse points, multiply linked files,
unexpected members, or a second writer fail closed.

## Child completion and resume semantics

One child is one canonical input and contains exactly its existing U7.11A
PNG16 output, strict recipe, and a small child receipt. A child is rendered in
an owned reserved transient, validated with the existing recipe verifier and
full-file hashes, then published by no-replace rename to its canonical child
directory. Only then may the atomic checkpoint advance.

Before skipping a completed child, every invocation independently revalidates
the workspace identity, current input bytes/filesystem identities, child member
set, image and recipe hashes, recipe input/style/look/profile/software/output
semantics, and child receipt. A complete child omitted by a stale ledger may be
reconciled only after full validation. A ledger that claims a missing,
truncated, tampered, foreign, or semantically inconsistent child rejects
without deletion or rerender.

An optional `maximum_new_jobs` integer in `1..100` provides a deterministic
pause for testing and bounded operation. A cancel event checked before each new
child and before final publication also returns a resumable paused receipt.
Both cases preserve all verified children, publish no destination, and leave no
owned partial current child. Child-process failure has the same preservation
semantics after attributable partial files are removed safely.

The workspace lease is a sibling Windows process-lifetime byte-range lock. A
concurrent invocation rejects before workspace mutation or rendering; process
exit releases the OS lock without trusting a stale owner record.

## Final publication and receipt

After all children validate, the implementation revalidates inputs,
configuration, code, checkpoint, member set, and current desktop session, then
builds a separate reserved flat publication stage. Image and recipe bytes are
copied create-only from the validated child directories, rehashed, and placed
under the exact U7.11A flat names; the exact U7.11A aggregate receipt is written
last. The implementation then performs one same-parent Windows no-replace
rename from the flat publication stage to the absent destination. A destination
that appears late is preserved, the owned publication stage is removed, and the
unchanged recovery workspace remains resumable. Directly renaming the
checkpoint workspace is forbidden because its resume/checkpoint and
child-directory topology cannot equal the U7.11A final directory.

The final directory and aggregate receipt must be byte-for-byte identical to
the unchanged U7.11A uninterrupted batch for the same inputs, style, amount,
software commit, and destination semantics. No resume-only field may enter the
final U7.11A receipt identity. Only after the renamed destination and every
member validate may the owned recovery workspace and lease be removed.

## Frozen formal roles and gates

The committed formal fixture uses eight existing deterministic small raster
inputs in canonical order, one fixed `portra_400` selection, and look amount
`0.625`. The uninterrupted control uses unchanged U7.11A. The resumable path
pauses after three completed children, then a fresh process resumes and renders
only the remaining five.

Success requires:

1. the paused destination is absent; exactly three fully validated children
   and no partial child exist;
2. the fresh resume reuses exactly three children and executes exactly five
   child render commands;
3. resumed and uninterrupted final member names, bytes, image hashes, recipe
   hashes, aggregate receipt, and strict replay results are exact;
4. forward and reverse caller enumeration yield one canonical scientific
   identity and deterministic progress ordering;
5. cancellation after child three, child-four nonzero exit, and an injected
   attributable partial child all preserve the same three validated children
   and remain resumable;
6. stale-ledger reconciliation, missing/tampered claimed child, input drift,
   style/look/profile/software/core drift, unexpected member, hardlink,
   reparse-point, concurrent writer, late foreign destination, and post-rename
   verification controls all fail closed with foreign artifacts preserved;
7. existing `export_batch` still cleans its stage on cancel/failure and emits
   byte-identical uninterrupted U7.11A results;
8. source files remain immutable, network reads are zero, all outputs are
   finite/bounded PNG16, recipes remain strict Look Approximation receipts, and
   owned recovery/publication transient residue is zero after terminal
   publication;
9. focused U7.8B/U7.11A/U7.12A/U7.12B/U7.12C regressions, Ruff, compile, JSON,
   diff, and committed-head replay pass.

## Stop rule and claim ceiling

Any formal gate failure closes this exact resumable single-look design. Do not
rescue it by changing the child count, pause point, style, amount, output
format, destination semantics, checkpoint acceptance, cohort, threshold, or
comparison. Do not weaken full-file hashing, strict recipe verification,
exclusive lease, source/session binding, or create-only publication.

A pass establishes only private Windows/Python restart recovery for one exact
eight-input desktop batch using one user-selected deterministic film-inspired
Look Approximation. It is not hard-power-loss durability during an executing
child, arbitrary-input or 24MP performance, calibrated stock response,
physical-film reproduction, stock distinguishability, public API, installer,
release readiness, or product-value evidence.

## Commit and propagation order

1. Commit this contract/config alone before implementation or candidate runs.
2. After U7.12C terminal propagation, commit the additive recovery core and
   focused tests separately without changing legacy `export_batch` behavior.
3. Commit the formal audit before running it from clean committed HEAD.
4. Preserve raw forward/reverse reports outside Git until evidence binding.
5. Commit evidence/tests, then propagate tracker and agent log in one short
   final window. Do not push.
