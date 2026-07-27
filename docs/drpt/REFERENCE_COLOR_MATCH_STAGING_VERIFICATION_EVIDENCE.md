# Reference Color Match Staging Verification Evidence

Date: 2026-07-28

Status: **restart verification complete; delivery remains closed**.

## Decision

P34 closes the stale-object gap after P33. A later process can verify one
committed staging run using only:

- the P33 report path;
- the expected report SHA-256 retained by the caller;
- the expected canonical P33 run ID retained by the caller.

The verifier does not write, move, render, promote, deliver or apply anything.

## Verification contract

`verify_external_core_staging_v1`:

1. validates the two expected identities before file access;
2. reads at most 16 MiB of report data;
3. verifies exact report bytes before parsing;
4. parses the strict P33 schema and recomputes its canonical run ID;
5. requires the report to remain at its transaction-bound absolute path;
6. rereads every ordered output and recomputes its exact file SHA-256;
7. emits a canonical `verified-staging` binding containing report, run,
   authorization, reference-intent, receipt, path and file identities.

The claim ceiling is `verified-staging-not-delivered`.

P33 now records absolute output/report paths so the restart check does not
depend on the later process working directory. This is a v1 hardening, not a
producer or pixel-format change.

## Adversarial evidence

Nine P34 tests prove:

- clean restart verification matches the P33 run and every committed file;
- report byte append fails its expected hash;
- a wrong or malformed expected run ID fails;
- output byte mutation fails;
- a missing output fails;
- copying the report to another path fails even when the caller supplies the
  copied file's new hash;
- mutated delivered state, source order, applied claim ceiling and canonical
  verification identity fail validation.

## Verification

- P33+P34 dedicated: 18 passed;
- combined P27-P34 plus local transaction/composition adjacency: 144 passed;
- compileall and diff check: passed;
- complete CPU suite: 1254 passed, one skipped and the unchanged 36
  isolated-worktree failures;
- no P34 or reference-colour-match test failed.

The 36 failures remain missing ignored evidence and legacy Windows checkout
hash conditions, not new implementation regressions.

## Latest-main propagation

- common base: `c03c321`;
- main snapshot: `b71fb68`;
- consumer implementation: `f1e8d35`;
- consumer changed paths: 164;
- main changed paths: 100;
- exact path overlap: zero;
- merge tree: `d9d6fe5b740231e518dc937eac437250c94fec05`;
- a fresh detached synthetic merge passed all 144 selected tests and was
  removed.

Main's concurrent modified Kodak AA1 files, `.codex/` and `tmp/` were not
touched. D-PCT was observed at stable `c688c32`; no producer interface was
consumed or changed.

## Handoff

P34 supplies durable evidence suitable for a later delivery/composition
decision, but does not authorize one. A real external run still requires a
fixed D-PCT invocation and A1/A4/A5 promotion. Any future delivery leaf must
bind the exact P34 verification ID and must not infer `applied` from file
existence.

