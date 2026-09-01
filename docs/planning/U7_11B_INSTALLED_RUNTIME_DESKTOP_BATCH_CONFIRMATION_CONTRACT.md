# U7.11B — Installed-runtime desktop batch confirmation

## Decision

U7.11B is the separately frozen confirmation that the unchanged U7.11A
single-look batch core remains exact when executed by the existing U7.9A/C
private installed runtime.  It does not add a renderer, product API, hidden UI
automation hook, or installer feature.

The evidence is deliberately split:

1. the unchanged installed `kmcfm-desktop.cmd` must start the real Tk mainloop
   from a foreign working directory under a hostile `PYTHON*` environment and
   close through its existing smoke path; and
2. the same installed `runtime\\Scripts\\python.exe -I` must execute an
   audit-owned ephemeral worker which imports the exact committed repository
   and calls the unchanged `ProductDesktopWorkflow` batch core.

This proves installed-runtime batch-core composition.  It does **not** claim
native multi-select or Save-dialog interaction, which remains outside the
automation boundary closed by U7.10C.

## Frozen sources and roles

- Fresh runtime installation from the existing repo-relative
  `tmp/u7_9a_wheelhouse_v1` only.
- Exact U7.9A/C installer, requirements and runtime receipt semantics.
- Exact U7.10B installed desktop launcher evidence.
- Exact U7.11A batch evidence, core and UI identities.
- Two deterministic RGB8 inputs, one explicit `ektar_100` Look Approximation,
  amount `0.625`, PNG16 output, one strict recipe per child and one `batch.json`.
- Forward and reverse outer runs change only input enumeration order; canonical
  input order and all scientific identities must remain exact.

## Frozen execution

1. Refuse a dirty tracked tree or source identity drift before installation.
2. Create one fresh installed runtime below repo-relative `tmp`.
3. Verify all twelve wheel identities, `pip check`, receipt, source commit,
   requirements and launcher identities.
4. Start the installed desktop launcher from a foreign CWD with hostile Python
   variables; it must enter and leave the real mainloop without residue.
5. Run the fixed batch once with the current committed interpreter and once
   with installed Python `-I`, at the same source and destination paths.
6. Compare every PNG, raw strict recipe and raw `batch.json` byte-for-byte;
   strict replay must reproduce every installed PNG byte-for-byte.
7. With installed Python, execute frozen child-nonzero, cancel-after-current and
   late-foreign-destination controls.  Every control must fail closed, remove
   only identity-owned stages and preserve foreign content.
8. Remove the fresh installation and all owned case/process/controller roots.

The ephemeral worker is formal audit material only.  It is not installed,
exported, copied into the runtime, or promoted as a product interface.

### Additive execution lock (before any product render)

- One exact ephemeral bootstrap byte sequence and SHA-256 is generated once and
  reused unchanged by direct and installed executions.
- The installed worker must report `sys.executable`, and every captured batch
  child command `argv[0]` must equal the receipt-bound
  `runtime\\Scripts\\python.exe`; importing under installed Python alone is not
  sufficient.
- The fresh install is local-wheelhouse-only (`pip --no-index`).  The observed
  installer command ledger must contain no network locator and must point
  `--find-links` only to the exact local wheelhouse; this proves the formal path
  exposes no package-index/network request capability without pretending to
  measure operating-system traffic.
- Direct and installed executions serially reuse the exact same final source and
  destination paths.  The direct tree is removed only after its complete member
  size/SHA-256 snapshot still matches the owned snapshot.
- PNG, strict recipe and `batch.json` identities are compared raw.  No path,
  receipt, recipe or artifact normalization is permitted.  Report arrays alone
  are placed in their predeclared canonical key order before serialization.
- Source, requirements, all four launcher files and bootstrap bytes are checked
  before/after their respective execution boundary.

## Frozen gates

- fresh install and twelve-wheel identity exact;
- installed Python is CPython 3.12 and `pip check` succeeds;
- runtime receipt binds the current committed checkout, requirements and all
  four launcher files;
- installed desktop launcher enters the real Tk mainloop from foreign CWD;
- hostile `PYTHON*` variables do not enter child processes;
- direct and installed canonical order, PNG16, raw recipe and raw batch receipt
  are byte-exact;
- every installed output strictly replays byte-exact;
- U7.11A claim ceiling remains exact;
- installed child failure and cancel controls publish no destination;
- late foreign destination is preserved exactly;
- tracked sources remain exact and all owned residue is zero;
- two fresh complete reports are byte-exact after canonical report-only array
  ordering; raw artifacts, receipts and recipes are never normalized.

Any failed gate closes U7.11B without changing the installer, runtime,
renderer, batch core, UI, thresholds, fixtures, comparison rules or roles.

## Claim ceiling

Pass permits only:

> Private Windows repository-bound installed-runtime batch-core composition for
> deterministic film-inspired / Look Approximation output.

It does not establish calibrated stock response, physical-film reproduction,
stock distinguishability, arbitrary-media support, native dialog automation,
standalone/public packaging, cross-platform GUI behavior or product-market
validation.
