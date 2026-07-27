# Reference-match complete run transaction evidence

Date: 2026-07-27

Status: P17 complete for fit and stored-recipe replay file paths.

## Closed failure

Before P17, image outputs and an optional recipe were committed as one
transaction, but the CLI wrote the provenance report afterward. A report-path
collision, report construction error or final report replacement failure
could therefore return an overall failure after leaving successful new image
artifacts behind.

P17 extends the existing staging/backup/rollback primitive to the entire run:

```text
validate every input and destination
  -> stage N encoded outputs
  -> optionally stage fitted recipe
  -> bind hashes from staged bytes
  -> construct and stage provenance report
  -> commit outputs + recipe + report in order
  -> on any replacement failure, restore all prior destinations
```

No report is requested for low-level callers unless `report_path` is explicit.
Existing API behavior without a report remains unchanged.

## Invariants

1. Report paths cannot alias the reference, stored recipe, any source or any
   image output.
2. A report destination must be JSON and not a directory.
3. Output and recipe identities in the report are computed from the exact
   staged bytes that will be renamed into their final paths.
4. Report construction occurs before the first destination replacement.
5. Report replacement is the last commit step.
6. If that final step fails, existing output, recipe and report bytes are
   restored in reverse order.
7. Temporary stages and backups are removed on both success and handled
   failure.
8. Fit and recipe-replay use the same transaction primitive.

## Fault-injection evidence

- output/report alias rejects before creating an output or recipe;
- injected fit-report builder failure preserves all existing destinations;
- injected final fit-report replacement failure restores output, recipe and
  report byte-for-byte;
- injected final replay-report replacement failure restores output/report and
  does not modify the stored recipe;
- successful result report/output SHA-256 values match committed bytes;
- no stage or backup debris remains in any tested branch.

Verification:

- dedicated transaction tests: 5 passed;
- transaction plus adjacent file/report/replay tests: 32 passed;
- focused colour-match/preprocess tests: 161 passed;
- complete CPU collection: 1033 passed, one skipped and the unchanged 36
  ignored-output/CRLF-hash failures.

This strengthens product failure closure. It does not promote the current
statistical matcher, expand SDR ingress or change the film-simulation
composition boundary.
