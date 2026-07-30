# Shared Staging Restart Verification Evidence

Date: 2026-07-28
Node: P51A-D
Decision: **restart-safe shared staging verification complete**

## Result

P51 is a bounded, read-only verifier for one P50 staging transaction. The
caller must retain and supply:

- exact report file SHA-256;
- P50 run ID;
- P49 authorization ID;
- P48 numeric guard batch ID; and
- P47 operator ID.

The verifier rereads the report bytes under a 16 MiB bound, verifies UTF-8 and
the caller-held hash, reruns strict P50 parsing, checks the recorded report
path and all expected chain identities, then rehashes every ordered output.
Its canonical result binds the report/run/auth/guard/operator/reference and
every apply receipt, producer result, absolute output path and file hash.

Repeated verification is exact and does not modify report/output bytes or
timestamps. The state is `verified-shared-staging` under
`verified-shared-staging-not-delivered`; it is not composition, applied state
or delivery.

## Structure and failure closure

P34 and P51 share a package-internal bounded report/output read helper. P34's
schema and identities remain unchanged and all adjacent tests pass.

P51 rejects:

- report byte tamper, wrong expected hash, non-UTF-8 or oversized input;
- report relocation even when the caller supplies its new file hash;
- run, authorization, numeric guard or operator identity substitution;
- missing, unreadable or byte-modified output;
- reordered or duplicate receipt/result/path identities;
- state, claim, verification identity or unknown JSON-field mutation.

## Frozen identities

- implementation commit:
  `9816c4bfb31d49feb2e7756e04304d68ac8d7052`;
- shared verification SHA-256:
  `b0aa36072217397ab63e03d28e3744b79380e8b5bc145d5e14a7b97a3d091577`;
- common verification I/O SHA-256:
  `30de6ef05f2c35d4dc25a616ba44b200612eb903ead464002765b4cd636c0e2c`;
- schema SHA-256:
  `c1765f0bd57a0df99d826ccff7ec4c888ee849aa10c9e3adf3dca436ab01c869`.

## Verification and propagation

- 33 P34/P50/P51 staging and restart tests pass.
- 612 combined color-match/FilmFX tests pass.
- Full suite: 1418 pass, one skip and the same 36 unrelated isolated-output
  or tracked-asset-hash failures.
- Latest main: `1dce72949ca98db73126991328969feebe911fa9`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 232/170 with zero exact overlap.
- Merge tree: `03ffb2de34ad00b0909b8507cb31abd36612c944`.
- Fresh detached merge: 33 verification tests pass; temporary worktree
  removed.

SPGIN-v0 remains an active producer calibration run without a stable report,
model, capability, package, fixture or product rights. P51 verifies synthetic
contract evidence only.
