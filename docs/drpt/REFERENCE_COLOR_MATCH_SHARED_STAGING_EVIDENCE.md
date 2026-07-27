# Shared Operator Atomic Staging Evidence

Date: 2026-07-28
Node: P50A-D
Decision: **rollback-safe shared staging complete; delivery remains closed**

## Result

P50 consumes one exact P47 batch, its P48 numeric guard, the P49 staging
authorization and every live prepared shared apply. It revalidates and
cross-binds all identities before accepting destinations, then:

1. preflights unique SDR output paths, report path, bit depth and extensions;
2. encodes every exact display-linear sRGB output to staged files;
3. hashes those encoded files and builds one canonical report binding P47,
   P48, P49 and every producer result/diagnostics/output identity; and
4. commits all outputs and the report through the existing backup/rollback
   batch primitive.

The resulting state is `committed-to-shared-staging` under
`shared-staging-files-committed-not-delivered`. It is not local delivery,
app-level applied state or FilmFX composition.

## Structure and compatibility

P33's generic SDR path validation and encoding were moved without semantic
change to `src/color_match/staging_io.py`. P33, procedural FilmFX staging and
local delivery now reuse that helper. Their schemas and canonical identities
are unchanged.

The shared path adds an encoding-time bounded-sRGB check. A sparse out-of-range
pixel that remains below P48's batch-fraction thresholds still cannot be
silently clipped into a committed product file.

## Failure closure

Tests prove no new files, no debris, or exact restoration of every prior byte
for:

- P45, promotion or P48 fallback;
- foreign/reordered/incomplete shared applies or numeric guard;
- duplicate output paths and report/output collisions;
- unsupported extension or bit depth;
- bounded-sRGB encoding failure;
- injected final report commit failure; and
- report state/order/claim/identity or unknown-field mutation.

## Frozen identities

- implementation commit:
  `97f4131aa01c0d0708e06aa9e60b20e72cabc883`;
- shared transaction SHA-256:
  `a9301ac7e3e987be27f6b37352d9dde44c7226882bb87c163689ce561f738b19`;
- common staging I/O SHA-256:
  `4423561a318f70b06e9a9eb3b7d3d8407db6c4f6fc0b763a5da3d150d4a20dc5`;
- shared staging schema SHA-256:
  `f858ef52418062fceb7a9ee28b8648ceef981131caa5936f61f3e222f4255262`.

## Verification and propagation

- 54 P33/P36/P39/P49/P50 transaction tests pass.
- 600 combined color-match/FilmFX tests pass.
- Full suite: 1406 pass, one skip and the same 36 unrelated isolated-output
  or tracked-asset-hash failures.
- Latest main: `1dce72949ca98db73126991328969feebe911fa9`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 227/170 with zero exact overlap.
- Merge tree: `a04585aecf188ad1c3dd608ba4811492a865824e`.
- Fresh detached merge: 54 transaction tests pass; temporary worktree
  removed.

D-PCT SPGIN-v0 is only preregistered research at producer commits
`f345b22`/`6256a89`/`e8bbe0c`; it has no calibration, model, capability,
package, fixture or product rights. P50 therefore has no real producer output
to stage.
