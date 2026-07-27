# Shared Operator Numeric Guard Evidence

Date: 2026-07-28
Node: P48A-D
Decision: **numeric guard implemented; product authorization remains closed**

## Result

P48 applies the unchanged P29 numeric policy to every exact P47 shared-bundle
application:

- maximum out-of-gamut fraction `0.25`;
- maximum clipping fraction `0.05`;
- maximum projected fraction `0.25`;
- maximum new-boundary fraction `0.05`; and
- boundary epsilon `1/65535`.

Each source binds producer diagnostics identity, finite state, output extrema,
OOG/clipping/projection fractions and its exact receipt. The consumer verifies
the live output extrema and independently measures new-boundary fraction from
the bound source and output pixels.

One policy covers the complete ordered batch. Any failed source forces
`identity-fallback` for the batch. A passing batch stops at
`eligible-for-transaction` with claim ceiling `numeric-only-no-visual-claim`;
it is not authorized, staged, applied or delivered.

## Failure closure

The implementation rejects:

- foreign diagnostics or receipt identities;
- stale output extrema;
- non-finite or out-of-range fractions;
- missing, reordered or incomplete facts;
- policy substitution;
- source/receipt/output/batch identity mutation; and
- unknown JSON fields.

Exact output new-boundary measurement catches large finite excursions even if
producer fractions are small. Producer pre-policy OOG/clipping/projection
facts remain producer-owned and must be mapped through a future exact
conformance fixture.

## Frozen identities

- implementation commit:
  `5c50874c6d2d4b0600f9820f21a5da3d42f28b2a`;
- implementation SHA-256:
  `f851778b7845ced63b248775c84d67b6134f602326c22362a51e866c2986ac67`;
- schema SHA-256:
  `0f290a6e48e9162a2108e6af612f38ca306342b50dcb44b9ac059b3eb1e850cc`.

## Verification and propagation

- 38 P45/P47/P48 focused tests pass; 13 belong to P48.
- 577 combined color-match/FilmFX tests pass.
- Full suite: 1383 pass, one skip and the same 36 unrelated isolated-output
  failures.
- Latest main: `2bc968dd4d4cfaf07a8ce694a02997e7b50e582b`.
- Consumer/main paths: 218/167 with zero exact overlap.
- Merge tree: `4eb40c39b5d307572b5973caedad8524709165fe`.
- Fresh detached merge: 51 pass and four exact-wheel tests skip because
  ignored package evidence is absent; temporary worktree removed.

D-PCT still has no RGIN calibration report/model/capability/package, so no
producer facts are mapped and no compatibility claim opens.
