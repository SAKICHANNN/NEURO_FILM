# Shared Verified Reference Composition Evidence

Date: 2026-07-28
Node: P52A-D
Decision: **shared colour/FilmFX ownership plan complete; not rendered**

## Result

P52 creates one canonical no-write plan from an exact P51 verification. It
binds P51/P50/P49/P48/P47 identities and preserves the verified shared
reference operator as the only colour owner.

Two modes exist:

- `reference-look`: execution order is only
  `verified_shared_reference_color`;
- `reference-look+film-effects`: a hash-bound Neuro-Film profile may append
  `film_effects` after the verified colour.

FilmFX remains procedural grain, halation and dust. The plan hard-codes:

- `film_color_profile_id = null`;
- `film_stock_identity_claimed = false`; and
- `calibrated_reference_claimed = false`.

Thus a reference match cannot be relabeled as a stock-specific or calibrated
film simulation. State is `composition-ready-not-rendered`; the plan performs
no file access, rendering, staging or delivery.

## Failure closure

The implementation rejects:

- applied/delivered state or changed colour owner/status/claim ceiling;
- film-colour profile injection, stock identity or calibrated claim;
- reversed or otherwise changed execution order;
- missing/extra FilmFX inputs or unbound profile hash;
- FilmFX strength outside `[0,1]`;
- verification/auth/guard/operator/reference identity mutation; and
- plan identity or unknown JSON-field mutation.

## Frozen identities

- implementation commit:
  `9f75297e5d162a2ed5416055bc5350a7f022009f`;
- implementation SHA-256:
  `2cd476eb6946e493415fc52bcff96fa159dfaf1d8ababefa565d9b28fdb63156`;
- schema SHA-256:
  `e7735bf1c88fc951a75812a16f95c0d6ee87a5b30582ab9e1e9c7a46e8add4ab`.

## Verification and propagation

- 35 P35/P51/P52 composition and verification tests pass.
- 624 combined color-match/FilmFX tests pass.
- Full suite: 1430 pass, one skip and the same 36 unrelated isolated-output
  or tracked-asset-hash failures.
- Latest main: `1dce72949ca98db73126991328969feebe911fa9`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 236/170 with zero exact overlap.
- Merge tree: `2c0438b63677008797e9fe78329a2d23d10f12df`.
- Fresh detached merge: 35 composition tests pass; temporary worktree
  removed.

SPGIN-v0 remains producer research below P45. P52 has no real shared colour
run to compose.
