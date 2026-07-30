# Shared Operator Product Authorization Evidence

Date: 2026-07-28
Node: P49A-D
Decision: **staging authorization boundary complete; no real shared candidate**

## Result

P49 provides the no-write product boundary for a future reference-only shared
operator. `authorized-for-staging` requires all three independent locks:

1. the exact P45 successor declaration evaluates both `evaluation_ready` and
   `product_ready`;
2. the exact P47 batch and P48 guard bind the same operator and complete
   ordered source inventory, with every source numerically eligible; and
3. a Neuro-Film `PromotionDecision` is `promoted` and is bound to the same
   declaration, stable evidence, frozen gate policy, capability, producer
   commit, model fingerprint and options hash.

The canonical successor declaration is embedded in the authorization identity.
Offline deserialization reruns P45 admission and compares the exact decision
and reasons. Declaration pass booleans alone cannot authorize without the
independent promotion binding.

Success stops at `authorized-for-staging` under
`staging-only-not-committed`. The contract contains no pixels or destination
paths and creates no applied, committed, FilmFX or delivery state.

## Failure closure

The implementation rejects or atomically falls back on:

- a non-product-ready or non-evaluation-ready P45 declaration;
- a rejected or visual-review-only promotion decision;
- producer commit, capability, producer profile or lower compatibility
  substitution;
- stable evidence, model fingerprint or options substitution;
- a foreign P48 guard, operator, batch, source, receipt or numeric decision;
- one numerically rejected source in an otherwise eligible batch;
- non-canonical embedded declaration JSON;
- incomplete, reordered or duplicate source identities; and
- authorization, admission, state, reason or unknown JSON-field mutation.

## Frozen identities

- implementation commit:
  `14fc7bfa2e15abfff7c08ab15065c0233ee2703e`;
- implementation SHA-256:
  `e95f1adc731a176aec9d06e98c7821de6b6fc2d60eccad3b4d147786b6011843`;
- schema SHA-256:
  `1bb69b96b35fe1feb9fcf3ffb0f78f5950239b3b359375bdae9251fe0d5511d8`.

## Verification and propagation

- 49 P45/P47/P48/P49 focused tests pass.
- 588 combined color-match/FilmFX tests pass.
- Full suite: 1394 pass, one skip and the same 36 unrelated isolated-output
  or tracked-asset-hash failures.
- Latest main: `1dce72949ca98db73126991328969feebe911fa9`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 222/170 with zero exact overlap.
- Merge tree: `bf2eb95969c9ed4eb21a609dd55d87a485148e82`.
- Fresh detached merge: 49 focused tests pass; temporary worktree removed.

D-PCT RGIN-v0 closed at stable snapshot `fd036aa`: 20/20 frozen uncertainty
projections failed calibration and it published no model, capability, wheel,
bundle fixture or compatibility declaration. Therefore P49 has no real
candidate to authorize and the product remains identity fallback.
