# Reference Color Match External Composition Evidence

Date: 2026-07-28

Status: **composition contract complete; rendering and delivery remain
closed**.

## Decision

P35 defines how a future genuinely promoted external reference look can
coexist with Neuro-Film FilmFX. It deliberately does not reuse P18's local
research-baseline semantics.

The external reference transform owns colour. A plan may append only the
existing validated procedural effect fields: grain, halation and dust. It
cannot append a film colour profile, claim a film stock identity, render
effects, mutate files, deliver outputs or mark anything applied.

## Contract

`ExternalReferenceCompositionV1` binds:

- exact P34 staging verification ID;
- exact P33 run ID;
- exact P30 authorization ID;
- shared reference-intent ID;
- source count;
- optional validated `FilmEffectBinding`.

Fixed semantics:

- state: `composition-ready-not-rendered`;
- colour owner: `external-reference-look`;
- reference colour status: `verified-staging`;
- claim ceiling: `reference-look`;
- order without effects: `verified_reference_color`;
- order with effects: `verified_reference_color -> film_effects`;
- `film_color_profile_id=null`;
- `film_stock_identity_claimed=false`.

P35 extracts the existing FilmEffectBinding builder/validator into public
package functions. The local P7/P12/P18 composition behavior is unchanged and
its tests continue to pass.

## Adversarial evidence

Eleven dedicated P35 tests cover:

- exact P34 identity binding without FilmFX;
- validated profile binding with FilmFX after reference colour;
- strict JSON roundtrip and schema validation for both branches;
- missing or unsolicited profile inputs;
- applied state injection;
- film-colour stacking;
- film-stock identity escalation;
- reversed execution order;
- changed staging-verification identity;
- out-of-range effect strength;
- unknown serialized fields.

All invalid cases fail without rendering or writing.

## Verification

- dedicated/existing composition and P34 adjacency: 40 passed;
- combined P27-P35: 147 passed;
- compileall and diff check: passed;
- complete CPU suite: 1265 passed, one skipped and the unchanged 36
  isolated-worktree failures;
- no P35, P18 or reference-colour-match test failed.

## Latest-main propagation

- common base: `c03c321`;
- main snapshot: `cfd1271`;
- consumer implementation: `249e415`;
- consumer changed paths: 168;
- main changed paths: 100;
- exact path overlap: zero;
- merge tree: `ce79f519ac95e83184e953f71dd9c305eed21959`;
- a fresh detached synthetic merge passed all 147 P27-P35 tests and was
  removed.

Main's concurrent Kodak AA1 results/governance files, `.codex/` and `tmp/`
were not touched. D-PCT remained stable at `c688c32`; P35 adds no producer
requirement.

## Handoff

P35 completes external reference/FilmFX ownership planning, not FilmFX
execution. A future renderer must first rerun P34 verification, bind this
exact plan ID, render effects through Neuro-Film's existing product renderer
and create a new atomic report. Real use remains closed until a D-PCT
invocation artifact and A1/A4/A5-promoted candidate exist.
