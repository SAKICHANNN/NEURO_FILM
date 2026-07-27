# Shared Reference Operator Batch Evidence

Date: 2026-07-28
Node: P47A-D
Decision: **consumer binding implemented; no producer compatibility claimed**

## Contract

P47 supports the product meaning of `reference-only-shared/shared-bundle`:

1. one opaque operator identity binds the exact reference, fixed producer
   capability, commit, bundle, model and options;
2. the operator identity contains no source identity;
3. each ordered source application has its own exact source, producer result,
   diagnostics and output-pixel receipt; and
4. the complete N-source batch binds every receipt to the same operator.

The consumer never serializes producer parameters or claims how the producer
builds the bundle. A future adapter must independently prove that its bundle
was formed only from the reference, fixed model and options.

## Failure closure

The implementation rejects:

- source-reference/per-source-fit relabeling;
- mixed operators or producer bundles;
- foreign reference or colour profile;
- reordered, duplicate, missing or partial source/application inventories;
- duplicate producer apply results;
- wrong output shape;
- non-finite output pixels;
- mutated receipt, batch or output-view identities; and
- unknown serialized fields.

Input output arrays are copied into dense float32 storage and made read-only.
Every output view binds exact f32 pixel bytes. The only state is
`candidate-only-awaiting-a1-a4-a5-and-numeric-product-guards`; no
`applied`, `promoted` or `delivered` state exists.

## Frozen identities

- implementation commit:
  `02ba18306173954cffbedb75915e7b776b981970`;
- implementation SHA-256:
  `d0cac3d4448ebf531569961da93d74a15a4bbd57b0c03b8f389183fefe9626d1`;
- schema SHA-256:
  `c64375f2a08fdcb54b2212c69feb49503a86e3a83f0a8daca4f90b61dcb944d1`.

## Verification and propagation

- 25 P45/P47 focused tests pass; 14 belong to P47.
- 564 combined color-match/FilmFX tests pass.
- Full suite: 1370 pass, one skip and the same 36 unrelated isolated-output
  failures.
- Latest main: `cc453e7105ae9ae86cf58dae0bcf4d1dbfbaa09d`.
- Consumer/main paths: 214/163 with zero exact overlap.
- Merge tree: `8204e0027f4d2e9a528e6467132ab5f160d97742`.
- Fresh detached merge: 38 pass and four exact-wheel tests skip because
  ignored package evidence is absent; temporary worktree removed.

## Producer boundary

D-PCT has agreed that a future shared producer path will expose two
independent identity chains:

- build a shared bundle from reference plus fixed model/options/capability;
- apply that same bundle to each exact source and return source-bound results.

At producer `b9642091223c202bdc5c5321a90e7fe6a4959a2e`, RGIN calibration is
still running and there is no report, model, bundle fixture, capability or
package. P47 therefore has no compatibility profile and cannot be used for
real invocation.
