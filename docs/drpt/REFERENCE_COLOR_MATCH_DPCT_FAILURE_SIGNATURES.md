# Invoked D-PCT Failure Signatures

Date: 2026-07-28
Node: P46A-D
Decision: **diagnostic complete; P44 rejection unchanged**

## Frozen input and replay

The analyzer consumes only the two existing P44 progress files. It does not
invoke producer code, read targets, render pixels or change any threshold.

Both inputs produce byte-identical 8,710-byte reports:

- report SHA-256:
  `3512d4f4d9ebef0ccef2687d0741eab28c711dc76e025221123461cbd528c9fc`;
- stable evidence ID:
  `b341016804db32b16641448aad5ebf8cfeef54314e1c9cf967f8e3efafa88fe3`;
- analyzer SHA-256:
  `3aa35d29e7a0e84a14a30ca1b85be5086476081102734f2a56f9ca6bb06afc39`;
- implementation commit: `d55142e`.

Timing-dependent diagnostics, apply-result, response and consumer-receipt IDs
are validated but deliberately excluded from the stable diagnostic identity.
Request-independent producer bundle and consumer transform identities remain
bound.

## Failure mechanisms

### 1. Universal global overcorrection

- Candidate error exceeds the unchanged source error in 30/30 rows.
- Median candidate/source target-error ratio is `3.012941`.
- Minimum ratio is still `1.117144`; maximum is `6.693720`.
- Every one of the 30 rows produces a unique bundle and consumer transform.

This is not a narrow tail failure. The fitted mapping moves every evaluated
source farther from the known target.

### 2. Clipping is not the primary cause

- Median producer clipping fraction is only `0.894447%`.
- 28/30 regressions occur with clipping at or below `5%`.
- Pearson correlation between clipping and error magnification is `0.143433`.

Two rows do exceed 5% clipping, so the report does not claim clipping is
irrelevant. It establishes that fixing only gamut clipping cannot explain or
repair the dominant failure.

### 3. Both source and reference strata matter

- Source `09` has the worst median error ratio: `5.420546`.
- Reference `09` has the worst median error ratio: `5.375421`.
- Every source and every reference stratum has five unique fitted bundles and
  a negative median improvement.

The current joint fit therefore has no stable “good reference” or “good
source” subgroup that could justify a product allow-list.

### 4. Source-conditioned fitting creates an A5 hazard

All six context probes produce different bundles for the two source contexts,
and all six fail. Median producer clipping across their twelve calls is only
`0.401476%`, while worst shared-colour median/p95/maximum drift is
`72.6391/96.9062/119.6310` Delta E76.

This directly supports a genuinely reference-only shared operator as the next
discriminating semantic family. It does not prove that family will pass:
ROGR-v0 already failed A1 development, and RGIN-v0 remains sealed research.

### 5. Photographic corruption is multidimensional

- neutral-axis failure: 6/6;
- new-boundary failure: 4/6;
- semantic-hue failure: 1/6;
- worst neutral chroma: `50.1215`;
- worst boundary fraction: `28.6965%`;
- worst semantic hue rotation: `89.0774` degrees.

A successor needs bounded identity shrinkage and explicit neutral/boundary
controls in addition to shared-operator semantics. These are algorithm design
constraints, not a request to tune the consumer gate.

## Producer-facing consequence

The strongest next hypothesis is not “the same source-reference fit with a
different blend.” It is a new reference-only/shared-bundle family with:

1. conservative identity shrinkage under uncertainty;
2. explicit neutral-axis and boundary preservation;
3. bounded operator magnitude;
4. no hidden source-context adaptation; and
5. unchanged P44 evaluation after independent producer confirmation.

D-PCT RGIN-v0 states the correct future semantics
`reference-only-shared/shared-bundle` and uses disagreement-based identity
shrinkage, but at producer `b964209` it has no calibration pass, model,
capability, package or P45 declaration. No consumer admission opens.

## Verification and propagation

- 12 focused analyzer/P44 tests pass.
- 550 combined color-match/FilmFX tests pass.
- Full suite: 1356 pass, one skip and the same 36 unrelated isolated-output
  failures.
- Latest main: `c20c14ef7c77026f3e4c1fe38557654d06609b67`.
- Consumer/main paths: 210/157 with zero exact overlap.
- Merge tree: `ff0152516edb0349c0471e7a170c0c500f2aa00a`.
- Fresh detached merge: 24 pass and four exact-wheel tests skip because
  ignored package evidence is absent; temporary worktree removed.
