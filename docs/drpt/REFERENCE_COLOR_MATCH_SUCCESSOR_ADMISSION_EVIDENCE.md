# Reference Match Successor Admission Evidence

Date: 2026-07-28
Node: P45A-D
Decision: **implemented; no successor candidate admitted**

## Purpose

P44 proves that a valid invocation package and exact receipts do not make its
algorithm product-worthy. P45 therefore makes candidate succession explicit
and fail closed. It does not invoke a producer or relax any gate.

## Two readiness states

`evaluation_ready` requires:

- a new full producer snapshot, package source, fixture and conformance
  identity;
- a new exact wheel and capability identity rather than an alias of the P44
  rejection;
- the already mapped relative display-linear sRGB profile and lower consumer
  compatibility profile;
- explicit `source-reference-per-source` plus `per-source-bundle`, or
  `reference-only-shared` plus `shared-bundle`, semantics;
- deterministic static execution without hidden state; and
- affirmative evaluation rights.

`product_ready` additionally requires:

- a new stable P44 evidence identity under the unchanged gate policy;
- A1, A4 and A5 pass states plus an independent blind-aesthetic pass;
- commercial-use and redistribution rights; and
- actual Windows x64, macOS arm64, iOS arm64 and Android arm64 runtime
  evidence.

The declaration is canonical JSON with a domain-separated SHA-256 identity.
Unknown fields, non-finite JSON, shortened Git identities, malformed hashes,
semantic contradictions and identity mutation fail closed.

## Frozen identities

- policy SHA-256:
  `282bf159e96148b489d2cdb1a89b924b0addf3b2aa4bc87f8e6e4d26dba8f26c`;
- declaration schema SHA-256:
  `aedc03e3ab0f7ea4aa5b9c395447d6f0c193c5d58a3a34f150ebaa7a6854a5d3`;
- implementation SHA-256:
  `de8267c693d4e1d696752f1ede7cc147bcd0906458a9423aa2ddba9dbcc276ff`;
- implementation commit: `7b0ec11dbc9433e8babe5b3727e5a41138eba43f`.

The rejected P44 capability, wheel, producer snapshot and stable evidence
identity are all pinned independently. Reusing any relevant execution identity
blocks evaluation; reusing the rejected evidence identity blocks product
readiness.

## Verification

- 19 successor/invocation/P44 focused tests pass.
- 541 combined color-match and FilmFX tests pass.
- Full isolated suite: 1347 pass, one skip and the same 36 unrelated
  output/asset/hash failures; the eleven new tests all pass.
- Latest main snapshot:
  `f309c978522874f585ab8ed0edc008505045544b`.
- Common base:
  `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 207/153 with zero exact overlap.
- Synthetic merge tree:
  `d69c3b3c03cf2b262b7bd453e81f62fbeab874e7`.
- A fresh detached merge passes 15 tests and skips four exact-wheel tests
  because ignored wheel/runtime evidence is intentionally absent; the
  temporary worktree was removed.

## Producer propagation

D-PCT `ef9a4cd4ed174ba28ff88b50d93d97af8e900603` independently closes
ROGR-v0: its best preregistered alpha improves only 78/128 rows and has a
severe worst regression, so no confirmatory set or model opens. It publishes
no successor invocation capability. This is consistent negative research,
not a P45 declaration.

The next genuinely different producer package can populate the declaration,
pass evaluation intake and then reuse P43 plumbing only after a new exact
compatibility audit. P44 is rerun without changing thresholds.
