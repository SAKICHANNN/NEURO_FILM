# U7.8D Transaction Software-Provenance Snapshot Contract

## Question

Can one multi-input three-look transaction bind one immutable software commit
before its first child render, pass that exact identity to every child recipe,
and fail closed before final publication if the live repository identity drifts?

U7.8C established the defect on six real Canon inputs: all eighteen RGB16
outputs were exact across two runs, while every recipe and child-manifest hash
drifted because the unchanged child renderer read live `git rev-parse HEAD` for
each child. U7.8D repairs only this current product transaction-provenance
defect. It does not rerun, rewrite or promote U7.8C.

## Parent and non-goals

U7.8D is a prospective correctness successor to U7.8A and U7.8B. U7.8A
remains the uninterrupted input-batch result; U7.8B remains the resumable
workspace result and continues to reject resume under a different software
commit or changed core/config/input identity.

The leaf must not change pixel arithmetic, product look parameters, the recipe
schema, output formats, stock labels, claims, source loaders, RAW/HDR/OpenEXR
paths, public APIs, cache policy or U7.8C artifacts. Velvia 50, Portra 400 and
Ektar 100 remain user-selected `film-inspired / Look Approximation` labels,
not calibrated or physical stock responses.

## Snapshot and propagation contract

The private child renderer accepts an optional explicit `software_commit`.
When omitted, its existing direct-child behavior remains exact: it resolves
the current repository HEAD once for that child invocation. When supplied,
the value is a format-bound caller-supplied commit token: it must be lowercase
forty-hex and is written unchanged into all three strict recipes. The child does
not independently resolve the object. Its only callers in this leaf obtain the
token immediately from successful `git rev-parse HEAD`; repository-object
resolvability is therefore a transaction-caller responsibility rather than an
additional per-child Git lookup. Invalid supplied values reject before input
decode, render or output-stage creation.

U7.8A resolves repository HEAD exactly once after the complete manifest,
source and configuration preflight and before creating its transaction stage.
It passes that value to every child. Before final publication it resolves HEAD
again; any change rejects the complete transaction, removes its owned stage
and preserves an absent or foreign destination. Thus a live commit cannot
produce a successfully published mixed-provenance transaction.

U7.8B reuses the `software_commit` already frozen in its static workspace
state and passes it to each child. Its existing final state recheck and
cross-commit resume rejection remain unchanged. U7.8D does not authorize a
workspace created under one commit to resume or publish under another.

## Formal test

The committed-head formal fixture uses three deterministic 32x24 RGB8 inputs.
One stable control transaction and one injected transaction run from zero in
each outer order. The injected transaction advances only the mocked live HEAD
after its first child render; all child recipe commits must still equal the
transaction-start snapshot, and U7.8A must reject before publication at its
final live-HEAD recheck. A separate fixed-head transaction must reproduce the
stable control byte for byte.

The U7.8B control proves that its child call receives the exact state snapshot
while an invocation-local live-HEAD change cannot alter staged recipes; its
existing final-state check must still reject publication. An attempted resume
under a changed software commit must continue to reject before render.

Negative controls cover malformed explicit commit values, unavailable Git
identity, late foreign destination, child failure, input/config drift and
unexpected recipe software identity. Direct-child omission retains its prior
single-live-read behavior.

## Gates

- one transaction-start software-commit resolution before the first child;
- every child recipe receives the identical frozen full commit;
- injected live-HEAD drift never publishes a U7.8A destination;
- owned transaction scratch is removed and a late foreign destination is
  preserved;
- stable transactions reproduce output, recipe, child-manifest, receipt and
  batch identities exactly across forward/reverse outer execution;
- U7.8B passes its existing state snapshot to each child but still rejects
  cross-commit resume and final-state drift;
- invalid explicit commit rejects before decode/render/stage creation;
- direct-child default semantics and existing U7.6B/U7.8A/U7.8B tests remain
  exact;
- source bytes remain immutable, network requests are zero and owned formal
  residue is zero.

## Stop rules

- Do not rewrite or rerun U7.8C to obtain a favourable identity.
- Do not strip, normalize, ignore or replace recipe provenance.
- Do not permit resume or publication across changed software, core,
  configuration or input identities.
- Do not alter recipe schemas, public interfaces, stock parameters, colour
  arithmetic, output media or claim ceilings.
- Do not expand into cache, packaging, release, RAW/HDR/OpenEXR or stock-source
  work.
- Any formal gate failure closes this exact repair without threshold, cohort,
  ordering or output-only rescue.

## Claim ceiling

Passing establishes only private Windows/Python transaction-scoped software
provenance mechanics for the existing three deterministic film-inspired Look
Approximations. It does not establish calibrated stock response, image quality,
product value, public API/package/release readiness or cross-commit recovery.
