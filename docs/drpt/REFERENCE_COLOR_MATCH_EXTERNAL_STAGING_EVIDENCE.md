# Reference Color Match External Staging Evidence

Date: 2026-07-28

Status: **transaction mechanics complete; real external product use remains
closed**.

## Decision

P33 adds the consumer-owned transaction endpoint after P30. It can commit a
complete authorized external-core batch to product staging, but it cannot
create authorization, invoke D-PCT, promote an algorithm, deliver an image or
mark an output applied.

The only positive test input is the existing synthetic promoted conformance
case. It proves rollback and identity mechanics, not photographic quality or
product readiness.

## Contract

`commit_external_core_staging_v1` requires:

- one valid P28 `DpctBatchResolutionV1` in `pending-product-guard`;
- its exact P30 `CoreProductStagingAuthorizationV1` in
  `authorized-for-staging`;
- one ordered P27 `AdaptedDpctCandidateV2` per source;
- exact source, reference, transform, receipt and output-view identities;
- one shared `reference_intent_id` across the source-bound transforms;
- the narrow P27 display-relative linear-sRGB profile;
- unique bounded SDR output destinations and one distinct JSON report.

The function validates the complete batch before creating a stage file. It
then encodes every isolated receipt buffer, hashes each staged image, creates
a strict canonical report and commits all images plus the report through the
existing rollback-safe batch primitive.

The only success state is `committed-to-staging`; the claim ceiling is
`staging-files-committed-not-delivered`.

## Failure-closed evidence

Nine dedicated tests prove:

- a promoted, non-research two-source control commits two 16-bit PNG files and
  one schema-valid report with exact file hashes;
- a research override remains full identity fallback and writes nothing;
- candidate permutation fails before any path is created;
- duplicate destinations fail before any path is created;
- source-bound transforms with different reference intents cannot form one
  staged batch;
- an injected report-commit failure restores both previous output files and
  the previous report, leaving no stage or backup debris;
- mutated state, source order, claim ceiling and canonical run identity fail
  validation.

No fallback, partial or `applied` state exists in the schema.

## Verification

- dedicated P33: 9 passed;
- combined P27-P33: 107 passed;
- adjacent schema/file transaction/report/composition/preprocess/encoding:
  74 passed;
- compileall: passed;
- complete CPU suite: 1245 passed, one skipped and the unchanged 36
  isolated-worktree failures;
- no P33 or reference-colour-match test failed.

The 36 full-suite failures remain the documented unrelated environment
families: ignored `outputs/` evidence is absent from this worktree and legacy
asset hashes observe Windows checkout line endings.

## Latest-main propagation

- common base: `c03c321`;
- main snapshot: `e7da085`;
- consumer implementation: `0cb94c3`;
- consumer changed paths: 160;
- main changed paths: 94;
- exact changed-path overlap: zero;
- merge tree: `4344cbf1ce2e316b6e00d0eea3317bca55579084`;
- a fresh detached synthetic merge passed all 107 P27-P33 tests;
- the exact temporary worktree was validated and removed.

Main's untracked `.codex/`, Kodak AA1 files and `tmp/` remained untouched.
D-PCT was observed clean at stable RPSCT commit `1bcb1d1`.

## Claim boundary and handoff

P33 does not change the P32 critical prerequisites. Real execution remains
closed until D-PCT publishes a fixed invocation artifact and the exact
candidate passes A1/A4/A5 without research override. Even then P33 creates
only committed staging files. A later product-delivery decision must consume
the exact P33 run identity, retain atomicity and separately authorize any
FilmFX composition or user-visible delivery.

