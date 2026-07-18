# U2.6A profile evidence inspection contract

Date: 2026-07-18

Node: `ULT > U2 > U2.6 > U2.6A`

Status: **frozen before implementation**

## Parent evidence and question

U2.1A already stores a strict evidence ledger in every v1 render profile:
`data_grade`, `expert_grade`, `method`, `claim_ceiling` and
`calibrated_reference_allowed`. Validation already requires held-out S3 data
and expert evidence plus a non-look interpretation before calibrated Reference
can be enabled. The remaining U2.6 gap is a supported read-only CLI/API surface
that exposes this ledger without asking callers to parse internal profile JSON.

## DoR

- U2.1A profile schema, safe-rich migration and asset verification pass;
- the tracked safe-rich profile remains heuristic/non-calibrated;
- no schema migration, renderer change, stock fitting or new evidence is needed;
- current worktree is clean and U1.5C is closed.

## Allowed implementation

1. Add one public, non-mutating profile-evidence summary function under
   `src/inference/`.
2. The function must validate the complete profile first and return only exact
   identity/evidence fields plus profile identity; it must derive no stronger
   label.
3. Add one read-only CLI script under `scripts/` that loads a profile with
   repository-root asset verification and writes deterministic JSON to stdout.
4. Add focused API/CLI/fail-closed/determinism tests.

## Frozen output keys

- `schema_id` = `kmcfm.profile-evidence-summary.v1`;
- `profile_id`;
- `profile_version`;
- `film_stock_id`;
- `latent_mode_id`;
- `interpretation`;
- `data_grade`;
- `expert_grade`;
- `method`;
- `calibrated_reference_allowed`;
- `claim_ceiling`.

No timestamps, machine paths or mutable environment state are allowed in the
summary.

## Gates

- the tracked safe-rich summary is byte-deterministic across two CLI runs;
- it reports `none / none / heuristic / false` and the existing claim ceiling;
- invalid schema, evidence escalation and asset hash mismatch fail non-zero;
- API output mutation cannot mutate the input profile;
- existing profile/recipe/render hashes and renderer bytes remain unchanged;
- focused tests, complete CPU suite, compile and diff checks pass.

## Forbidden fallback and claim ceiling

Do not change the v1 profile or recipe schema, migrate profiles, add a
calibrated profile, infer evidence from visual appearance, weaken S3 gates,
change renderer output, or expose a writable profile-management interface.

Passing U2.6A proves only that existing declared profile evidence is exposed
through a deterministic, validated read-only API/CLI. It does not improve the
evidence grade or establish stock authenticity, calibration or safety.

