# Reference Color Match Main Integration Handoff

Date: 2026-07-28

Status: **consumer implementation ready for main-owner review; real
external-algorithm admission remains closed**.

## Frozen snapshots

- consumer payload branch: `codex/reference-color-match`;
- complete P1-P40 payload head: `6db15c4fb439ac16628eac5e128a582ed2495882`;
- common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`;
- main read-only snapshot:
  `9fea35b1c8e3141ee4f7c9dd628edd7ca8836203`;
- D-PCT read-only snapshot:
  `fa592e2e3c5d4ac7ee16d9a7bb6b906a712a1d2d`;
- conflict-free main/payload merge tree:
  `831c34ac17788600f4750c02ed241b58271de71d`.

The payload and main snapshots have zero exact changed-path overlap from the
common base. The main worktree's `.codex/` and `tmp/` remain owner-controlled
and were never modified by this branch.

## What the payload provides

The consumer module implements:

- one uploaded reference plus ordered N-source semantics;
- immutable recipes, reports and replay identities;
- fail-closed local statistical baseline and research-only controls;
- A1/A4/A5 evaluation and promotion gates;
- strict D-PCT v2 relative-SDR fixture compatibility;
- exact producer receipt, ordered batch, numeric and product-authorization
  guards;
- portable consumer identity conformance across Python, MSVC, LLVM-MinGW and
  Android cross-link evidence;
- P33-P40 staging, restart verification, external-reference/FilmFX
  composition, atomic procedural FilmFX staging, no-write local authorization,
  atomic local export and restart verification.

P39/P40 local delivery states describe only a verified local file transaction.
They do not mean app-level `applied`, film-stock identity, calibrated
reference, public sharing or algorithm promotion.

## Frozen P33-P40 wire hashes

| Contract | SHA-256 |
|---|---|
| external core staging run | `dd2c7af1c47893dd3f0e904b4a115580fcc4fc966fca4d399950cc8b6e63bc04` |
| core staging verification | `28b0575db48f43fc54788ba5f3bfeefb49187aa591d2cb43cfbc0c4b836831d5` |
| external composition | `2896b04d239c2e381158c05c9a55466160f23abd86d99d3fd5cdcd7346101eff` |
| external FilmFX run | `5d6d27b6c993e37e8e285fd4113b3a1433385e999f6fc72bdab8ff8bd2b0f96e` |
| FilmFX staging verification | `4e6f883ea3243fc88fa121d065bc08ea08459f49e7625e99d44f1ce1ad1facfa` |
| local delivery authorization | `cd1b97aa1a48e0be37f66aa455aee31337efdd72004cf612ae41f65c803b067b` |
| local delivery | `b8b41a3023e63f1c1c59d7962567fae881cde801c9bb69e2d8f9609944143f0c` |
| local delivery verification | `abe4270def5a0c60a1ff70674d188fbfa411871b116b59120eac2adbab23bf43` |

## Integration procedure for the main owner

1. Refresh main instructions and preserve its uncommitted/untracked work.
2. Review `c03c321..6db15c4`; do not copy files manually and do not import
   mutable paths from the D-PCT repository.
3. Recompute `git merge-tree --write-tree <reviewed-main> 6db15c4`.
4. Perform a normal reviewed merge of the payload branch in the main task.
5. Run all `tests/test_color_match*.py` plus halation, tiled dust and tiled
   grain tests.
6. Run the full main suite where its ignored evidence outputs and tracked
   asset-byte policy are available.
7. Keep the product default at identity fallback until a real producer
   invocation and A1/A4/A5 promotion exist.

The consumer task does not perform this merge because the main task owns its
dirty worktree, Ultimate tracker and product integration decisions.

## Current evidence

- latest consumer combined color-match/FilmFX suite: 479 passed;
- latest isolated consumer full suite: 1319 passed, one skipped, 36 unchanged
  environment/output/hash failures;
- latest detached synthetic main merge: 149 focused tests passed and the
  temporary worktree was removed;
- consumer worktree is clean after every stable leaf.

## External blockers that remain real

1. D-PCT must publish a fixed invocation package/ABI that produces a genuine
   source-bound receipt under an explicitly compatible profile.
2. One candidate must pass A1 reference identifiability, A4 photographic
   preference/severe-tail review and A5 batch consistency without research
   override.
3. Android device/JNI and Apple compiler/runtime/invocation evidence remain
   open; cross-compilation is not runtime proof.
4. The main owner must review and merge the payload.

D-PCT `fa592e2` strengthens BMKL's development evidence on Volga2K but also
shows that blend strength is task-profile-dependent. The dataset is
same-scene/two-camera matched features, its legal review remains open, and the
result is explicitly not a full-image or product promotion. It changes no
producer schema, ABI or receipt, so the consumer must not admit it.
