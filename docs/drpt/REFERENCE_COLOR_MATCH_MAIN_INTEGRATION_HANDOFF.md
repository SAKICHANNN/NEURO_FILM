# Reference Color Match Main Integration Handoff

Date: 2026-07-28

Status: **consumer implementation ready for main-owner review; real
external-algorithm admission remains closed**.

## Frozen snapshots

- consumer payload branch: `codex/reference-color-match`;
- complete P1-P60 implementation head:
  `eee47a2eac544b7583309b3a96fd93ff1ae6c6e9`;
- P58 non-self-referential reviewed payload:
  `1aee24f1d0a76da91079f6b88c58036dbcf6c57e`;
- common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`;
- P58 manifest main snapshot:
  `1f61119087cdb72d939b8db0c7b915e4adb7c5ce`;
- latest read-only main preflight:
  `4b762be1ecd4f9591564392c78ce39c9a3c3932a`;
- D-PCT read-only snapshot:
  `ffdfd98`;
- conflict-free main/payload merge tree:
  `88198dd70551e7b4872c2b84d327d0376cf330ee`.

The payload and main snapshots have zero exact changed-path overlap from the
common base. The main worktree's `.codex/` and `tmp/` remain owner-controlled
and were never modified by this branch.

The deterministic P58 review manifest is
`configs/reference_match_main_integration_manifest_v1.json`, SHA-256
`db06eb9b47becb0a2a0f00e3a8a9f8d045e5e88e59cc285cc6f8b638a49a6a5b`.
It binds 253 payload Git blobs, 178 main changed paths, zero overlap, 11
required public exports and nine shared schemas. It deliberately binds the
preceding payload commit so it never hashes itself.

The independent strict P59 schema is
`configs/schemas/reference_match_main_integration_manifest_v1.schema.json`,
SHA-256
`dcaa3caca5cee9082eb8ece9b5206ce2a337528f8ffb894019e76c2bb0f08a06`.
It rejects malformed review data before resolving Git commits; the P58 exact
rebuild remains authoritative for shape-valid blob or hash tampering.

## What the payload provides

The consumer module implements:

- one uploaded reference plus ordered N-source semantics;
- immutable recipes, reports and replay identities;
- fail-closed local statistical baseline and research-only controls;
- A1/A4/A5 evaluation and promotion gates;
- strict D-PCT v2 relative-SDR fixture compatibility;
- exact producer receipt, ordered batch, numeric and product-authorization
  guards;
- exact-wheel local invocation, frozen A1/A4/A5 execution and a strict
  two-stage successor-candidate intake after the current capability failed;
- deterministic failure-signature analysis showing universal overcorrection,
  weak clipping association and source-context-dependent bundle drift;
- one-reference shared-operator semantics with ordered N source-bound exact
  output receipts for a future reference-only producer;
- per-source shared-apply numeric facts and all-or-nothing batch safety under
  the existing P29 thresholds;
- promotion-bound shared-path staging authorization that embeds and reruns
  the exact P45 declaration and cannot be opened by declaration booleans
  alone;
- rollback-safe shared-path output/report staging that binds P47/P48/P49 and
  preserves the existing per-source/FilmFX/local-delivery wire identities;
- read-only restart verification of the exact P50 report, P49/P48/P47 chain
  IDs and every committed output file;
- a no-write P51-to-FilmFX ownership plan that forbids film colour, stock
  identity and calibrated-reference claim escalation;
- rollback-safe shared-path procedural FilmFX staging that reruns P51,
  protects every P50 artifact and atomically commits ordered outputs/report;
- read-only shared FilmFX restart verification that caller-binds P53 and
  rehashes every P50 input plus every P53 output;
- no-write shared local-delivery authorization that live-reruns P54 and
  requires exact product-ready P49 plus all ordered source approvals;
- rollback-safe shared local export that reconstructs P55, preserves exact
  P47 receipt/result lineage and atomically commits files plus report;
- read-only shared local-export restart verification that caller-binds P56
  and rehashes every staging source and delivered file;
- exact successor runtime-evidence binding that distinguishes host/device
  execution from cross-compile, link-only and object-only evidence and binds
  versioned environment matrices plus runner/executable/report identities;
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

## Frozen P53 shared FilmFX identities

| Artifact | SHA-256 |
|---|---|
| shared FilmFX transaction implementation | `7f02d5dab1fd485dcc6fe7e1969babd3112ea6cf820bf02dd61001f0030ee969` |
| common FilmFX staging helper | `2f0e5b8e134a456deaa89aec005df28efcf5fa94cb2a01c5f28d0d75aff045f4` |
| shared FilmFX run schema | `0d48d27b8cc120ea6703a5502a3e8ebf9f05bcddcc272d8dec1274c25a36720f` |
| shared FilmFX verification implementation | `c3e036040235a0f56fdb4ae44ca34faee0b90a504e16f58346edf8de2edd7c39` |
| shared FilmFX verification schema | `70aafd69a66eda74dab76361cd1a3f95022c7a835ef743ad4463e53ea79fad0f` |
| shared delivery authorization implementation | `fed1d5fb4af0839602b39c3c924a0e872ae21f9baa34c4da442dcea40006a66d` |
| shared delivery authorization schema | `bfbf365666d98bf642a5af6b4c1c517bfc5885f648e72f483063be75eb0c3a73` |
| shared local delivery implementation | `1a4b4337ecd01c74a83e3e1d3e13d5963a4911febf01c24aeb923d4895979c0e` |
| shared local delivery schema | `adccb9b822b51e8048f3e89e069ef84b1fbd2d89837abc1c8893112fd6541efc` |
| shared local delivery verification implementation | `8e37189b519a83c27f7940c85842920aa384983cfc877f6d195cb54d0275333e` |
| shared local delivery verification schema | `0325a907cfc9fb8dc3562382283e55cf10b52b10a22d1b1ae3c77170ec091ac4` |

## Integration procedure for the main owner

1. Refresh main instructions and preserve its uncommitted/untracked work.
2. Verify the committed P59 schema and P58 manifest, then review
   `c03c321..eee47a2`; do not copy files manually and do not import mutable
   paths from the D-PCT repository.
3. Recompute `git merge-tree --write-tree eee47a2 <reviewed-main>`.
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

- latest complete `test_color_match*` suite: 621 passed;
- latest isolated consumer full suite: 1516 passed, one skipped, 36 unchanged
  environment/output/hash failures;
- latest detached synthetic main merge: 42 P60/P59/P57 integrity tests
  passed; the temporary worktree was removed;
- consumer worktree is clean after every stable leaf.

## External blockers that remain real

1. D-PCT must publish a genuinely different fixed invocation package after
   the current exact capability failed P44.
2. The successor must pass P45 intake, A1 reference identifiability, A4
   photographic preference/severe-tail review and A5 batch consistency
   without research override.
3. P60 now makes the distinction explicit, but Android device/JNI and Apple
   host/device runtime/invocation evidence remain open; cross-compilation is
   not runtime proof.
4. The main owner must review and merge the payload.

D-PCT RGIN-v0 closed at `fd036aa`: all 20 frozen uncertainty projections
failed its calibration and it emitted no model, capability, wheel or bundle
fixture. BMKL, ROGR and other development results likewise remain non-callable
research evidence. None may be substituted into the consumer by algorithm
name.

SPGIN-v0 closed negative at producer `a2e5ed9`: none of 12 frozen safety
configurations passed, and it emitted no model, capability, package or shared
fixture. It remains below P45 and cannot enter P49/P50.

CGIN-v0 also closed negative at producer `ffdfd98`: none of nine grouped
contrastive safety configurations passed and it emitted no downstream
artifact. Producer research now treats the four-generation same-60-image
family as saturated; a successor needs materially new evidence rather than
another architecture/loss sweep on the same development set.
