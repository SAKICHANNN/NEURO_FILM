# Reference Color Match Main Integration Handoff

Date: 2026-07-28

Status: **P1-P104 consumer payload is pinned by the immutable P105 v16 review
manifest; metadata-attested local sRGB MatchViews are complete, while real
external-algorithm admission and delivery remain closed**.

## Frozen snapshots

- consumer payload branch: `codex/reference-color-match`;
- complete reviewed P1-P104 payload head:
  `e73a42ce8c2d0c45c0b2b7e2d89224d82efa05de`;
- P58 non-self-referential reviewed payload:
  `1aee24f1d0a76da91079f6b88c58036dbcf6c57e`;
- common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`;
- P58 manifest main snapshot:
  `1f61119087cdb72d939b8db0c7b915e4adb7c5ce`;
- latest read-only main preflight:
  `4eb571c785f9a5d5109767c11a3b1bf16eac8430`;
- D-PCT read-only snapshot:
  `46b77bb`;
- conflict-free main/evidence-head merge tree:
  `7b0da2298fd26424cc139cecc74dbf64cb5a9b6a`.

The payload and main snapshots have zero exact changed-path overlap from the
common base. The main worktree's `.codex/` and `tmp/` remain owner-controlled
and were never modified by this branch.

The deterministic P58 review manifest is
`configs/reference_match_main_integration_manifest_v1.json`, SHA-256
`db06eb9b47becb0a2a0f00e3a8a9f8d045e5e88e59cc285cc6f8b638a49a6a5b`.
It binds 253 payload Git blobs, 178 main changed paths, zero overlap, 11
required public exports and nine shared schemas. It deliberately binds the
preceding payload commit so it never hashes itself.

P58 remains immutable and intentionally does not cover P60-P63. P64 adds
`configs/reference_match_main_integration_manifest_v2.json`, SHA-256
`ae67b3ef7ec6fcf169aa2a4a97485693c341caf5624a2a8de6c5e866a7d784dd`.
It binds 273 payload Git blobs, 213 latest-main changed paths, zero overlap,
20 required public exports, 14 exact schemas and the immutable P58 blob/hash.
The v2 manifest deliberately binds the preceding P63 payload commit and never
hashes itself.

P66 adds `configs/reference_match_main_integration_manifest_v3.json`,
SHA-256
`fc9373a0452df7f1bff309b1bf9a2be83fc50a1cfa025e20236942b1bd2cf07c`.
It binds 280 payload blobs, 218 latest-main changed paths, zero overlap, 25
public exports, 15 exact schemas and the immutable P64 v2 manifest hash
`ae67b3ef...d784dd`. V3 binds P65 payload `e83c18a` and does not hash itself.

P68 v4 binds P67 payload `a2bb952` and preserves P66 v3. P70 v5 then binds
P69 payload `272db64` against main `50b38dd`: 294 payload blobs, 223 main
paths, zero overlap, 35 exports and 17 schemas. V5 SHA-256 is
`782043cc29db7a2489188c980e658fb5cc9f214808ba85b87d146bc07f0c97eb`;
it binds P68 v4 SHA-256 `612c0977...daae14` and never hashes itself.

P73 v6 binds the P71/P72 implementation at `3126657` against main
`2f9a7c0`, adding exact colour-attestation and attested-MatchView schemas.
P74 then closes the discovered `core.autocrlf=true` checkout drift for the
entire historical manifest chain. P75 v7 binds that checkout-safe payload
`a305497`: 309 payload blobs, 236 main paths, zero overlap, 45 exports and 19
schemas. V7 SHA-256 is
`18a7a31dde03d448357d31bd94d6d204f5cc42c7f61e23690227dbdc5c72afc7`;
it binds v6 SHA-256 `01aa9137...412d024` and never hashes itself.

P76 `31f03fd` supersedes only the P71/P72 implementation-byte identities by
making both consumers use one pinned 588-byte ICC asset rather than calling
the encoder's runtime generator. P77 v8 binds that payload against main
`93a7b66`: 318 payload blobs, 238 main paths, zero overlap, 47 exports and 20
schemas. V8 SHA-256 is
`f40e32b922defecd6294dbb9de5bb790d336b44e6b1532f3cbbbdc1ee6a54ee9`;
it binds v7 SHA-256 `18a7a31d...afc7` and never hashes itself.

P78 `983810a` plus evidence `d661fa9` add the freestanding native ICC
accessor and factual Windows/Android/Apple portability ceilings. P79 v9 binds
that complete P1-P78 payload against main `60b9bfa`: 329 payload blobs, 243
main paths, zero overlap, 47 exports and 20 schemas. V9 SHA-256 is
`041e28a2d7cb71b4655eac240abdbc4fac1a8df371a708b3af21cc3b21d04f8f`;
it binds v8 SHA-256 `f40e32b9...4ee9` and never hashes itself. A later
read-only preflight against main `1ffbb5e` remains zero-overlap and passes the
detached merge suite.

P80 `b550d8e` adds reproducible MSVC/LLVM-MinGW DLLs and independent dynamic
ABI invocation; evidence `c87d366` records exact DLL identities. P81 v10
binds that P1-P80 payload against main `21a877f`: 333 payload blobs, 247 main
paths, zero overlap, 47 exports and 20 schemas. V10 SHA-256 is
`6663dbcabceaeed6d88ec54912377bee8d2d61f1962eba1be92c51edcada7ea1`;
it binds v9 SHA-256 `041e28a2...04f8f` and never hashes itself.

P82 `8bb9e0b` plus evidence `20fb34a` add the exact portable decoded-sample
sRGB EOTF ABI: both Windows compilers execute every uint8/uint16 code exactly,
while Android remains link-only and Apple remains object-only. P83 v11 binds
that P1-P82 payload against main `a264a82`: 343 payload blobs, 253 main paths,
zero overlap, 47 exports and 20 schemas. V11 SHA-256 is
`b3bb7dee19a38c41620a00ae6e525ff20492adb3db705db9b8d26e85b7deccfe`;
it binds v10 SHA-256 `6663dbca...7ea1` and never hashes itself. Main later
advanced to `4bbf362`; the refreshed 347-versus-254 path comparison remains
zero-overlap and the detached merge tree is `f1df3ff9...b065`.

P84 `091e688` hardens the same two-symbol EOTF ABI with explicit binary32,
little-endian and typed-pointer alignment preconditions while preserving the
table identity and exhaustive values. P85 v12 binds that P1-P84 payload
against main `bc04943`: 347 payload blobs, 258 main paths, zero overlap, 47
exports and 20 schemas. V12 SHA-256 is
`2e082062d65c909cc6a6ff3e2ecb903912b03834328c760a29c799bf88bd1a66`;
it binds v11 SHA-256 `b3bb7dee...ccfe` and never hashes itself.

P87 `2d237e7` freezes the current staging OETF and 8/16-bit quantization as
exact first-float32 threshold tables; P88 `61842db` proves 24MP bounded
streaming. P89 v13 binds that P1-P88 payload against main `9301cba`: 361
payload blobs, 261 main paths, zero overlap, 47 exports and 20 schemas. V13
SHA-256 is
`64ef7be01b7a281e785294cda1820c8cca09ff6388fa3cc6ae451d1c26fb147e`;
it binds v12 SHA-256 `2e082062...1a66` and never hashes itself.

P90-P95 add a consumer-owned dual-ABI Android instrumentation package and
prove the exact quantizer twice on a cold Android 14 x86_64 emulator after
closing target-DEX, ABI-soname and instrumentation-lifecycle defects. The
stable runtime identity is `sha256:079d19c2...ff225`; this remains virtual
device evidence and does not substitute for arm64 physical-device runtime.
P96 v14 binds P1-P95 `2c33809` against main `8dcfdac`: 376 payload paths,
310 main paths and zero overlap. V14 SHA-256 is
`12a951f758c4ac8d29a82aee76ff2fd1b6ac6a88bfdb385019b7a0ac90b0c734`;
it binds v13 and never hashes itself.

P97-P99 extend that Android virtual runtime to the exact 588-byte ICC
accessor and every 8/16-bit EOTF code, with complete EOTF-to-OETF roundtrip.
P100 v15 binds P1-P99 plus the v13-v15 LF identity correction at `10b1e4c`
against main `5d355bd`: 380 payload paths, 320 main paths and zero overlap.
V15 SHA-256 is
`949ed4c63ce9afae0a395084b053ee25c0a6312aba99ffe4f781310fd5a4c440`;
it binds v14 and never hashes itself.

P101-P104 execute the unchanged P42 canonical core inside the dual-ABI Android
instrumentation package. Two cold Android 14 x86_64 emulator boots reproduce
all ten P28-P30 canonical identities, SHA failure atomicity and the complete
staging truth table under stable identity `sha256:a3fa50e0...e03f20`.
P105 v16 binds payload `e73a42c` against main `4eb571c`: 384 payload paths,
333 main paths and zero overlap. V16 SHA-256 is
`0174ab22fd913d340b1c86c6ac7133fe4204a1ccd1c0285f288ee27036bb67ae`;
it binds v15 and never hashes itself.

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
- a no-write runtime qualification over exact P49/P60 so unbound runtime
  booleans cannot enter any future versioned durable staging path;
- create-only runtime-qualified P62 staging that snapshots caller pixels,
  requires an exact consumer-pinned P61, publishes each output with
  no-replace semantics and publishes its canonical report last as the sole
  commit marker;
- read-only P63 restart observation that caller-pins P62, opens report and
  outputs once, binds handle identities and hashes the same handles under
  strict count/size budgets; it explicitly does not authorize a later path
  reopen or delivery;
- process-local P65 byte capture that reads from those same live handles,
  returns no snapshot unless every final rehash succeeds, and persists only a
  path-free no-authority receipt;
- strict P67 in-memory PNG/JPEG/TIFF sample decoding with whole-batch geometry
  preflight and no partial decoded return;
- P69 deterministic decoded-sample to display-relative linear-sRGB MatchView
  bridge whose validator re-executes EOTF and provenance reconstruction,
  retained as assumption-bound v1;
- strict P71 encoded-byte metadata attestation for the exact sRGB ICC with
  bounded/no-conflict PNG, JPEG and TIFF parsing;
- P72 metadata-attested MatchView v2 that reruns P71 plus the frozen EOTF and
  binds every metadata identity into provenance;
- portable consumer identity conformance across Python, MSVC, LLVM-MinGW and
  Android cross-link evidence;
- P33-P40 staging, restart verification, external-reference/FilmFX
  composition, atomic procedural FilmFX staging, no-write local authorization,
  atomic local export and restart verification.

P39/P40 local delivery states describe only a verified local file transaction.
They do not mean app-level `applied`, film-stock identity, calibrated
reference, public sharing or algorithm promotion.

## Frozen P71-P72 identities

| Artifact | SHA-256 |
|---|---|
| colour-attestation implementation after P76 | `df880fd5fecb34275789df50d24b0d6db6669f21dc9c0fb5c649472f1fc7b6f3` |
| colour-attestation schema | `4db512eb2ced28d9cdf0b176393d917cff35c65396f07b45a7e06632b793a93a` |
| attested MatchView v2 implementation after P76 | `99fa3bdc84e9470c2fc665a6e3c9e168bcd787ca50139bdfb14338aa07ec6e10` |
| attested MatchView v2 schema | `ce4d6919e6b9e729a24fe2b1f37f7b1096391846d7879a80098d5aca4c9269f5` |
| pinned ICC module | `0fa5176a6b04f6fa036fc23fff190f4e212cd3790c9c0315b3fd292c82e0b414` |
| pinned ICC conformance schema | `d3f70d75b6ca5e2a923cdb49260c30bc57652588d102f2609364a2c80597232e` |
| pinned ICC conformance fixture | `82944c5d21829c62790e023c46a947f016e890db495e71f0a9d2f28a579aa9dd` |
| exact sRGB ICC | `217fe48ec958c667f8eef725aa27198f465df95d7662593b90d0a1cc30114356` |

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
2. Verify the committed P105 v16 schema and rebuild its manifest by both direct
   and module entry points. It transitively preserves v15/v14/v13/v12/v11/v10/v9/v8/v7/v6/v5/P68/P66/P64/P58.
3. Review `c03c321..e73a42c`; do not copy files manually and do not import
   mutable paths from the D-PCT repository.
4. Recompute `git merge-tree --write-tree e73a42c <reviewed-main>` against the
   refreshed main head.
5. Perform a normal reviewed merge of the selected payload in the main task.
6. Run all `tests/test_color_match*.py` plus halation, tiled dust and tiled
   grain tests.
7. Run the full main suite where its ignored evidence outputs and tracked
   asset-byte policy are available.
8. Keep the product default at identity fallback until a real producer
   invocation and A1/A4/A5 promotion exist.

The consumer task does not perform this merge because the main task owns its
dirty worktree, Ultimate tracker and product integration decisions.

## Current evidence

- latest detached main/P1-P104 merge: 1076 color-match tests passed, 22
  platform/data skips and zero failures;
- latest isolated consumer full suite: 1898 passed, four skipped, 36 unchanged
  environment/output/hash failures;
- detached merge tree is `7b0da229...9b6a`; the temporary worktree was removed;
- consumer worktree is clean after every stable leaf.

## External blockers that remain real

1. D-PCT must publish a genuinely different fixed invocation package after
   the current exact capability failed P44.
2. The successor must pass P45 intake, A1 reference identifiability, A4
   photographic preference/severe-tail review and A5 batch consistency
   without research override.
3. D-PCT R0CB `34af2fa` records the integrated local boundary:
   `NOT_FREEZE_READY`, with external quality, D1, D3, expert blind review,
   real RAW, real HDR, real video and license/provenance gates still failed.
   Windows runtime facts, Android compile/link-only facts and Apple
   object-only facts remain correctly separated.
4. P71 now proves the exact embedded sRGB ICC and rejects alternate metadata;
   P72 consumes that exact proof and reruns the frozen EOTF. This does not
   prove arbitrary-profile conversion, third-party ICC application or target
   runtime parity, and P69 v1 remains assumption-bound.
5. P63/P65/P67/P69/P71/P72 persisted records never authorize path reopen,
   persistence, application or delivery.
6. The main owner must review and merge the payload.

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

D-PCT R0bw at corrected commit `3a4948a` independently demonstrates the same
Vulkan 1.1 SPIR-V on NVIDIA and AMD Windows devices with repeated exact
per-device output and fail-closed negative vectors. This is useful factual
Windows host-runtime evidence for a future P60 mapping, but it creates no P45
successor capability/package and supplies neither Android device nor Apple
runtime. It therefore does not qualify P61 or open staging.
