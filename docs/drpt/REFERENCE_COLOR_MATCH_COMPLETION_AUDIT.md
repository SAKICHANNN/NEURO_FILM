# Reference Color Match Completion Audit

Date: 2026-07-29

Status: **consumer module complete; real external-algorithm product delivery
not yet complete**.

## Purpose

This audit separates implemented Neuro-Film product machinery from evidence
that still depends on the equal D-PCT producer task, target-platform runtime
access, or repository-owner integration. It is a completion map, not a new
wire contract and not an algorithm promotion.

## Audited scope and ownership

- Consumer branch now continues through P172; v42 is the latest immutable
  main-review manifest and binds P1-P168, including local 24 MP,
  maximum-count, bounded SDR file-format, unsupported-media atomicity,
  BT.2020 SDR and advertised-output execution evidence.
- Main Neuro-Film latest bound committed snapshot is `f61c131`; its untracked
  `.codex/`/`tmp/` belong to the main task and were not touched.
- D-PCT latest stable communicated snapshot is `4b3e857`; R0EE independently
  qualifies 15 rights-cleared native 2x2 RGB Bayer rows across eight makes
  and all four patterns. R0EF then freezes all 15 even-aligned 1024 crops plus
  one 3040x2024 full frame before execution. MSVC, LLVM-MinGW and Python,
  across two processes and two internal evaluations, are byte-exact at
  `057907a9...780f3`; maximum absolute error/RMSE are
  `2.3841858e-7`/`1.4531343e-8`, and the full-frame workspace is about
  234.7 MiB. This proves rights-cleared native-mosaic decoding plus private
  portable arithmetic only. There is no scene truth, quality, native
  sensor-noise/optics/colour or product admission, and no
  package, compatible consumer rail, public schema, receipt or capability.
  R0EG additionally executes one non-identity DDFAPD probe eight times over
  two independent runners and four cold/wiped Android 14 x86_64 virtual-device
  boots. Every Android JSON result exactly equals the frozen MSVC/LLVM oracle;
  stable identity is `b3ce13b0...12797`, and both runners finish with zero
  owned emulator/QEMU processes. This is private virtual-device arithmetic
  only, not physical arm64, JNI/app/media, native-RAW quality, Apple runtime
  or a public producer interface.
  R0EH then proves a conservative 12-row-halo, full-width stripe assembly
  byte-exact against the unchanged full-frame C output over 72 synthetic
  combinations and all 15 native crops. At the correct 3040x2024 geometry,
  its compute-only workspace falls from 61,529,600 to 2,675,200 float32
  values (-95.6522%). The stripe assembler remains a private execution
  hypothesis without a public tiled ABI, package, schema, receipt or
  capability.
  R0EI realizes a separate-header private atomic tiled C ABI while preserving
  the frozen v1 header and R0EG oracle identity. Across MSVC/LLVM, 72
  synthetic and 15 native cases remain byte-exact against full-frame output;
  five invalid-input/contract classes preserve caller output and diagnostics.
  Corrected 3040x2024 geometry gives an honest atomic workspace of 21,936,640
  float32 values (-64.3478% versus full frame), not the earlier transposed
  model. It
  remains a private CPU ABI with no public product contract.
  R0EJ reproducibly links Android arm64/x86_64 and emits macOS/iOS arm64
  two-object bundles with exactly four historical-plus-tiled exports and zero
  unresolved symbols at each final boundary. On the 3040x2024 native CFA,
  MSVC/LLVM tiled outputs remain byte-exact to full v1 and formal
  tiled/full timing ratios are 0.9328--1.6096, below the frozen 2.0 ceiling.
  This adds target build evidence and local Windows performance only, not
  Android/Apple runtime or a public producer interface.
  R0EK executes a two-stripe non-identity probe eight times across two
  independent runners and four cold/wiped Android 14 x86_64 virtual-device
  boots. Full/tiled output, replay and two failure-atomicity paths exactly
  match the frozen MSVC/LLVM oracle at stable identity
  `21a5ecc4...29f6c`; owned emulator/QEMU processes and temporary mappings
  return to zero. This remains private virtual-device arithmetic, not physical
  arm64, JNI/app/media, RAW quality or public capability evidence.
  P172 therefore remains fail-closed and R0DO/R0DP absolute-HDR diagnostics
  remain explicitly unmapped.
  NFCM has Android 14 x86_64 emulator evidence for its consumer quantizer, but
  no arm64 physical-device run. P106 confirms the current official QEMU2
  emulator also rejects arm64 images on this x86_64 host before boot.
- Neuro-Film owns one-reference/N-source product semantics, replay,
  transaction, reporting, A1/A4/A5, delivered-pixel guards and FilmFX
  composition.
- D-PCT owns matching algorithms, canonical media rails, RAW/HDR/video,
  producer packages/ABI and native algorithm execution.

## Requirement matrix

| Requirement | Current evidence | State | Required next evidence / owner |
|---|---|---|---|
| One uploaded reference plus ordered N sources | P1-P8 local fit/render/replay and P28 external per-source binding | COMPLETE | Maintain order and shared-reference identity in future adapters / Neuro-Film |
| Immutable replayable look intent | `ReferenceLookRecipe` JSON/canonical identity and stored-reference-free replay | COMPLETE for local baseline | A future D-PCT-backed recipe must reference fixed producer bundle identities rather than copy their parameter layout / Neuro-Film |
| Rollback-safe image/recipe/report transaction | P16-P18 transactional local file path with fault injection | COMPLETE for current local renderer | Bind P30 authorization and real external output receipts to a separately versioned transaction leaf / Neuro-Film |
| Product safety and claim ceiling | Identity default, strict research override, P29 numeric guard, P30 staging authorization | COMPLETE as fail-closed machinery | No state above `authorized-for-staging` until every later gate passes / Neuro-Film |
| A1 reference identifiability | P111 exact BMKL improves 13/30 cross-content rows, median -12.06%, worst -122.56%; this is safer than P44 D-PCT but still fails | LATEST INVOKED CANDIDATE REJECTED | A genuinely different versioned capability must rerun the same frozen gate / D-PCT evidence, Neuro-Film adjudication |
| A4 photographic preference and severe tail | P111 BMKL passes 6/6 automated photographic probes with zero new boundary, but A1/A5 failure prevents blind review | AUTOMATED SAFETY PASS, PRODUCT REJECTED | Blind review opens only for a candidate that also clears upstream automated gates / shared evidence, Neuro-Film decision |
| A5 album/batch consistency | P111 BMKL fails 6/6 shared-colour probes with worst median/p95/max drift 21.99/35.62/44.74 Delta E76 | LATEST INVOKED CANDIDATE REJECTED | A source-independent or tightly bounded adaptation strategy must pass unchanged gates / shared evidence, Neuro-Film decision |
| Current candidate failure diagnosis | P46 proves 30/30 universal overcorrection, weak clipping/error association and 6/6 source-context bundle changes | COMPLETE NEGATIVE DIAGNOSIS | Prioritize a bounded reference-only shared operator with neutral/boundary controls / D-PCT research |
| Exact producer/consumer compatibility | P27 pins v2 envelopes; P43 pins producer source/wheel/runtime/request/response and invokes through the P27 adapter | COMPLETE for fixed local relative-SDR package | Preserve both package and lower-envelope identities; re-audit any producer package change / D-PCT then Neuro-Film |
| Capability-neutral producer invocation | P107 binds full producer/package/wheel/runtime/wire/capability/rights identity, preserves v1 wire bytes, executes the exact installed wheel and independently reconstructs request identity before output verification | COMPLETE AS FAIL-CLOSED TRANSPORT, NO NEW CANDIDATE | Instantiate only from a genuinely different producer package and rerun P45/P44; profile substitution cannot promote a candidate / D-PCT then Neuro-Film |
| Actual D-PCT algorithm invocation | P43 executes exact verified wheel bytes and independently rebuilds a candidate receipt | LOCAL RESEARCH INVOCATION COMPLETE, PRODUCT ADMISSION CLOSED | Pass genuine A1/A4/A5, establish release rights and target runtime evidence / shared evidence then Neuro-Film |
| External D-PCT output to durable product staging | P33 atomically commits exact authorized N-source receipt buffers; P34 restart-verifies report and every file | MECHANICS COMPLETE, REAL USE CLOSED | Supply a real promoted/invoked batch; synthetic promotion proves mechanics only / shared evidence |
| Final user-visible delivery state | P37 verifies P36; P38 reauthorizes; P39 atomically commits local files; P40 restart-verifies report plus staging/delivered bytes | LOCAL TRANSACTION AND RESTART INTEGRITY COMPLETE, REAL USE CLOSED | Admit a real invocation only after A1/A4/A5 and merge the reviewed module into main / shared evidence and Neuro-Film integration |
| FilmFX composition | P18 binds local runs; P35 binds exact P34 external verification; P36 atomically renders its procedural branch; P37 restart-verifies report, inputs and outputs | VERIFIED STAGING COMPLETE, DELIVERY OPEN | Bind a real promoted producer invocation before final product delivery; physical halation still requires separately resolved controls / Neuro-Film |
| SDR colour rail | Local product accepts display-linear relative linear-sRGB/Rec.2020; P27 compatibility is narrow relative sRGB | PARTIAL BY EXPLICIT PROFILE | Add only producer-published compatible rails and versioned trusted bridges / D-PCT rail, Neuro-Film adapter |
| File input compatibility preflight | P148 hashes and decodes a bounded ordered input batch through the main `WorkingImage` loader, reports the actual rail and accepts only display-linear sRGB/Rec.2020; its claim is explicitly non-authorizing | COMPLETE AS ADVISORY PRODUCT-SHELL CONTRACT | Render still revalidates every input; add rails only after main/producer colour-state support is versioned / Neuro-Film |
| Unified product-shell discovery | P170 publishes one strict read-only payload for operations, 1--64 ordered sources, accepted decoded rails, recipe semantics, exact existing output/metadata contracts and the current identity-fallback delivery truth. The legacy output-only capability response is unchanged | COMPLETE AS LOCAL UI/IPC DISCOVERY | Re-version rather than mutate when a rail, encoder, metadata policy or promoted algorithm changes / Neuro-Film |
| SDR JPEG/TIFF transaction matrix | P163 executes JPEG8, TIFF8 and profiled TIFF16 input-to-output file transactions twice at 2048x1536. Every case preserves decoded display-linear sRGB, declared format/depth, exact output/recipe/normalized-report replay and identity fallback under 361 MB peak | EXACT LOCAL WINDOWS/PYTHON MATRIX PASS | Add only separately frozen formats/metadata semantics and obtain target-platform execution; do not infer arbitrary codec, profile, RAW/HDR or product readiness / Neuro-Film |
| BT.2020 SDR PNG/CICP transaction | P165 executes BT.2020-only and ordered sRGB/BT.2020 mixed 16-bit PNG transactions twice at 2048x1536. Rail/profile order, output/recipe/report replay, identity fallback and <=515 MB peak all pass | EXACT LOCAL RELATIVE-SDR WINDOWS/PYTHON PASS | Keep absolute HDR/PQ/HLG and arbitrary profile conversion unmapped; obtain target-platform/media evidence separately / Neuro-Film |
| Advertised output capability truth | P166 executes all nine tuples exposed by `reference-file-output-capabilities.v1` twice, including `.jpeg`/`.tif` aliases. Public inventory, format/depth/profile, artifacts, fallback and cleanup all match exactly | COMPLETE AS LOCAL EXECUTABLE V1 ADVERTISEMENT | Re-version the capability contract for any new output tuple and obtain target-platform evidence separately / Neuro-Film |
| Output metadata minimization | P168 independently layers a hash-bound streaming metadata policy over all nine advertised tuples and enforces it before transaction publication. Exact ICC/cICP/JFIF/structural TIFF metadata is retained while source EXIF/GPS/XMP/IPTC/text/private metadata and non-identity orientation fail closed. Two 24 MP PNG16 runs remain exact at 1.875 GB peak | COMPLETE FOR CURRENT REFERENCE-MATCH ENCODERS | Re-version and re-audit for every new encoder/format; this is not an arbitrary third-party sanitizer / Neuro-Film |
| Unsupported-media batch atomicity | P164 places transparent RGBA PNG, two-page TIFF or a pinned real libultrahdr MPO second in a two-source transaction. All six runs encode source one, reject source two, preserve 24/24 output/recipe/report target hashes and leave zero staging residue | EXACT LOCAL THREE-FIXTURE FAIL-CLOSED PASS | Keep decoder claims narrow; extend only with separately pinned real fixtures and do not infer complete HDR/gain-map/media detection / Neuro-Film |
| High-resolution image and batch lifetime | P159 passes two exact 24 MP single-source runs. P160 passes two ordered three-source 24 MP runs; P161 releases encoded renders before the next load and lowers interleaved median peak 2.167 -> 1.886 GB with exact artifacts and no wall regression. P162 passes two exact 64-source 1 MP maximum-count transactions at about 167 MB peak | EXACT LOCAL 24 MP THREE-SOURCE PLUS 64-SOURCE 1 MP PASS | Obtain real target-platform/media execution; do not combine the separate scale/count facts into a 24MP-by-64 claim / Neuro-Film |
| RAW, HDR/gain-map and video | Deliberately outside this consumer branch; D-PCT absolute BT.2020 HDR is explicitly unmapped. P172 proves that a successfully decoded, orientation-applied scene-linear RAW is reported as `unsupported-decoded-rail`; when it is the second source, all existing destinations remain byte-exact and no transaction residue survives | OPEN, CORRECTLY SEPARATED AND FAIL-CLOSED | Producer media evidence plus explicit scene/display bridge and new compatibility profile; never relabel scene-linear or absolute HDR as relative SDR / D-PCT then Neuro-Film |
| HDR shot reuse invalidation | P109 pins R0cn model/assessment schemas and fixture, reconstructs both producer identities and emits a persisted consumer decision with `reuse_authorized=false`; only `invalidate-reuse` forces refit | COMPLETE AS VETO-ONLY MAPPING, NO CACHE AUTHORITY | A future shot cache may consume the veto only after exact descriptor/input invocation binding; `not-invalidated` never authorizes reuse / D-PCT then Neuro-Film |
| Portable consumer identity chain | P42 routes exact P28-P30 vectors through one freestanding C ABI; MSVC/LLVM-MinGW execute, Android arm64/x86_64 link, macOS/iOS arm64 objects compile. P101-P104 additionally execute all ten frozen canonical payloads, SHA failure atomicity and all eight staging predicate inputs through JNI on two cold Android 14 x86_64 emulator boots with stable identity `a3fa50e0...e03f20` | COMPLETE for host identity, cross-target compilation and Android x86_64 virtual runtime | Physical arm64 and Apple runtime remain open; emulator evidence is not a physical-device or producer-algorithm claim / platform integration |
| Successor-candidate substitution | P45 requires new exact producer/package/wheel/capability identities, explicit fit/batch semantics and two-stage readiness | COMPLETE AS FAIL-CLOSED INTAKE | Populate only after a producer publishes a genuinely new callable package / D-PCT then Neuro-Film |
| Successor target-runtime evidence | P60 binds exact declaration/producer/capability/profile and per-target environment matrix, binary/report hashes, proof class and repeated factual gates; corrected D-PCT R0bw `3a4948a` supplies actual dual-vendor Windows Vulkan host runtime, R0bx is Android compile/link-only, and R0by is an existing source-bound Windows CPU ABI | COMPLETE AS CONSUMER EVIDENCE CONTRACT, REAL MATRIX WINDOWS-ONLY/UNMAPPED | Publish a fixed successor declaration, bind applicable runtime reports, and supply macOS/iOS/Android runtime reports; compile/link/object evidence never substitutes / D-PCT then Neuro-Film |
| Reference-only shared batch semantics | P47 binds one source-free operator to ordered N source-bound exact output receipts | COMPLETE AS CONSUMER CONTRACT | Map only to a producer build/apply fixture that proves source-free bundle construction / D-PCT then Neuro-Film |
| Shared-batch numeric safety | P48 binds per-source producer facts, verifies output extrema/new boundary and falls back atomically | COMPLETE AS CONSUMER GUARD | Map exact producer diagnostics then replay before any shared-path authorization / D-PCT then Neuro-Film |
| Shared-path product authorization | P49 requires product-ready P45 admission, exact P47/P48 binding and an independent promoted decision; P61 additionally binds factual P60 runtime evidence | COMPLETE THROUGH RUNTIME QUALIFICATION, REAL USE CLOSED | A real callable shared producer must pass every lock and a durable consumer must require exact P61; declaration booleans alone are insufficient / D-PCT then Neuro-Film |
| Shared-path runtime qualification | P61 embeds and replays exact P60 against exact P49; only four factual target proofs plus upstream authorization yield runtime-qualified staging | COMPLETE AS NO-WRITE GUARD | P62 consumes only an exact caller-pinned P61; never relabel historical P50 / Neuro-Film |
| Runtime-qualified shared staging | P62 snapshots pixels, binds P61/P60/P49/P48/P47 and create-only publishes outputs followed by one canonical report commit marker | COMPLETE AS MANIFEST-LAST STAGING MECHANICS, REAL USE CLOSED | Report-less orphans are never consumed or auto-deleted; P63 verifies the committed run / Neuro-Film |
| Runtime-qualified handle observation | P63 caller-pins P62 report/run/qualification, opens each object once, binds handle identity, double-reads and final-rehashes the same handle under bounded counts/sizes | COMPLETE AS OBSERVATION, PATH CONSUMPTION CLOSED | Windows denies write/delete sharing during verification; POSIX is sequential observation with an irreducible post-read window. P65 must consume bytes while the same verified handles remain live / Neuro-Film |
| Same-handle byte consumption | P65 captures bounded immutable output bytes during the same P63 handle session and returns them only after every final rehash; its record carries no artifact path and fixes path/persistence/delivery authority false | COMPLETE AS PROCESS-LOCAL BYTE SNAPSHOT, DECODE/DELIVERY CLOSED | P67 must decode only returned bytes and prove declared encoded format/depth/frame/geometry before any pixel consumer / Neuro-Film |
| Path-free encoded decode | P67 preflights the entire batch then strictly decodes one-page/frame RGB PNG/JPEG/TIFF to readonly uint8/uint16 arrays with exact decoded hashes; P71 independently re-inspects the exact bytes and requires the frozen sRGB ICC plus a narrow no-conflict metadata policy; P76 pins those 588 ICC bytes as an independent canonical fixture | COMPLETE AS STRUCTURAL/SAMPLE AND EMBEDDED-METADATA EVIDENCE | This proves the encoded metadata carried by P62, not arbitrary-profile conversion or that a third-party renderer honored ICC / Neuro-Film |
| Decoded samples to MatchView | P69 remains the immutable assumption-bound v1; P72 v2 accepts only P71-attested batches, reruns P71 and the exact float32 IEC sRGB EOTF, and binds profile/metadata attestation identities into every MatchView provenance chain; P76 removes its runtime-generator dependency. P97-P99 execute the exact 588-byte ICC accessor and all 8/16-bit EOTF codes twice on Android 14 x86_64 emulator | COMPLETE AS METADATA-ATTESTED PROCESS-LOCAL BRIDGE PLUS ANDROID VIRTUAL SDR-BOUNDARY RUNTIME | Encoded media parsing, ICC application/conversion, physical arm64 and Apple runtime remain separate / Neuro-Film |
| Deterministic SDR output quantization | P87 freezes the current float32 staging OETF plus 8/16-bit round-to-nearest behavior as first-float32 threshold tables; P88 runs 24MP/72M scalars twice through MSVC and LLVM-MinGW with exact replay/oracle output and <=8.39MB tracked arrays. P95/P99 execute dual-ABI packaging on Android 14 x86_64 emulator with exact repeated 4096-vector outputs and complete EOTF-to-OETF code roundtrip | COMPLETE AS PORTABLE CONSUMER ABI AND ANDROID VIRTUAL RUNTIME; PHYSICAL/APPLE OPEN | Obtain arm64 physical-device and Apple runtime evidence; no producer or algorithm admission follows / Neuro-Film |
| Shared-path durable staging | P50 atomically commits exact P49-authorized outputs plus a P47/P48/P49-bound report and restores prior bytes on failure | MECHANICS COMPLETE, REAL USE CLOSED | Restart-verify P50, then bind optional composition only for a real promoted producer / Neuro-Film |
| Shared-path restart verification | P51 caller-binds report/run/auth/guard/operator and rehashes every P50 file without writes | COMPLETE AS RESTART-SAFE VERIFIER | A later shared composition/delivery path must consume this exact verification / Neuro-Film |
| Shared-path FilmFX ownership and staging | P52 preserves the verified shared look as sole colour owner; P53 reruns P51 and atomically stages only profile-bound procedural effects afterward | COMPLETE THROUGH ROLLBACK-SAFE STAGING, NOT DELIVERY | Restart-verify the exact P53 report, inputs and outputs before any later authorization; never infer stock/calibrated identity / Neuro-Film |
| Shared-path FilmFX restart integrity | P54 caller-binds the exact P53 report/run and read-only rehashes every P50 base and P53 output while preserving the P49/P48/P47 receipt chain | COMPLETE AS VERIFIED STAGING, NOT AUTHORIZATION | A later authorization must consume the exact P54 verification and still requires a real promoted shared producer / Neuro-Film |
| Shared-path local-delivery authority | P55 reruns P54, rebinds P52/P51/P50/P49 and emits only a canonical no-write local scope when every product-ready source remains authorized | COMPLETE AS AUTHORIZATION MECHANICS, REAL USE CLOSED | Atomically deliver only from exact P55 after a genuine shared producer passes P45/P49 / Neuro-Film |
| Shared-path atomic local export | P56 reconstructs exact P55, protects all staging artifacts and atomically commits byte-identical ordered files plus report with rollback | COMPLETE AS LOCAL TRANSACTION MECHANICS, REAL USE CLOSED | Restart-verify the exact P56 report, staging and delivered bytes / Neuro-Film |
| Shared-path local-export restart integrity | P57 caller-binds exact P56 report/delivery and read-only rehashes every P53 source and delivered file while preserving P55/P54/receipt lineage | COMPLETE AS VERIFIED LOCAL FILE MECHANICS, REAL USE CLOSED | Main integration and a genuine P45/P49-passing producer remain required / shared evidence |
| Main-integration evidence | Earlier manifests remain immutable; v42 binds P1-P170 payload `544c6b5`, 586 payload paths, 70 exports, 26 schemas, exact v41 identity and zero overlap against main `f61c131`; manifest/schema SHA-256 are `d6e2815b...12f17c` / `74c08d1f...a4eb2` | COMPLETE AS REVIEW MANIFEST, NOT MERGED | Repository owner verifies v42, reviews and merges / main task or owner |
| Main-integration wire validation | P59 independently validates the strict manifest shape before Git access, then P58 reconstructs exact commit-derived facts | COMPLETE AS FAIL-CLOSED REVIEW CONTRACT | Keep schema and pinned Git reconstruction together during owner review / main task or owner |
| Main-project availability | P170 payload remains zero-overlap against main `f61c131` and merges conflict-free as tree `6f7c0cce...b3876`; detached merge `be9e126` passes 985 non-manifest color/reference-match tests with 30 platform/data skips; direct v41/v42 rebuild/tamper is 10/10. The tests-only P172 successor also merges conflict-free with main `0a99ad0a` at tree `0b79cba6...e0e65` and passes 41/42 file/capability checks with one environment skip | READY FOR REVIEW, NOT MERGED | Repository-owner review and merge, then main-worktree full suite with local ignored evidence / main task or owner |

## Critical path

The shortest honest path to a non-identity D-PCT-backed product render is:

1. D-PCT publishes a genuinely different versioned callable package. It must
   pass P45 evaluation intake, a new exact P43 compatibility audit and P44
   A1/A4/A5 without research override.
2. Release rights and required target runtime evidence are established for
   the fixed package or its future native replacement.
3. Neuro-Film selects the declared semantics. A per-source capability replays
   P27-P30. A genuinely reference-only capability replays P47 exact
   build/apply binding, P48 numeric guard and P49 promotion-bound staging
   authorization.
4. P33 stages and commits all N outputs and its exact report atomically, then
   P34 restart-verifies the report and every file.
5. P35 binds optional procedural FilmFX to the exact P34 verification without
   adding film colour. P36 re-verifies P34 and atomically renders that exact
   procedural plan to separate staging outputs with ceiling
   `filmfx-staging-not-delivered`.
6. A later product-delivery decision consumes the exact P36 run; physical
   halation requires a separate resolved-control binding. P37 must first
   reverify the P36 report plus every recorded input and output file.
7. P38 must then rebind P30, P34, P35 and live P37 before issuing the
   no-write `authorized-local-delivery-not-committed` capability.
8. P39 reconstructs that exact authorization immediately before atomically
   committing byte-identical local files and a delivery report.
9. P40 binds the caller-held report hash/delivery ID and rehashes every
   staging and delivered file before a later replay can trust the export.
10. Platform runtime evidence is collected per target; cross-compilation alone
   cannot close device support.
11. On the shared path, P52 binds ownership, P53 reruns P51 before atomically
    staging procedural FilmFX, and P54 restart-verifies the exact P53 report,
    P50 bases and FilmFX outputs. P55 binds exact P54 to product-ready P49 but
    creates no destination; P56 binds exact P55 and atomically commits local
    files; P57 restart-verifies exact P56 before any replay/trust path.
12. The main owner verifies the P58 deterministic integration manifest before
    merging; the manifest is review evidence and never upgrades candidate,
    transaction or product state.
13. P59 validates the manifest against its strict schema before Git-backed
    reconstruction; neither validation stage performs the merge.
14. P60 must bind actual host/device runtime reports to the exact successor
    declaration. P45 runtime booleans alone are not factual runtime evidence.
15. P61 must qualify the exact P49 authorization before any new durable
    staging path; historical P50 remains explicitly pre-runtime-qualification.
16. P62 may create a runtime-qualified staged run only from the exact
    consumer-pinned P61. Its report is published last; report-less output
    orphans never constitute a transaction.
17. P63 must bind the caller-held P62 report hash/run/qualification and hash
    the same open file handles whose identities it validates. Path lookup
    followed by a second path open is not sufficient.
18. A persisted P63 record is evidence of a completed bounded observation,
    not authority to reopen or consume any path later. P65 must integrate
    verification and consumption on the same live handles; Windows share
    denial and POSIX sequential observation must retain distinct claims.
19. P65 closes the later path reopen for encoded bytes only by returning a
    process-local immutable snapshot after every same-handle final rehash.
    Its persisted receipt grants no path, persistence or delivery authority;
    P67 must validate the encoded image entirely from those bytes.
20. P67 structural/sample decoding and P69 sRGB EOTF execution do not by
    themselves prove embedded colour semantics. P71 now attests the exact
    expected sRGB profile and rejects alternate metadata in memory; P72 v2
    consumes that proof. P69 v1 remains historical assumption-bound evidence
    and cannot be retroactively relabeled.

## Non-blocking work policy

The producer can continue algorithm research without waiting for this branch.
This branch can continue consumer-only audit, replay and integration
preparation, but it must not invent a producer ABI or use synthetic promoted
vectors to cross the transaction boundary. Absolute HDR, RAW and video remain
separate leaves and do not block the relative-SDR product shell from review.

## Verdict

The independent Neuro-Film reference-match consumer is implementation-complete
from advisory input/output capability discovery through fail-closed intake and
verified local export mechanics. P160 passes two exact ordered three-source
24 MP local batches below 2.25 GiB; P161 removes the evidenced prior-render
lifetime overlap and lowers median peak to about 1.886 GB without output or
wall regression. The broader long-term goal is not complete: local exact-wheel
invocation and Android x86_64 virtual
SDR-boundary runtime are verified, but no real external candidate is
product-promoted, producer redistribution rights and Apple/physical-Android
runtime evidence are absent, and reviewed v42 has not been merged into the
main project.
