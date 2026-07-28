# Reference Color Match Completion Audit

Date: 2026-07-28

Status: **consumer module complete; real external-algorithm product delivery
not yet complete**.

## Purpose

This audit separates implemented Neuro-Film product machinery from evidence
that still depends on the equal D-PCT producer task, target-platform runtime
access, or repository-owner integration. It is a completion map, not a new
wire contract and not an algorithm promotion.

## Audited scope and ownership

- Consumer reviewed payload: `codex/reference-color-match` through P95 at
  `2c33809`; P96 v14 review evidence binds it without self-inclusion.
- Main Neuro-Film latest read-only snapshot observed during P96 is `8dcfdac`;
  its concurrent AN0 work and untracked `.codex/`/`tmp/` belong to the main
  task and were not touched.
- D-PCT latest observed core snapshot is `46b77bb`; its polynomial RGB ABI is
  controlled-colorimetric only and is not a P45 arbitrary-look successor.
  NFCM has Android 14 x86_64 emulator evidence for its consumer quantizer, but
  no arm64 physical-device run.
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
| A1 reference identifiability | P44 exact `dpct-chroma` improves 0/30 cross-content rows, median -201.29% | CURRENT INVOKED CANDIDATE REJECTED | Publish a genuinely different versioned capability and rerun the frozen gate / D-PCT evidence, Neuro-Film adjudication |
| A4 photographic preference and severe tail | P44 current candidate fails 6/6 photographic probes; no blind review opens | CURRENT INVOKED CANDIDATE REJECTED | Only an automated passing candidate may enter independent blind review / shared evidence, Neuro-Film decision |
| A5 album/batch consistency | P44 current candidate fails 6/6 shared-colour probes with worst median drift 72.64 Delta E76 | CURRENT INVOKED CANDIDATE REJECTED | Require a new capability with bounded adaptation or fixed operator and rerun / shared evidence, Neuro-Film decision |
| Current candidate failure diagnosis | P46 proves 30/30 universal overcorrection, weak clipping/error association and 6/6 source-context bundle changes | COMPLETE NEGATIVE DIAGNOSIS | Prioritize a bounded reference-only shared operator with neutral/boundary controls / D-PCT research |
| Exact producer/consumer compatibility | P27 pins v2 envelopes; P43 pins producer source/wheel/runtime/request/response and invokes through the P27 adapter | COMPLETE for fixed local relative-SDR package | Preserve both package and lower-envelope identities; re-audit any producer package change / D-PCT then Neuro-Film |
| Actual D-PCT algorithm invocation | P43 executes exact verified wheel bytes and independently rebuilds a candidate receipt | LOCAL RESEARCH INVOCATION COMPLETE, PRODUCT ADMISSION CLOSED | Pass genuine A1/A4/A5, establish release rights and target runtime evidence / shared evidence then Neuro-Film |
| External D-PCT output to durable product staging | P33 atomically commits exact authorized N-source receipt buffers; P34 restart-verifies report and every file | MECHANICS COMPLETE, REAL USE CLOSED | Supply a real promoted/invoked batch; synthetic promotion proves mechanics only / shared evidence |
| Final user-visible delivery state | P37 verifies P36; P38 reauthorizes; P39 atomically commits local files; P40 restart-verifies report plus staging/delivered bytes | LOCAL TRANSACTION AND RESTART INTEGRITY COMPLETE, REAL USE CLOSED | Admit a real invocation only after A1/A4/A5 and merge the reviewed module into main / shared evidence and Neuro-Film integration |
| FilmFX composition | P18 binds local runs; P35 binds exact P34 external verification; P36 atomically renders its procedural branch; P37 restart-verifies report, inputs and outputs | VERIFIED STAGING COMPLETE, DELIVERY OPEN | Bind a real promoted producer invocation before final product delivery; physical halation still requires separately resolved controls / Neuro-Film |
| SDR colour rail | Local product accepts display-linear relative linear-sRGB/Rec.2020; P27 compatibility is narrow relative sRGB | PARTIAL BY EXPLICIT PROFILE | Add only producer-published compatible rails and versioned trusted bridges / D-PCT rail, Neuro-Film adapter |
| RAW, HDR/gain-map and video | Deliberately outside this consumer branch; D-PCT absolute BT.2020 HDR is explicitly unmapped | OPEN, CORRECTLY SEPARATED | Producer media evidence plus explicit scene/display bridge and new compatibility profile; never relabel absolute HDR as relative SDR / D-PCT then Neuro-Film |
| Portable consumer identity chain | P42 routes exact P28-P30 vectors through one freestanding C ABI; MSVC/LLVM-MinGW execute, Android arm64/x86_64 link, macOS/iOS arm64 objects compile | COMPLETE for host identity and cross-target core compilation | Android/Apple device runtime, SDK/app/JNI/Swift boundary and real invocation remain open / platform integration |
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
| Decoded samples to MatchView | P69 remains the immutable assumption-bound v1; P72 v2 accepts only P71-attested batches, reruns P71 and the exact float32 IEC sRGB EOTF, and binds profile/metadata attestation identities into every MatchView provenance chain; P76 removes its runtime-generator dependency | COMPLETE AS METADATA-ATTESTED PROCESS-LOCAL BRIDGE | Still no persistence/application/delivery authority; target-platform profile-byte and runtime evidence remain separate / Neuro-Film |
| Deterministic SDR output quantization | P87 freezes the current float32 staging OETF plus 8/16-bit round-to-nearest behavior as first-float32 threshold tables; P88 runs 24MP/72M scalars twice through MSVC and LLVM-MinGW with exact replay/oracle output and <=8.39MB tracked arrays, while Android remains link-only and Apple object-only | COMPLETE AS PORTABLE CONSUMER ABI, DEVICE RUNTIME OPEN | Bind this ABI into a future platform shell only after actual target runtime evidence; no producer or algorithm admission follows / Neuro-Film |
| Shared-path durable staging | P50 atomically commits exact P49-authorized outputs plus a P47/P48/P49-bound report and restores prior bytes on failure | MECHANICS COMPLETE, REAL USE CLOSED | Restart-verify P50, then bind optional composition only for a real promoted producer / Neuro-Film |
| Shared-path restart verification | P51 caller-binds report/run/auth/guard/operator and rehashes every P50 file without writes | COMPLETE AS RESTART-SAFE VERIFIER | A later shared composition/delivery path must consume this exact verification / Neuro-Film |
| Shared-path FilmFX ownership and staging | P52 preserves the verified shared look as sole colour owner; P53 reruns P51 and atomically stages only profile-bound procedural effects afterward | COMPLETE THROUGH ROLLBACK-SAFE STAGING, NOT DELIVERY | Restart-verify the exact P53 report, inputs and outputs before any later authorization; never infer stock/calibrated identity / Neuro-Film |
| Shared-path FilmFX restart integrity | P54 caller-binds the exact P53 report/run and read-only rehashes every P50 base and P53 output while preserving the P49/P48/P47 receipt chain | COMPLETE AS VERIFIED STAGING, NOT AUTHORIZATION | A later authorization must consume the exact P54 verification and still requires a real promoted shared producer / Neuro-Film |
| Shared-path local-delivery authority | P55 reruns P54, rebinds P52/P51/P50/P49 and emits only a canonical no-write local scope when every product-ready source remains authorized | COMPLETE AS AUTHORIZATION MECHANICS, REAL USE CLOSED | Atomically deliver only from exact P55 after a genuine shared producer passes P45/P49 / Neuro-Film |
| Shared-path atomic local export | P56 reconstructs exact P55, protects all staging artifacts and atomically commits byte-identical ordered files plus report with rollback | COMPLETE AS LOCAL TRANSACTION MECHANICS, REAL USE CLOSED | Restart-verify the exact P56 report, staging and delivered bytes / Neuro-Film |
| Shared-path local-export restart integrity | P57 caller-binds exact P56 report/delivery and read-only rehashes every P53 source and delivered file while preserving P55/P54/receipt lineage | COMPLETE AS VERIFIED LOCAL FILE MECHANICS, REAL USE CLOSED | Main integration and a genuine P45/P49-passing producer remain required / shared evidence |
| Main-integration evidence | Earlier manifests remain immutable; P96 v14 binds P1-P95 `2c33809`, 376 payload paths, 47 exports, 20 schemas, exact v13 identity and zero overlap against main `8dcfdac`; P74's CRLF checkout protection remains transitive | COMPLETE AS REVIEW MANIFEST, NOT MERGED | Repository owner verifies v14, reviews and merges / main task or owner |
| Main-integration wire validation | P59 independently validates the strict manifest shape before Git access, then P58 reconstructs exact commit-derived facts | COMPLETE AS FAIL-CLOSED REVIEW CONTRACT | Keep schema and pinned Git reconstruction together during owner review / main task or owner |
| Main-project availability | P96 payload remains zero-overlap against main `8dcfdac` and merges conflict-free as tree `8d869107`; a fresh detached merge passes 1040 color-match tests with 22 platform/data skips and zero failures | READY FOR REVIEW, NOT MERGED | Repository-owner review and merge, then main-worktree full suite with local ignored evidence / main task or owner |

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
from fail-closed intake through verified local export mechanics. The broader
long-term goal is not complete: local exact-wheel invocation is now verified,
but no real external candidate is product-promoted, producer redistribution
rights and Apple/Android device runtime evidence are absent, and the reviewed
payload has not been merged into the main project.
