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

- Consumer payload: `codex/reference-color-match` through P45 implementation
  at `7b0ec11`.
- Main Neuro-Film read-only snapshot: `f309c97`; its `.codex/` and `tmp/`
  files belong to the main task and were not touched.
- D-PCT read-only snapshot: `ef9a4cd`; its only callable relative-SDR
  capability is audited and rejected by P44, while ROGR-v0 also closes as
  non-callable negative development evidence.
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
| Reference-only shared batch semantics | P47 binds one source-free operator to ordered N source-bound exact output receipts | COMPLETE AS CONSUMER CONTRACT | Map only to a producer build/apply fixture that proves source-free bundle construction / D-PCT then Neuro-Film |
| Shared-batch numeric safety | P48 binds per-source producer facts, verifies output extrema/new boundary and falls back atomically | COMPLETE AS CONSUMER GUARD | Map exact producer diagnostics then replay before any shared-path authorization / D-PCT then Neuro-Film |
| Main-project availability | Latest-main synthetic merges are conflict-free and focused suites pass | READY FOR REVIEW, NOT MERGED | Repository-owner review and merge, then main-worktree full suite with local ignored evidence / main task or owner |

## Critical path

The shortest honest path to a non-identity D-PCT-backed product render is:

1. D-PCT publishes a genuinely different versioned callable package. It must
   pass P45 evaluation intake, a new exact P43 compatibility audit and P44
   A1/A4/A5 without research override.
2. Release rights and required target runtime evidence are established for
   the fixed package or its future native replacement.
3. Neuro-Film replays the fixed invocation conformance, P27 receipt binding,
   P28 atomic batch resolution, P29 numeric guard and P30 product
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
