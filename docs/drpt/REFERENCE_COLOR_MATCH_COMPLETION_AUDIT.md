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

- Consumer worktree: `codex/reference-color-match` at `0d23e27`.
- Main Neuro-Film read-only snapshot: `e7da085`; its `.codex/` and `tmp/`
  files belong to the main task and were not touched.
- D-PCT read-only snapshot: `ed01054`; its uncommitted RPSCT files belong to
  the producer task and were not touched.
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
| A1 reference identifiability | Fixed-moment, quantile, linear and current W1 routes are closed by negative evidence | CLOSED CORRECTLY, no promoted algorithm | A genuinely different D-PCT candidate must pass the frozen cross-content gate / D-PCT evidence, Neuro-Film adjudication |
| A4 photographic preference and severe tail | Baseline rejected; evaluator and ordering are frozen | GATE IMPLEMENTED, EVIDENCE OPEN | Broader frozen suite followed by independent blind aesthetic review for an identified candidate / shared evidence, Neuro-Film decision |
| A5 album/batch consistency | Baseline failed six shared-colour context probes | GATE IMPLEMENTED, EVIDENCE OPEN | Candidate-specific shared-colour drift pass with source-bound transform policy made explicit / shared evidence, Neuro-Film decision |
| Exact producer/consumer compatibility | P27 pins corrected D-PCT v2 relative-SDR schemas and exact success/failure fixtures | COMPLETE for the pinned synthetic conformance profile | A real producer package must declare and pass the same explicit compatibility profile / D-PCT then Neuro-Film |
| Actual D-PCT algorithm invocation | No frozen library/package/ABI or product invocation exists; P27 consumes fixtures only | BLOCKED ON PRODUCER ARTIFACT | Publish fixed package/ABI, capability identity, invocation conformance and real source-bound receipt / D-PCT |
| External D-PCT output to durable product staging | P33 atomically commits exact authorized N-source receipt buffers; P34 restart-verifies report and every file | MECHANICS COMPLETE, REAL USE CLOSED | Supply a real promoted/invoked batch; synthetic promotion proves mechanics only / shared evidence |
| Final user-visible delivery state | P37 verifies P36; P38 reauthorizes; P39 atomically commits local files; P40 restart-verifies report plus staging/delivered bytes | LOCAL TRANSACTION AND RESTART INTEGRITY COMPLETE, REAL USE CLOSED | Admit a real invocation only after A1/A4/A5 and merge the reviewed module into main / shared evidence and Neuro-Film integration |
| FilmFX composition | P18 binds local runs; P35 binds exact P34 external verification; P36 atomically renders its procedural branch; P37 restart-verifies report, inputs and outputs | VERIFIED STAGING COMPLETE, DELIVERY OPEN | Bind a real promoted producer invocation before final product delivery; physical halation still requires separately resolved controls / Neuro-Film |
| SDR colour rail | Local product accepts display-linear relative linear-sRGB/Rec.2020; P27 compatibility is narrow relative sRGB | PARTIAL BY EXPLICIT PROFILE | Add only producer-published compatible rails and versioned trusted bridges / D-PCT rail, Neuro-Film adapter |
| RAW, HDR/gain-map and video | Deliberately outside this consumer branch; D-PCT absolute BT.2020 HDR is explicitly unmapped | OPEN, CORRECTLY SEPARATED | Producer media evidence plus explicit scene/display bridge and new compatibility profile; never relabel absolute HDR as relative SDR / D-PCT then Neuro-Film |
| Portable consumer identity chain | Python, MSVC and LLVM-MinGW execute exact P28-P30 vectors; Android arm64/x86_64 link | COMPLETE for host identity logic | Android device execution, Apple compiler/runtime, JNI/Swift boundary and real invocation remain open / platform integration |
| Main-project availability | Latest-main synthetic merges are conflict-free and focused suites pass | READY FOR REVIEW, NOT MERGED | Repository-owner review and merge, then main-worktree full suite with local ignored evidence / main task or owner |

## Critical path

The shortest honest path to a non-identity D-PCT-backed product render is:

1. D-PCT freezes a real invocation package/ABI and emits a source-bound factual
   receipt under an explicit compatible SDR profile.
2. That exact candidate passes A1, A4 and A5 without research override.
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

The independent Neuro-Film reference-match module is implementation-complete
as a fail-closed local product shell. The broader long-term goal is not
complete: no real external candidate has both product promotion and a frozen
invocation artifact, external pixels have only a synthetic-tested staging
transaction and no final delivery state, Apple/device runtime evidence is
absent, and the branch has not been merged into the main project.
