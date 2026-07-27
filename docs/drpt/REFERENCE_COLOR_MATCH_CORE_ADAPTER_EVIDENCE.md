# Reference Color Match Core Adapter Evidence

Date: 2026-07-28

Status: **P25B consumer contracts pass; no external core is integrated**.

## Delivered boundary

The additive `src/color_match/core_contracts.py` module defines four strict
Neuro-Film consumer records:

- `MatchViewV1`: dense read-only float32 RGB metadata with exact colour-state
  profile, pixel identity, render bridge and provenance identity;
- `TransformBundleV1`: an opaque payload binding to one source, one reference,
  one intent and one pinned producer/build;
- `DiagnosticsV1`: core execution facts and fail-closed status;
- `CapabilitiesV1`: the exact schemas, profiles, algorithms, features and
  determinism advertised by one pinned producer/build.

Four Draft 2020-12 JSON Schemas mirror those records under
`configs/schemas/reference_core_*.schema.json`. Identity-bearing records use
the existing language-neutral canonical byte stream, not JSON text hashes.

## Safety decisions

- v1 transform bundles are always `source-bound`;
- no `shared-operator` representation exists in v1;
- a producer/build, contract schema, algorithm, profile, capability,
  determinism, source view, reference view or diagnostics mismatch fails
  closed;
- only `ok` diagnostics may have no fallback reason, and `ok` requires finite
  output;
- profile names bind exact domain, primaries, white point and absolute/relative
  luminance semantics;
- current Neuro-Film relative-display profiles are not assumed compatible with
  D-PCT. A future pinned producer must advertise the exact profile;
- this contract does not promote pixels. Existing A1/A4/A5 and
  `reference-render-guard.v2` remain authoritative.

## Verification

- `20 passed` in `tests/test_color_match_core_contracts.py`;
- `54 passed` across new contracts plus existing recipe, schema, canonical and
  replay suites;
- `compileall` passes for `core_contracts.py` and the public package surface;
- all four new schemas pass `Draft202012Validator.check_schema`;
- JSON roundtrips preserve immutable tuple-backed Python records while wire
  payloads use JSON arrays;
- negative tests exercise unknown fields, canonical identity, colour-profile
  semantics, absolute luminance, source/reference binding, producer/build,
  algorithm, capability requirements and diagnostics identity.

## Peer boundary

D-PCT stable HEAD `4b71a8a` changes no shared producer schema. Its PST50 sRGB
development run reports identity mean E00 `10.102`, D-PCT `11.951`,
D-PCT+chroma.25 `11.901` and statistical `13.144`; the frozen candidate wins
only 23/50 against identity. Those results directly support keeping the
Neuro-Film cross-content gates and identity default, but they are not imported
as a product dependency.

## Not yet delivered

- no external process, package, dynamic library or C ABI call;
- no D-PCT producer schema or conformance fixture;
- no RAW/HDR/video bridge;
- no shared cross-content transform;
- no algorithm promotion, FilmFX change, main-branch merge or push.

## P25C WorkingImage adapter

`src/color_match/core_adapter.py` now prepares supported product images without
claiming a D-PCT canonical rail:

- `linear_srgb/display_linear` maps exactly to
  `neuro-film.display-relative-linear-srgb-d65.v1`;
- `linear_rec2020/display_linear` maps exactly to
  `neuro-film.display-relative-linear-rec2020-d65.v1`;
- the adapter copies pixels into an isolated, C-contiguous, read-only float32
  buffer;
- pixel identity is SHA-256 over row-major IEEE-754 binary32 network-order
  bits;
- provenance identity binds source profile, transfer states, HDR metadata,
  orientation, alpha policy, bit depth and warnings, but not a local path;
- scene-linear, display-referred/unknown, unsupported working space,
  non-absent alpha and unapplied orientation fail closed;
- producer invocation remains forbidden unless pinned capabilities advertise
  the exact profile, algorithm, contract schema and required feature flags.

Verification:

- `36 passed` across dedicated core contract and adapter suites;
- `123 passed` across adapter, core contracts, existing recipe/schema/canonical
  replay/file handling and preprocess/colour/output encoding;
- input pixels are not mutated or shared with the prepared buffer;
- non-contiguous input is normalized to exact dense strides;
- buffer mutation, descriptor drift and capability drift are detected.

This adapter does not convert relative SDR to D-PCT's scene-relative ACEScg or
display-absolute XYZ rails. Such conversion still requires a separately
versioned trusted render bridge.

## P25D1 product intake decision

`src/color_match/core_acceptance.py` binds one internally valid core execution
to the existing Neuro-Film promotion state:

```text
valid source/reference/transform/capabilities/diagnostics
  + A1/A4/A5 PromotionDecision
  -> identity-fallback
     or candidate-for-product-guard
```

`candidate-for-product-guard` is deliberately not `applied`. The candidate
must still pass Neuro-Film's delivered-pixel guard and transactional product
path. A non-`ok` core status always falls back to identity, including under the
explicit research override. An unpromoted algorithm also falls back unless the
override is explicit and recorded.

Evidence:

- `48 passed` across acceptance, adapter and core contract suites;
- strict Draft 2020-12 acceptance schema and canonical `decision_id`;
- A1/A4/A5 are an exact required tuple;
- promoted, rejected, research-override, unsupported, invalid, fallback,
  unknown-field and internally inconsistent decisions are covered.

## P25D2 synthetic consumer conformance

The frozen
`configs/reference_match_core_consumer_conformance_v1.json` bundle exercises
the consumer adapter without claiming a D-PCT producer implementation:

- two 2x2 extended-float vectors cover relative display-linear sRGB and
  Rec.2020, including negative and above-one samples;
- pixels are encoded as exact IEEE-754 binary32 big-endian bits;
- expected `MatchViewV1` payloads bind pixel, colour-profile, provenance and
  descriptor identities;
- one explicitly synthetic `CapabilitiesV1` advertises only the fixture
  algorithm/schema and both exact Neuro-Film profiles;
- repeated verification and result serialization are byte-identical;
- path changes cannot affect the prepared view;
- tampered identity and unknown fields fail structurally, while a validly
  rehashed descriptor/capability mismatch produces a failed case.

The fixture text contains neither `D-PCT` nor `Zhuise`. Passing it establishes
only the Neuro-Film consumer implementation. A real producer must publish and
pass its own fixed conformance bundle before compatibility can be claimed.

Verification: `9 passed` dedicated; `57 passed` across all external-core
contract, adapter, acceptance and conformance tests.

## P25 integration preflight

Against Neuro-Film main `60617f9`, this consumer branch has zero changed-path
overlap from common base `c03c321`. Merge tree
`24ff7653d1ced56c7de6cb82a2204770517e179c` is conflict-free, and its detached
synthetic merge passes all 282 selected reference-match, preprocess,
colour-engine and output-encoding tests.

The branch itself passes the same 282 tests. Full collection reports
`1138 passed, 1 skipped, 36 failed`; all 36 belong to the already classified
missing ignored-output/CRLF asset-hash families, and no colour-match test
fails. The temporary integration worktree was removed after verification.

This is consumer integration evidence, not authorization or evidence to merge
into main. It also does not establish conformance by D-PCT or any mobile/native
port.

## P26B exact external-output receipt

`src/color_match/core_apply_receipt.py` copies a future returned buffer into
consumer-owned storage and binds:

- exact row-major IEEE-754 binary32 output bytes and `MatchViewV1`;
- source, reference and source-bound transform identities;
- exact producer capability identity;
- the canonical hash of complete diagnostics;
- consumer output provenance and a same-profile/same-shape contract.

The copied buffer is finite, dense, C-contiguous, read-only float32 RGB.
Negative and above-one extended values are preserved. A non-ok diagnostic,
shape/profile mismatch, changed diagnostics, swapped execution identity,
mutated pixels, non-finite nested JSON, unknown field or `applied` state fails
closed.

The receipt remains `candidate-only`. It neither defines D-PCT's planned
producer `ApplyResult` nor enters the delivered-pixel path. Verification:
11 dedicated and 68 combined external-core tests pass.

## P26C receipt-bound candidate admission

`CoreCandidateAdmissionV2` combines the exact `CoreApplyReceiptV1.receipt_id`
with the frozen `CoreAcceptanceDecisionV1.decision_id`. It additionally binds
the transform and output-view identities. Accepted candidates are only
`pending-product-guard`; rejected candidates remain `identity-fallback`.
Neither the Python contract nor its strict schema contains an `applied` state.

Admission revalidates the prepared pixel bytes, full receipt, execution
binding and A1/A4/A5 decision. Pixel mutation, receipt replacement, decision
replacement, state contradiction, gate removal and unknown fields fail
closed. Verification: 11 dedicated and 79 combined external-core tests pass.

P26 broad verification passes 304/304 adjacent tests. Full collection reports
`1160 passed, 1 skipped, 36 failed`; all failures remain in the previously
classified missing ignored-output/CRLF asset families. Main path overlap
remains zero. P26 therefore closes the consumer-side output identity gap while
leaving producer compatibility and delivered-pixel integration closed.
