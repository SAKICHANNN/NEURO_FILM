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
