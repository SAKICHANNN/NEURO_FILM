# P93 DNG standard-default receipt contract

Date frozen: 2026-08-21  
Node: `ULT > U1 > U1.3F / P93`  
Status: frozen before implementation and before re-inspecting the five U1.3D files

## Direction and question

P93 is a bounded RAW/DNG engineering leaf. It does not search for another
after-only colour model and does not change raster decoding or rendering.

U1.3D remains a formal structural failure under the exact U1.3C v1 receipt:
the first file in each enumeration direction omitted `CFAPlaneColor` and
`CFALayout`. P93 asks whether a separately versioned receipt can represent the
two omissions without guessing by applying only defaults stated explicitly in
the official DNG 1.7.1.0 specification, while preserving fact provenance and
the complete U1.3D LibRaw comparison gates.

## Primary standard binding

The authoritative document is Adobe's official *Digital Negative (DNG)
Specification 1.7.1.0*, September 2023, stored through the repository-relative
P-backed data junction as:

`data/external/adobe_dng_spec_1_7_1_0/DNG_Spec_1_7_1_0.pdf`

Exact SHA-256:
`abdecfd8e104e8b86cc054d3a40b677bb2ebe87131bfde080a1df3ee16eb2f8d`.

The specification defines:

- `CFAPlaneColor` / tag 50710 / BYTE / default `[0, 1, 2]`; and
- `CFALayout` / tag 50711 / SHORT / count 1 / default `1`.

No other missing TIFF/DNG field may be synthesized by P93.

## Version and provenance rules

- U1.3C `neuro_film.dng_capture_metadata_receipt.v1` and its public Python
  entry point remain byte-compatible and retain their fail-closed behavior.
- P93 adds a distinct v2 receipt entry point and schema.
- Every v2 fact records `value_origin` as either `explicit_ifd_tag` or
  `dng_standard_default_1_7_1_0`.
- An explicit tag always wins and is never normalized to a default.
- A standard default has an empty `ifd_paths` list and records the exact
  specification binding. It must never be represented as if bytes existed in
  the source file.
- Defaults are permitted only for a selected CFA raw IFD. LinearRaw behavior
  remains unchanged.
- Missing required facts other than these exact two fields fail closed.
- Raster sample reads, TIFF `asarray`, raw mosaics, LibRaw `postprocess`, RGB
  decode and renderer calls remain forbidden.

## Roles

The exact five U1.3D rows are already consumed engineering evidence and all are
mandatory. No replacement or fresh-science claim is permitted:

1. `motorola_moto_g_7_play`
2. `lg_lg_h850`
3. `autel_robotics_xb015`
4. `xiaomi_m2010j19cg`
5. `blackmagic_pocket_cinema_camera_4k`

The four U1.3C devices are regression-only: their v1 receipt bytes must remain
unchanged. They do not decide the P93 conformance result.

## Fixed execution and gates

Before interpreting results:

1. validate the contract, implementation, comparison core, raw inspector,
   source-manifest and official-spec hashes;
2. build v2 receipts for all five mandatory rows in forward and reverse order;
3. require exact source hash/size and source-manifest rights bindings;
4. require any synthesized fact to be only tag 50710 or 50711 with the exact
   values, TIFF types and provenance above;
5. run the unchanged U1.3D comparisons for raw geometry, uniform WhiteLevel,
   mappable BlackLevel, AsShotNeutral-derived camera white balance, explained
   visible geometry, finite values and zero warnings;
6. require zero pixel/RGB decode calls; and
7. require byte-exact normalized forward/reverse reports and a second fresh
   process replay.

The unchanged numeric gates are BlackLevel maximum absolute error `<= 1` code,
WhiteLevel error `0`, camera-WB maximum relative error `<= 1e-5`, five of five
rows, and zero metadata warnings.

Any implementation, binding, provenance, row, replay or numeric failure closes
P93 without changing defaults, LibRaw parameters, tolerances, source rows or
crop interpretation.

## Claim ceiling

A pass establishes only a private, source-bound v2 metadata receipt and exact
DNG-to-LibRaw conformance for these five consumed devices under two official
standard defaults. It does not reopen or rewrite U1.3D/U1.3C, establish
arbitrary DNG support, decode or render pixels, infer sensor calibration,
validate a camera IDT/noise profile, change the default loader, or open a public
schema, capability, package, product, film or stock claim.
