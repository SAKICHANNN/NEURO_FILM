# U1.6G4D Halation Capability-Binding Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4D`

**Decision:** pass static provider binding; graph execution remains absent

## Reproducibility

- implementation commit: `973b82c9c778bbab17c90190e0483a555afe1ae2`;
- audit commit: `6197ad225e8e1d52ca87f856dd0e3095711f14ee`;
- config SHA-256: `b641d9d7373a4e7b1556622f79a092184a3ff305be0d062d21ae3d84504ec74b`;
- ignored reports: `outputs/u1_6g4d/formal_6197ad2_{a,b}.json`;
- report SHA-256: `7fd7afcb0b6c2c083541f94be15494d72a3df02ea2083a0e66dd1606db50c050`;
- two executions produced byte-identical reports.

## Result

The resolver validates and returns exactly:

- `coordinate_exact_gradient_window` ->
  `coordinate-gradient-window-v1` / `coordinate_gradient_window`;
- `row_chunked_global_stage_builder` ->
  `shape-stable-global-resample-v2-explicit-f32` /
  `build_chunk_invariant_global_stage_from_rows`.

Both `physical-colour-v1` and `density-v1` plans are statically ready at
`1024x1536`/tile 256 and `4000x6000`/tile 512. Every plan has:

- `integration_ready=true`;
- no missing capabilities;
- no unresolved global workspace nodes;
- the same nodes, lifetimes, blur/percentile inventory, finite/global
  classification and resource fields as its unbound G3 plan.

The bound fingerprint differs only because capability state is intentionally
part of the fingerprint. Default empty and gradient-only behavior remains
closed and unchanged.

## Verification

- 80 focused G3/G4A/G4C tests pass;
- 511 complete CPU tests pass in 21.34 seconds;
- incomplete, duplicate and forged-version binding records fail closed;
- current effects, renderer, profiles, recipes and CLI remain unchanged.

## Decision and next branch

G3 prerequisites are now **statically ready from real providers**. This is not
an executable effect or product integration result. Open U1.6G4E to freeze the
smallest staged `density-v1` executor, with explicit operator-version drift,
effect parity, resource, full-resolution visual/severe-artifact and failure
gates before implementation.

## Claim ceiling

This proves static provider binding/readiness only. It does not prove graph
execution, effect parity, visual quality, physical accuracy, renderer
integration, total memory, streaming decode or 100MP readiness.
