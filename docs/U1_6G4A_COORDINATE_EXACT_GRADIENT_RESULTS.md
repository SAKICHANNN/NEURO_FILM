# U1.6G4A Coordinate-Exact Gradient Window Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4A`

**Decision:** pass the gradient-window capability; keep effect integration
closed on row-chunked global staging.

## Reproducibility

- implementation commit: `efd125a875287e4d12f2f6c9db0106ec49de5450`;
- gate-propagation test commit: `fbcef7b4a73d43c1c57b45ebe918f2dbb0e08750`;
- primitive: `coordinate-gradient-window-v1`;
- config: `configs/u1_6g4a_coordinate_gradient_v1.json`;
- config SHA-256: `aa62b97dddc675c8f838e527cb123fc6ba841edc123d80a4271dd92de9918634`;
- field: deterministic normal float32 `257x389`, seed 223;
- reference: current NumPy unit-spacing default gradient (`edge_order=1`).

## Formal parity

Reference SHA-256 values:

- `gy`: `30e1ee9ff307304c09e315a74bd0aac39a397af73dbd0670e71bc8dc666c53a4`;
- `gx`: `46bc074050ed07890d0b3e8de8a549f74f653a23fc7b4425073f84d74084ae65`.

| Assembly | Windows | Max expanded shape | Max expanded input | gy/gx max error | Bytes equal |
|---|---:|---:|---:|---:|---|
| tile 37 | 77 | 39x39 | 6,084 | 0 / 0 | yes |
| tile 64 | 35 | 66x66 | 17,424 | 0 / 0 | yes |
| row 1 | 257 | 3x389 | 4,668 | 0 / 0 | yes |
| row 19 | 14 | 21x389 | 32,676 | 0 / 0 | yes |

Every assembled SHA-256 equals the corresponding full-frame reference. The
reader receives exactly the requested core plus at most one available original
pixel per side. Outputs are contiguous read-only float32.

## Verification

- 34 focused gradient tests passed;
- full/interior/edge/corner/single-row/single-column cases pass;
- invalid source axes, bounds, reader shape/dtype/finiteness and reader failures
  fail closed;
- G3 regression proves providing only
  `coordinate_exact_gradient_window` leaves exactly
  `row_chunked_global_stage_builder` missing and keeps integration false;
- 62 combined G3/G4A tests passed;
- 493 complete CPU tests passed in 20.48 seconds.

## Branch and claim ceiling

G4A removes the coordinate-gradient capability gap. It opens only row-chunked
construction of U1.6G1 global coarse stages from derived field windows. No
effect path changed and integration remains false. This result does not prove
halation pixel parity, physical realism, renderer integration, stock/calibrated
response, total-memory bounds or 100MP performance.
