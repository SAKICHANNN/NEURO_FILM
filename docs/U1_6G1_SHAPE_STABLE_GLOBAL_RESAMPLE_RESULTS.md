# U1.6G1 Shape-Stable Global-Resample Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G1`

**Decision:** pass the independent numerical primitive; do not integrate an
effect yet.

## Reproducibility

- implementation commit: `fc8032c0df0ee2d6f35fbec4e8090ce33868acc3`;
- operator: `shape-stable-global-resample-v1`;
- config: `configs/u1_6g1_shape_stable_global_resample_v1.json`;
- config SHA-256: `0b02af4b01161640c881484f88eeb33bc29ff5cc6f239ab7fb8da1ba02ce7d6b`;
- deterministic NumPy PCG64 seed: `71`;
- source: synthetic finite float32 field, `257 x 389`;
- sigma: `52`, matching the decisive U1.6G0 geometry;
- tile sizes: `37` and `64`.

This mechanical source has no film, stock, rights or authenticity role.

## Frozen-case result

The original-image plan selected a `32 x 48` coarse field with vertical and
horizontal scale factors `8.03125` and `8.104166666666666`. The corresponding
coarse sigmas were `6.474708171206226` and `6.416452442159383`.

| Tile | Tiles | Max error | Seam max | Byte-identical | Output SHA-256 |
|---:|---:|---:|---:|---|---|
| 37 | 77 | 0 | 0 | yes | `c3a59c15ff785f1b2e64fa146cf6dda4f557fdab2d5e8026720660a9fa7d3c38` |
| 64 | 35 | 0 | 0 | yes | `c3a59c15ff785f1b2e64fa146cf6dda4f557fdab2d5e8026720660a9fa7d3c38` |

The read-only coarse field is 6,144 bytes. The required final float32 output is
399,892 bytes. Maximum reconstruction-window shapes were `37 x 37` and
`64 x 64`; no full-resolution reconstructed context is stored in the stage.
Observed smoke timings were approximately 1.3 ms for staging, 2.2 ms for one
full reconstruction, and 14-22 ms for tiled reconstruction. These local
timings are diagnostics, not performance gates or 100MP projections.

## Verification

- 45 focused global-resample/halation tests passed;
- 420 complete CPU tests passed in 20.73 seconds;
- irregular 2-D/multi-channel, anisotropic, identity, repeated-run, bounds,
  dtype, forged-plan and read-only-stage cases passed;
- existing `gaussian_filter_safe`, effects, renderer and CLI did not change.

## Decision and next branch

U1.6G1 passes as a reusable prerequisite because tile size no longer changes
the coarse grid or reconstructed pixels. U1.6 remains active. The next physical
halation work must still separately solve staged percentiles and field-DAG
lifetimes, then give colour and density families independent integration and
visual/severe-artifact gates.

No current effect is switched to this operator. The old numerical approximation
remains the compatibility path; simple halation remains independently eligible.

## Claim ceiling

This result proves exact full/tiled equality for the new versioned global-grid
primitive on the frozen mechanical cases. It does not prove legacy halation
parity, physical realism, calibrated response, stock authenticity, complete
renderer tiling, streaming input, bounded total memory or 100MP readiness.
