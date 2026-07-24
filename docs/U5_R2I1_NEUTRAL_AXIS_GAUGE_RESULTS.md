# U5.R2I1 Neutral-Axis Gauge Results

**Decision:** numerical pass; real-image frontier opens  
**Date:** 2026-07-24  
**Report SHA-256:** `c3746f59edd9f242c32b6d83ac5ef0f3c2258367317474096d081de2d435b00d`

The ungauged U2.2B neutral response has maximum channel spread `0.1630335`.
The frozen 1,025-knot inverse-neutral gauge reduces dense-grid channel spread
to `6.4059e-6` and maximum absolute neutral error to `1.8196e-4`. Black and
white endpoints are exact.

The result is not a look collapse: identity RMSE remains `0.13174`, and the
best-affine residual remains `0.05493`. Random outputs remain inside the cube,
the minimum sampled composition Jacobian determinant is `0.01480`, and
partition/replay/source-preservation and fail-closed domain gates pass. Two
formal runs have identical pre-provenance array/report hashes; all 811 CPU
tests pass.

The construction is entirely data-independent: its only samples are neutral
inputs evaluated through the immutable U2.2B operator. It reads no photograph,
film scan, metadata, label or visual result.

This opens a separately frozen real-image strength frontier. It does not prove
visual appeal, severe safety, stock response, calibration or product value and
does not alter current fitting/training/LSM stops or production defaults.
