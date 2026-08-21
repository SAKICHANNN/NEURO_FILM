# P96 — DNG camera-to-PCS portable C ABI contract

Status: frozen before native source implementation, compilation, or probe
execution.

## Question

Can the exact five P94 camera-to-D50 PCS matrices be executed by one small,
caller-buffer C11 ABI with strict finite-input validation, in-place parity,
failure atomicity, and cross-compiler numerical agreement?

This is a portable explicit-operator mechanics leaf. It is independent of the
failed P95 `dng_validate` cohort and does not compare image quality or rescue
any rejected DNG.

## Frozen parent and profiles

- P94 evidence SHA-256:
  `28dbe488dac052ca374d35b883d9eaec93d3f2adb63223bc04ab87290c4ba355`.
- P94 raw report SHA-256:
  `f5a4aeb31070efcc7bddd9ff485dccadaa6b04e95d0251483c4bc1c24a1ae216`.
- All five exact P94 matrices and source identities are mandatory.
- Matrix values are reconstructed by the exact P94 implementation from the
  bound DNG metadata; no raster or P95 output is read.

## Frozen ABI

The sole transform accepts one row-major float64 `3x3` matrix and `N`
interleaved float64 camera triplets, and emits `matrix @ camera` triplets.

- ABI version is 1.
- `N` must be positive and all matrix/input values finite.
- The matrix determinant magnitude must be at least `1e-12`.
- Input/output may be the same pointer; any other memory overlap is rejected.
- Invalid arguments, invalid matrices, non-finite input, undersized output, or
  non-finite computed output leave all caller output bytes unchanged.
- No clipping, tone mapping, chromatic adaptation, gamut mapping, allocation,
  global state, file I/O, or thread creation is allowed.

## Frozen probes and toolchains

Exactly 257 camera triplets are generated once from integer arithmetic and
converted to float64. They include black, unit axes, neutral values, extended
headroom, and dense asymmetric colours. The same probes and matrices are used
for every build.

Two independent builds per compiler are required:

- local MSVC x64 C11 `/O2 /fp:strict /W4 /WX /Brepro`;
- pinned repository-relative LLVM-MinGW clang x64 C11 `-O2 -ffp-model=strict
  -Wall -Wextra -Werror`.

## Frozen gates

- five profiles and 1,285 transformed triplets;
- Python/native maximum absolute error `<=5e-13` for both compilers;
- MSVC/LLVM maximum absolute error `<=5e-13`;
- repeat execution byte exact within each loaded binary;
- in-place output byte exact to out-of-place output;
- two independent same-compiler DLL builds byte exact;
- non-finite matrix, singular matrix, non-finite input, undersized output, and
  partial-overlap output are failure-atomic;
- forward/reverse profile enumeration and two fresh reports byte exact;
- source DNG sample/RGB reads remain zero.

Any failed build, profile, probe, replay, atomicity, or binding gate closes the
exact ABI. There is no compiler flag, tolerance, profile, probe, precision, or
platform rescue.

## Claim ceiling

PASS means only private Windows x64 arithmetic conformance for the exact five
P94 camera-to-D50 PCS matrices under MSVC and LLVM-MinGW. It does not establish
arbitrary DNG support, raster decode, sensor/IDT calibration, colorimetric or
photographic quality, chromatic adaptation, display rendering, mobile/Apple
runtime, project-loader integration, public package/schema/capability,
product, film, stock, or preference admission.

This product includes DNG technology under license by Adobe.
