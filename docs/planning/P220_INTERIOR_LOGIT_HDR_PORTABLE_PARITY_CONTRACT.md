# P220 — private interior-logit HDR portable-parity contract

Status: frozen after producer R1CG evidence review and before consumer native
implementation or execution.

## Question

Can a freestanding caller-buffer C11 primitive reproduce the exact frozen
`zhuise.interior-logit-quantile-transport` v1 apply semantics closely enough
for portable private execution, while preserving exact input boundaries,
never creating a new boundary from a strict-interior sample, and failing
atomically on malformed payloads or inputs?

This is an engineering handoff from a genuinely paired same-file UltraHDR
observation. It does not repeat R1CG's scientific scoring, fit a new bundle,
read an application target, or admit a consumer/product feature.

## Frozen authority

- producer R1CG evidence commit `b57fd41ea7010354403734698636f714afca0fcf`;
- R1CG implementation commit `0a8d82c0e42667a338473b4852e83154629f4ce6`;
- producer operator SHA-256
  `6a84416c42047062ac7c6b4adab400d5c8317ae8aced7532c31a15cf749fc3f5`;
- producer report SHA-256
  `1622aa050602e876effd4f9c2e84cfdd0adca10268c8e05aa35fcda00b47b7ff`;
- producer stable identity
  `d124e6b3b6fe66b56720ad54a59c7c9bf3be728953f55e620c46c8712297596c`;
- producer evidence SHA-256
  `b0b175a30c9b447f3ce67d0c4bb240c9c50d45e68207b480438dc0b3655e6fa7`.

R1CG passed 18/18 frozen gates on four fresh effects and sixteen application
rows. All applications improved, median/worst improvement was
`65.5299%/45.4404%`, and the candidate created zero new boundaries versus
`1.35295%` for the unchanged D-PCT control. These facts motivate parity only;
they are not re-adjudicated here.

## Frozen ABI and arithmetic

The private C11 ABI accepts:

- exactly 33 row-major RGB float32 source-logit knots;
- exactly 33 row-major RGB float32 reference-logit knots;
- interleaved float32 RGB values in `[0, 10000]` cd/m2;
- an explicit identity flag and caller-owned output buffer.

It performs the R1CG v1 apply only: input normalization by `10000`, clamping
for logit evaluation to `[1/65536, 1-1/65536]`, per-channel piecewise-linear
transport, extrapolation slope clamped to `[0,1]`, stable sigmoid, and scaling
by `10000`. Existing exact zero/10000 samples are restored exactly. Strict
interior samples must remain strict interior. Identity copies input bytes.

The implementation must not fit knots, decode UltraHDR, allocate heap memory,
clip a non-boundary result, or expose a public API/package/schema/capability.

## Frozen probes and gates

The independent consumer oracle uses four exact producer R1CG payloads and a
257-triplet probe containing exact boundaries, adjacent float32 values, every
source knot mapped back to luminance, deterministic logarithmic/interior
samples, and extrapolation samples.

- both MSVC and pinned LLVM-MinGW compile with warnings as errors;
- both compilers execute all four payloads twice and in reverse order;
- maximum absolute error versus the float32-order consumer oracle is at most
  `0.003` cd/m2 and maximum relative error is at most `3e-6`;
- exact zero and 10000 inputs remain byte-exact boundaries;
- strict-interior inputs never become zero or 10000;
- identity is byte exact and in-place execution equals out-of-place execution;
- malformed/non-increasing knots, non-finite/out-of-range input, invalid
  identity flag, undersized output, and partial overlap fail before any output
  byte or diagnostics byte changes;
- forward/reverse scientific reports are byte exact after excluding only
  compiler/runtime timing and temporary paths.

Any failed gate closes this exact ABI. Do not relax tolerance, change libm,
drop a payload/probe/compiler, add clipping, or reinterpret a failure as an
R1CG scientific failure.

## Claim ceiling

PASS means private Windows x64 arithmetic parity for four frozen R1CG
interior-logit payloads only. It is not captured-HDR truth, arbitrary UltraHDR
or gain-map support, after-only inference, portable target-runtime evidence,
public package/schema/capability, consumer admission, product admission,
film-stock evidence, preference, or photographic-quality evidence.
