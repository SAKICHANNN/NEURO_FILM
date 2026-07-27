# Reference Color Match Product-Chain Portable Conformance

Date: 2026-07-28  
Node: P31A-D  
Status: Windows exact execution and Android cross-link evidence

## Scope

The vector covers the consumer-owned P28 batch, P29 per-source decisions and
numeric batch, and P30 product staging authorization. It contains no producer
algorithm implementation, image pixels, media decoder or mutable cross-repo
import.

Two cases are frozen:

1. two genuinely promoted, non-research rows produce
   `authorized-for-staging`;
2. a research override passes P28 and relaxed P29 numeric eligibility but P30
   returns full `identity-fallback`.

Each case exposes five identity payloads: P28 batch, two P29 decisions, P29
batch and P30 authorization. The ten payloads are encoded as exact canonical
byte hex with expected SHA-256.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| golden fixture | `6cfb24bf7c1b2d4e9d8e550e1a7e4dea2cc58ff55d51bb5f8b309a0e77308fd9` |
| fixture schema | `a7864a71faaf9844f6ce3023951499b93b429e692b64e45dbffc9d5c28a8a95e` |
| C++17 verifier | `6e14ac0a1eb23c9b9494cc0fe6d1b973105d05624d52471b7ff014eba235a290` |
| consumer Android NDK pin | `d9267d0e364479cd15f49b250862c3d8769ae96e00c07c3e70f2cec0d46693ae` |
| consumer LLVM-MinGW pin | `f30ca2b5dc0bb90104d6ef911dc6fcdc649f64dc236c9d57b424a583ca1598c8` |

All canonical artifacts are forced to LF and regenerate exactly from the
corrected P27 producer fixture.

## Independent Windows execution

The standalone C++17 verifier implements its own SHA-256 and lowercase hex
decoder. It receives canonical bytes, not Python objects or precomputed
intermediate state.

- MSVC C++17 `/O2 /W4 /WX` compiles and links.
- All ten identities exactly match Python.
- The promoted and research-override state cases exactly match.
- Uppercase/noncanonical hex is rejected.

This is same-host independent-language evidence, not an independent algorithm
or product-quality result.

The consumer subsequently pinned a second compiler independently:
LLVM-MinGW 20260616, Clang 22.1.8,
`x86_64-w64-windows-gnu`. Its compiler and license hashes match the consumer
lock. A static build independently reproduces all ten identities and both
authorization states. This reduces compiler-specific risk but remains
same-host Windows x64 evidence.

## Android cross-compile

The consumer independently pins NDK r27d revision `27.3.13750724`, including
exact `source.properties` and `NOTICE` hashes. Clang 18 compiles and links:

| ABI | ELF machine | Bytes | SHA-256 |
|---|---|---:|---|
| arm64-v8a | AArch64 | 58,544 | `10977363b0d41443655a37657dc27682d8be06f2c73e5bdff3d9d17267432ffe` |
| x86_64 | Advanced Micro Devices X86-64 | 55,048 | `ed832df3f3a26abbf2b71fc9fd1a22aea0c8d4c36f30b964729ced0bb61af310` |

This proves compile/link and target architecture only. There is no device or
emulator execution, JNI/Kotlin packaging, image I/O, performance, thermal,
iOS or macOS evidence.

## Regression and propagation

- 83 combined P27-P31 tests pass.
- Full suite: 1236 passed, one skipped and the unchanged 36 environment
  failures; no colour-match/P31 failure.
- Latest main is `a8e5372`; common base is `c03c321`; path overlap is zero.
- Clean merge tree: `3117e3649da428452f0989f165411336cf7ba7e5`.
- Detached fresh-checkout merge `8c231adb...` passes all 83 tests, including
  MSVC and LLVM-MinGW execution plus Android arm64/x86_64 cross-linking.
- The temporary merge worktree was removed. Main's concurrent dirty files
  were observed read-only and were not included or changed.

## Current delivery boundary

No external algorithm has yet supplied both a real A1/A4/A5 promotion and a
frozen consumer invocation package. P31 therefore does not wire the synthetic
fixture into the existing file transaction or claim applied delivery.
