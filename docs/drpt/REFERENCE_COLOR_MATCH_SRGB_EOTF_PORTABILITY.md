# Reference sRGB EOTF Portability Evidence

Date: 2026-07-28

Status: **exact exhaustive Windows runtime parity complete; Android link-only
and Apple object-only evidence complete; device/app runtime remains open**.

## Purpose and ownership

P82 makes the consumer-owned P69/P72 IEC 61966-2-1 float32 decode operation
portable without implementing media I/O or a D-PCT algorithm. It accepts
already decoded native-endian uint8 or uint16 samples and emits display-linear
float32 samples. P84 hardens that same immutable ABI by requiring IEEE-754
binary32, verifying little-endian representation before output and rejecting
unaligned typed buffers.

The ABI is:

```c
const char *nf_srgb_eotf_f32_lut_sha256_v1(void);
int nf_srgb_eotf_f32_apply_v1(
    const void *samples,
    size_t sample_count,
    uint32_t bit_depth,
    float *output,
    size_t output_capacity);
```

## Exact arithmetic

Platform `powf` implementations are not used. The generator evaluates the
existing P72 Python/NumPy float32 contract for all 256 uint8 codes and all
65,536 uint16 codes, stores their exact IEEE-754 binary32 representations and
generates a freestanding C11 lookup implementation. The generated source
checks the compile-time binary radix, mantissa and exponent contract, verifies
the runtime byte representation of `1.0f`, and stores result bits through
character lvalues rather than implementation-defined union type-punning.

The concatenated big-endian uint8-table plus uint16-table identity is
`1b8f915b4ddf4dc2b4aa961934b05549d23f4593a7b65c2aacfdb7eaea80a1ca`.

| Artifact | SHA-256 |
|---|---|
| generated C source | `33a31fae5d2e6ef7d22778225fb3806fe70a2c6f60eff7ae2c9a393a83db579f` |
| public C header | `74884f0ca720f17253655d1fbf3dea0e224a516fcd8a6e83a33b875f0a00dd5a` |
| MSVC x86_64 DLL | `92f3e157d4097243b5521452fef3d9127bd3d292084d62be4773706f9e75fac8` |
| LLVM-MinGW x86_64 DLL | `62f05548ea167cf27c668b89b1bcc8fc7640202f030d9608027d09815667637f` |

Both DLLs are reproducible across separate build directories, export exactly
the two declared symbols, load through an independent FFI caller and match
the Python implementation bit-for-bit for every uint8 and uint16 code.

## Failure boundary

Before writing any output, the C ABI rejects:

- null input or output;
- zero sample count;
- bit depth other than 8 or 16;
- output capacity below sample count;
- a non-little-endian binary32 representation; unsupported binary32
  characteristics are a compile-time error instead;
- an output pointer not aligned for `float`;
- a 16-bit input pointer not aligned for `uint16_t`;
- sample/output byte-count integer overflow;
- any overlapping input/output ranges.

Sentinel-buffer tests cover invalid depth, zero count, null input/output,
short capacity, maximum `size_t` count, direct overlap and deliberately
unaligned input/output pointers. All remain byte-unchanged.

## Cross-target evidence

| Target | Artifact SHA-256 | Claim ceiling |
|---|---|---|
| Android arm64-v8a NDK r27d | `32ea03e079193a440f20d3480e5bc6c9d545cff661819514a35a2d31bf9b83a8` | shared-library compile/link only |
| Android x86_64 NDK r27d | `593e7c1b9663101894410e44f534f42454c3ff90f8bdebbe9fcbb8eb70aa25f3` | shared-library compile/link only |
| macOS 13 arm64 | `3d47f90b4c1f3ab3eb77bdb1d53d94cf13040bfc2e784896328dccd1b52e0f4e` | Mach-O object only |
| iOS 15 arm64 | `122e24bf7a52b98bb91b19e8faa3477d4f7fdcbe81d36beac3c495e2ca277400` | Mach-O object only |

Android libraries export exactly the two ABI symbols. Apple objects define
exactly those symbols. Each build repeats byte-identically.

## Explicit non-claims

- Android and Apple outputs were not loaded or executed.
- No SDK app, JNI, Kotlin, Swift, Objective-C, image I/O, ICC application,
  performance, memory-pressure or sandbox behavior is evidenced.
- P82/P84 do not decode PNG/JPEG/TIFF, map RAW/HDR/video, fit/apply a D-PCT
  transform, authorize application/delivery or change A1/A4/A5.
- The lookup table is a consumer conformance implementation of P72, not a new
  photographic algorithm.
