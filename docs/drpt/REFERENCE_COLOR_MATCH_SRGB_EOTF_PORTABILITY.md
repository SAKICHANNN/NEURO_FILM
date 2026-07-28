# Reference sRGB EOTF Portability Evidence

Date: 2026-07-28

Status: **exact exhaustive Windows runtime parity complete; Android link-only
and Apple object-only evidence complete; device/app runtime remains open**.

## Purpose and ownership

P82 makes the consumer-owned P69/P72 IEC 61966-2-1 float32 decode operation
portable without implementing media I/O or a D-PCT algorithm. It accepts
already decoded native-endian uint8 or uint16 samples and emits display-linear
float32 samples.

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
generates a freestanding C11 lookup implementation.

The concatenated big-endian uint8-table plus uint16-table identity is
`1b8f915b4ddf4dc2b4aa961934b05549d23f4593a7b65c2aacfdb7eaea80a1ca`.

| Artifact | SHA-256 |
|---|---|
| generated C source | `0835a1a6c948a61d26bfd656288e757565892604927a7ba7149dda2f7023ba35` |
| public C header | `74884f0ca720f17253655d1fbf3dea0e224a516fcd8a6e83a33b875f0a00dd5a` |
| MSVC x86_64 DLL | `f01d82bb708b7e15316ed2528e9042f90636bed58dcd513c837d6dfc4ccb086a` |
| LLVM-MinGW x86_64 DLL | `67d740d75eb0eb72a71f3911fa6738eea512037bb8a17fb51fbbcb1bf1196d0e` |

Both DLLs are reproducible across separate build directories, export exactly
the two declared symbols, load through an independent FFI caller and match
the Python implementation bit-for-bit for every uint8 and uint16 code.

## Failure boundary

Before writing any output, the C ABI rejects:

- null input or output;
- zero sample count;
- bit depth other than 8 or 16;
- output capacity below sample count;
- sample/output byte-count integer overflow;
- any overlapping input/output ranges.

Sentinel-buffer tests cover invalid depth, zero count, null input/output,
short capacity, maximum `size_t` count and direct overlap. All remain
byte-unchanged.

## Cross-target evidence

| Target | Artifact SHA-256 | Claim ceiling |
|---|---|---|
| Android arm64-v8a NDK r27d | `82585ceed311c819d42fef504b28f6e4b9c461296b67271ab838b5850deca944` | shared-library compile/link only |
| Android x86_64 NDK r27d | `814e40636c3ba3bd7ff5fe1f5678fbb04ca0b562bddb6529fb0796e43f62756a` | shared-library compile/link only |
| macOS 13 arm64 | `095d8f71d4edced6e37156dad06f4e661d8a3bb8dda5658c92c00356cf1e4d15` | Mach-O object only |
| iOS 15 arm64 | `c8324316add36c0298e5280da0183c6a6d28a380409a728336c8da6483b41455` | Mach-O object only |

Android libraries export exactly the two ABI symbols. Apple objects define
exactly those symbols. Each build repeats byte-identically.

## Explicit non-claims

- Android and Apple outputs were not loaded or executed.
- No SDK app, JNI, Kotlin, Swift, Objective-C, image I/O, ICC application,
  performance, memory-pressure or sandbox behavior is evidenced.
- P82 does not decode PNG/JPEG/TIFF, map RAW/HDR/video, fit/apply a D-PCT
  transform, authorize application/delivery or change A1/A4/A5.
- The lookup table is a consumer conformance implementation of P72, not a new
  photographic algorithm.
