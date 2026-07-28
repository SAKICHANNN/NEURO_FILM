# Reference sRGB ICC Portability Evidence

Date: 2026-07-28

Status: **Windows host runtime complete for the exact profile accessor;
Android link-only and Apple object-only evidence complete; target-device
runtime and ICC application remain open**.

## Scope

P78 exposes the P76 588-byte sRGB ICC conformance asset through a
freestanding C11 ABI:

```c
size_t nf_srgb_icc_profile_size_v1(void);
const char *nf_srgb_icc_profile_sha256_v1(void);
int nf_srgb_icc_profile_copy_v1(uint8_t *output, size_t capacity);
```

The copy entry point checks all preconditions before writing. A null output or
capacity below 588 returns zero and leaves caller memory unchanged. Success
copies exactly 588 bytes and returns one. The ABI does not parse an image,
apply an ICC transform, authorize delivery or execute a colour-match
algorithm.

## P78 frozen identities

| Artifact | SHA-256 |
|---|---|
| exact ICC payload | `217fe48ec958c667f8eef725aa27198f465df95d7662593b90d0a1cc30114356` |
| generated C source | `f11e749525d4a6c702126706e4f9929f56a12242034cb71e0a59b8a0a440d80d` |
| public C header | `7d3ae0e07f2dd79762d1bc3fef3916667a98f05daeb9861a0984c605ec98885d` |
| native verifier | `41857de59a73de2811b9ccd839450835a6debfc4fd4927a5bda1a2098591d26a` |
| independent canonical SHA core | `62c35870b86c0dab26e9d9fa2c5034a3a726b6b610a7c0baf781b311222a9fe4` |

The generator reconstructs the committed C/header bytes from the validated
P76 Python asset, and a test requires byte equality. The native verifier
copies through the ABI, independently hashes the returned bytes, verifies
ICC header fields and exercises null/short-buffer rejection.

These source/build identities describe P78 commit `983810a`. P80 commit
`b550d8e` supersedes the generated C source with a volatile destination loop
to prevent an optimizing compiler from introducing an unbound `memcpy` at a
no-runtime-library DLL boundary. The profile bytes, public header and ABI
symbols do not change.

## Platform evidence

| Target | Evidence | Artifact SHA-256 | Claim ceiling |
|---|---|---|---|
| Windows x86_64 MSVC | compiled and executed twice, byte-identical | `e7a33530ead4fcaa546bd89ee0ab563fd200c2f11be0b82d28b61e79241d8e4e` | same Windows host runtime |
| Windows x86_64 LLVM-MinGW 22.1.8 | independently compiled and executed twice, byte-identical | `5abb5e9d7f3a7c079092d4d681e3cbe8ca24ff3a8c434bac3e876ec29679bd7a` | second compiler, same host |
| Android arm64-v8a NDK r27d | shared library linked twice; exact three ABI exports | `a22398278b5703298e443009cf26c9a4f325c2165054e08d4ffe0999dda37169` | compile/link only |
| Android x86_64 NDK r27d | shared library linked twice; exact three ABI exports | `8e4d4b3a69e14e0753580165639f9fd44db2d0b3714e10becb3b9fbdc31ac85c` | compile/link only |
| macOS 13 arm64 | Mach-O relocatable object twice; exact three external definitions | `3846352ed18cafa1f335f5133a6f313af07d6509076e25869370f67c20baf6e4` | object only, no SDK link/run |
| iOS 15 arm64 | Mach-O relocatable object twice; exact three external definitions | `f9e6b6d34ade50d625d956179ce085d7f2d6b5d3e3e55c0f077d1e1ec922fc5c` | object only, no SDK link/run |

All five focused P78 tests and 27 adjacent ICC/product-chain portability tests
pass. The first Apple attempt failed because the generated source included
`string.h`; the final accessor uses an explicit byte loop and is genuinely
header-only/freestanding at the object boundary.

## Explicit non-claims

- Android libraries were not loaded or executed on a device or emulator.
- Apple objects were not SDK-linked, loaded, signed or run.
- No JNI, Swift, Objective-C, Kotlin, app packaging, image I/O, performance,
  thermal or sandbox behavior is evidenced.
- Exact ICC bytes do not prove a renderer applied that profile.
- P78 creates no D-PCT successor, A1/A4/A5 promotion, RAW/HDR/video mapping,
  product application or delivery authority.

## P80 Windows dynamic ABI evidence

P80 builds the exact same three-symbol ABI as a DLL with both compilers,
checks the complete export table, loads each DLL through an independent
foreign-function caller and invokes every function. The caller verifies:

- `size_v1()` returns exactly 588;
- `sha256_v1()` returns the exact P76 profile identity;
- null and 587-byte-capacity calls reject before changing a sentinel buffer;
- a successful copy independently hashes to the P76 profile identity.

| Toolchain | Reproducible DLL SHA-256 | Runtime fact |
|---|---|---|
| MSVC 19.50 x86_64 | `5213657c499a099d2ff3e7570f75d38769e68fcf5c8ffbfb8b7524066d6c06b8` | loaded and invoked through `ctypes` |
| LLVM-MinGW 22.1.8 x86_64 | `3a6d92670f990a90fdf9c551884ad96e6c0d0096a0c53ba9dda436f2af803e2f` | independently loaded and invoked through `ctypes` |

The P80 generated C source SHA-256 is
`5c8c1c7ab06e59c613a96f7982130f6928af6402a41f29fe6d9d76814c221d1d`.
Both compilers produce byte-identical DLLs across two separate build
directories. This closes Windows dynamic invocation for these profile bytes
only; it does not add application integration, image decoding or a target
device claim.
