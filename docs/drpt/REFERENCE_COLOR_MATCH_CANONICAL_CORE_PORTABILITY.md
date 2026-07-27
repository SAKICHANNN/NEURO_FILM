# Reference Color Match Canonical Core Portability

Date: 2026-07-28  
Node: P42A-D  
Status: host exact execution, Android link and Apple object evidence complete

## Result

The existing P28-P30 C++ conformance runner now delegates canonical SHA-256
and the staging authorization predicate to one consumer-owned, freestanding C
ABI. The C core has no system-header, libc, file-I/O, allocation, exception or
platform-framework dependency.

The frozen ten canonical identities and two state outcomes did not change.
MSVC and pinned LLVM-MinGW execute them through the exact same C object. Eight
additional message lengths cover both SHA-256 padding boundaries and a
multi-block input against Python `hashlib`.

## Frozen source identities

| Artifact | SHA-256 |
|---|---|
| freestanding C core | `62c35870b86c0dab26e9d9fa2c5034a3a726b6b610a7c0baf781b311222a9fe4` |
| C ABI header | `edb6b7b95f93a167985b7d41d4861a0420dc39c43991ad971f88e59648f7097c` |
| hosted C++ runner | `d59bc3adadb3296f63123b455866367c6ba62da537aa55c6bc4640fd6d7540fa` |
| unchanged golden fixture | `6cfb24bf7c1b2d4e9d8e550e1a7e4dea2cc58ff55d51bb5f8b309a0e77308fd9` |
| Apple object build script | `395a9af1949b6b44d2152bde794feb0e60a36efa24b4e499e89c8e835fea013b` |

## Platform evidence

| Platform | Evidence | Artifact SHA-256 |
|---|---|---|
| Windows x64 MSVC | `/O2 /W4 /WX`; exact execution | temporary test artifact |
| Windows x64 LLVM-MinGW | Clang 22.1.8 static executable; exact execution | `147162169e624df6abdb661d8027d92e761212f2d788d380c7a424f68449cc83` |
| Android arm64-v8a | NDK r27d/Clang 18 compile and ELF link | `14deb9c14ecc6cdf3a19bd13e6a5d6e14a0f97177d71bfd273b7965a4314b476` |
| Android x86_64 | NDK r27d/Clang 18 compile and ELF link | `06d2dbed2f5f5657ea47cfec5e0f97f6365344bb117f092acaf1df352b1781b7` |
| macOS arm64 | Clang 22.1.8 freestanding Mach-O object | `f093f4fb7b36b41803f7ce81a39ccff204ef80bd420114c5aeaf7caae4e0e6a9` |
| iOS arm64 | Clang 22.1.8 freestanding Mach-O object | `5e3fe02a065ba324fe1f127600dc71b8d6b55b21131211d55328ed388f7cb91e` |

Generated Android and Apple reports remain ignored build evidence:

- Android report SHA-256:
  `ff43b48819b15e9cbfb0e9a0ea451fafbb46d8e5a5fe90f023d9f9409262278a`;
- Apple report SHA-256:
  `13bd7a9e9664ab4fe2c19347ae2d6b8bfc6a8513180077a220aefc23eaeb0090`.

## Claim limits

Android evidence is compile/link only: there is no emulator/device execution,
JNI/Kotlin integration, image I/O, performance or thermal evidence.

Apple evidence is narrower still. The pinned Clang distribution emitted
Mach-O ARM64 relocatable objects from the no-header core. No Apple SDK,
platform libc, link, load, simulator/device execution, Swift/Objective-C
bridge, signing, app packaging, image I/O or performance claim exists. A
hosted C++ Apple-target attempt failed at the non-Apple CRT headers, which is
why the evidence is intentionally limited to the freestanding core.

This core verifies canonical bytes and one consumer authorization predicate.
It is not D-PCT algorithm code, a producer ABI, a media engine or proof that a
candidate is visually acceptable, promoted, applied or delivered.

## Verification and propagation

- 16 dedicated native/cross-target tests pass.
- 433 combined color-match/FilmFX tests pass.
- Full suite: 1328 passed, one skipped and the unchanged 36 isolated-worktree
  output/asset failures; no color-match, FilmFX or P42 failure.
- Latest main: `bebd34fb037e9f21df536295d7390b7b7d08618f`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 194/135, with zero exact overlap.
- Clean merge tree: `f356c1462e5cdbacebe4f9138314f34902c7093b`.
- A fresh detached merge passes all 16 P42-focused tests; its temporary
  worktree was removed.

The producer repository was observed read-only at `eb4b889`; its newly
committed invocation package remains a separate P43 compatibility audit and
does not change any P42 claim.
