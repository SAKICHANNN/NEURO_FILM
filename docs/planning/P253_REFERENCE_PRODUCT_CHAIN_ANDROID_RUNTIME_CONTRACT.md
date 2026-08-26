# P253 Reference Product-Chain Android Runtime Contract

## Scope

P253 closes one bounded platform gap left by P31C: execute the unchanged
native product-chain identity and staging verifier on an Android 14 x86_64
virtual device.  It does not change the canonical encoding, SHA-256 core,
staging policy, producer evidence, renderer, or product defaults.

This is a private target-runtime conformance leaf.  The arm64-v8a artifact is
rebuilt and link-checked only.  No claim is made for physical arm64 hardware,
JNI, an Android application, media I/O, performance, thermal behavior,
scientific quality, public packaging, or product admission.

## Frozen inputs

- `native/reference_product_chain_conformance.cpp`
- `native/reference_canonical_core.c`
- `native/reference_canonical_core.h`
- `tests/fixtures/reference_product_chain_conformance_v1.json`
- `scripts/build_reference_chain_android.py`
- `configs/reference_match_android_ndk_r27d.json`
- Android NDK r27d, revision `27.3.13750724`
- Android emulator 36.6.11.0 and API 34 Google APIs x86_64 system image

Exact byte identities are frozen in
`configs/p253_reference_product_chain_android_runtime_v1.json`.  Tool paths
remain runtime arguments so the project does not encode a physical drive.
The AVD home, build roots, logs, and reports must remain under the
repository-relative `outputs/tmp/p253_*` namespace.

## Execution

1. Build arm64-v8a and x86_64 artifacts twice in disjoint build roots.  Both
   artifact hashes must match across builds.
2. Create an owned API-34 x86_64 AVD on port 5594 from the locked minimal
   config in the runner.  The migrated SDK's system image and package index
   are complete and hash-locked, but its optional `devices.xml` catalogue is
   absent; a pre-build infrastructure probe showed that `avdmanager` therefore
   cannot materialize a named device.  The correction does not reuse or modify
   another project's AVD and does not alter the system image.
3. Run two complete committed-head processes: normal and reverse fixture
   enumeration.  Each process starts with `-wipe-data`, executes every hash
   and staging case as a fresh Android process, and publishes a canonical
   report independent of enumeration order.
4. After each process, stop only the owned serial/AVD and prove that no owned
   emulator or QEMU process survives.

## Frozen gates

- NDK, source, fixture, emulator, adb, avdmanager, and system-image package
  identities match the configuration.
- Both Android ABI builds are deterministic; x86_64 executes on Android 14
  and arm64-v8a remains link-only.
- Every fixture canonical byte string hashes to its frozen SHA-256 value.
- The complete eight-row staging truth table matches the unchanged core.
- Normal and reverse reports are byte-identical.
- Odd-length hex, non-lowercase hex, invalid state flags, and missing command
  arguments fail with no stdout payload.
- A one-nibble canonical-byte mutation changes the digest.
- All owned runtime processes and scratch files are removed after evidence is
  retained.

Any failed gate yields `FAIL_CLOSED_REFERENCE_PRODUCT_CHAIN_ANDROID_RUNTIME`.
There is no tolerance adjustment, fixture replacement, source patch, emulator
substitution, or same-leaf rescue.

## Claim ceiling

Passing permits only:

`PASS_PRIVATE_REFERENCE_PRODUCT_CHAIN_ANDROID14_X86_64_RUNTIME`

It means the exact native identity/staging verifier executed consistently on
one Windows-hosted Android 14 x86_64 virtual device.  It does not promote a
colour candidate, change A1/A4/A5 outcomes, authorize staging of any rejected
candidate, or establish a consumer package/capability/product claim.

## Prescore infrastructure correction

The first formal attempt stopped before build, boot, or fixture execution
because direct script import was not valid.  The next attempt also stopped
before build because `avdmanager` could not load the missing system-image
`devices.xml`.  The direct-entry import and owned minimal-AVD materialization
are additive infrastructure corrections.  All source bytes, fixture rows,
runtime image, tool identities, scientific gates, and claim ceiling remain
unchanged; formal execution restarts from the corrected commit.
