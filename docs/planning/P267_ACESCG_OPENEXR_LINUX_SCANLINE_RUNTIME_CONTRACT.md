# P267 — ACEScg OpenEXR Linux scanline runtime contract

Status: prospective freeze before implementation or Linux pixel execution.

## Question

Can the exact P248 official-source ordered-scanline mechanism execute the same
24 MP procedural ACEScg/AP1-D60 float32 ZIP OpenEXR workload under WSL2 Ubuntu
22.04 x86_64 while preserving decoded samples, metadata, negative/highlight
support, publication atomicity and bounded resources?

P267 changes the target operating system and the platform-specific atomic
publication primitive only. It does not change the P248 sample formula,
dimensions, 16-row block, ZIP compression or metadata.

## Frozen parents and inputs

- P248 evidence SHA-256:
  `88bb01452def2259a6f6b533ae849231b1f4bdf09e984de8dad100c9f974c27d`.
- P248 config SHA-256:
  `76579c30b46875d6a086308e56db004858cbd29ad3da077d9ba8e52e2ce58086`.
- P248 native source SHA-256:
  `4f2d9aba58cf80509c6c06a6a471aab608bbb5b9fdc4fcb00b91ba5dac97bc89`.
- OpenEXR 3.4.15 source archive: 25,840,011 bytes, SHA-256
  `ab893d8003773ccd9a5556b2caf38da591ae37e20b06ee9d589a08984c5191f2`.
- Imath 3.2.2 source archive: 689,217 bytes, SHA-256
  `b4275d83fb95521510e389b8d13af10298ed5bed1c8e13efd961d91b1105e462`.
- Repo-relative CMake 4.2.3 Linux wheel: 28,904,315 bytes, SHA-256
  `8e91b381aaea3c47110583dccc52f4562333d1accdbb806939f953c16e74ec0a`.
- WSL distribution: `Ubuntu-22.04`, architecture `x86_64`.
- Probe: 4000x6000x3 float32, period 1024, generation block 64,
  ordered write block 16, ZIP compression, exact P248 coordinate formula.

All source and build assets remain behind repo-relative paths. Formal execution
must not fetch packages or pixels.

## Implementation boundary

The new C++ source may replace only P248's Windows-only `MoveFileExW` publish
operation with a POSIX same-directory rename after the temporary OpenEXR has
closed. The destination parent must pre-exist. Invalid inputs and injected
pre-publish failure must leave no usable partial output. No change to pixel
generation, channel order, metadata or compression is permitted.

OpenEXR and Imath are built from the frozen archives in an owned temporary WSL
workspace. The frozen CMake wheel may be installed only into that workspace;
system packages and the WSL base image are not mutated.

## Formal roles and gates

Run two complete controllers in forward/reverse worker order. Each controller
must build from a fresh source/workspace and launch two fresh Linux workers.

Required gates:

1. every decoded float32 component equals the P248 formula exactly;
2. decoded pixel SHA equals P248
   `2b4890bd8fdc9af7caadccbdee56960300803a4ca7f5851fd00280ee0ea86adf`;
3. AP1/D60 chromaticities and adoptedNeutral are exact, scanline/ZIP/RGB-f32
   identities hold, values are finite, negative and above-one values survive,
   and new exact boundary count is zero;
4. each Linux worker uses at most 2 GiB process-tree RSS and 120 seconds wall;
5. output is at most 1 GiB;
6. invalid input rejects without output and injected failure preserves an
   existing foreign destination with zero temporary residue;
7. source archives, P248 parents and input bytes remain unchanged;
8. controller scientific payloads excluding resource measurements and ELF or
   container-byte identities are exact across forward/reverse order;
9. network requests during formal execution, external/project pixel reads and
   owned workspace residue are zero.

Linux ELF bytes and complete OpenEXR container bytes are diagnostics, not
cross-platform equality gates. A difference in either cannot override decoded
pixel, metadata, atomicity or resource gates.

## Stop rule and claim ceiling

Any frozen source, build, decoded-pixel, metadata, range, atomicity, resource,
order or cleanup failure closes this exact Linux mechanism. Do not change the
probe, block size, compression, compiler flags, thresholds or operating-system
target after formal observation.

A pass is private WSL2 Ubuntu 22.04 x86_64 24 MP OpenEXR scanline
runtime/resource evidence only. It is not native-device certification, natural
image quality, arbitrary OpenEXR support, AP0/ACES container admission, a
public dependency/API/package/schema/capability, product support, or candidate
3 evidence.
