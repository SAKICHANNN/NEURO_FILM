# P248 ACEScg OpenEXR scanline-streaming contract

## Question

Can the official OpenEXR 3.4.15 ordered-scanline API write the exact P247
24-megapixel procedural ACEScg/AP1-D60 float32 master in bounded row blocks,
preserving decoded pixels and metadata while keeping every fresh worker below
the unchanged 2 GiB process-tree RSS ceiling?

## Parent and material change

- Parent P247 closed the exact Python `OpenEXR.File` full-frame path because one
  of two workers peaked at 2,302,255,104 bytes, above 2,147,483,648 bytes.
- P248 does not change the image, ZIP compression, metadata, precision, resource
  limits or correctness gates. It changes the storage mechanism to official C++
  `OutputFile::writePixels()` calls over small, ordered scanline blocks.
- The implementation is isolated research code built from the exact official
  OpenEXR 3.4.15 source release. It does not replace P246/R1DO, enter the product
  renderer, or add a repository dependency.

## Frozen sources and toolchain

- Official release tag: `v3.4.15`, published by the Academy Software Foundation
  OpenEXR repository.
- Official release asset: `openexr-3.4.15.tar.gz`, 25,840,011 bytes, SHA-256
  `ab893d8003773ccd9a5556b2caf38da591ae37e20b06ee9d589a08984c5191f2`.
- Exact upstream-required Imath tag `v3.2.2`, Git object
  `1e480d11cb98b032a2dece9b9a8730512effc7f6`; the frozen tag archive is
  689,217 bytes with SHA-256
  `b4275d83fb95521510e389b8d13af10298ed5bed1c8e13efd961d91b1105e462`.
- Build only the required OpenEXR/Imath libraries and one private audit binary;
  tests, tools, examples, Python bindings and shared libraries are disabled.
- Windows x64 MSVC 19.50.35729, bundled CMake 4.2.3-msvc3 and Ninja 1.12.1
  are the frozen local build route. Build/install roots live only under the
  repo-relative P-backed `tmp/` entry and are removed after execution.

## Frozen writer and probe

- Input is generated, not read: exactly `4000x6000x3` float32 ACEScg,
  24,000,000 pixels and 288,000,000 logical input bytes, using the unchanged
  P247 integer-coordinate formula and 64-row generation blocks.
- The native writer accepts one row block at a time and never owns or receives
  a full-frame input array. Block height is exactly 16 scanlines, matching ZIP
  scanline chunking; scanlines are written in increasing order.
- File structure is single-part scanline RGB float32, ZIP lossless, with the
  exact P246/R1DO chromaticities and adopted-neutral metadata. Negative values
  and values above one are preserved.
- Publish through a sibling temporary file, close successfully, then atomically
  replace the requested destination. Invalid inputs and injected pre-publish
  failure must leave no destination mutation or temporary residue.

## Frozen controls and gates

1. Before build or formal scoring, bind the contract/config, P246/P247 evidence,
   official source asset, compiler/CMake/Ninja identities and native writer
   source/blob identities.
2. A small fixed probe must decode bit-exact with exact AP1/D60 metadata, reject
   malformed dimensions and injected pre-publish failure atomically, and replay
   in forward/reverse order.
3. Two fresh 24MP workers run sequentially. Each worker includes procedural row
   generation, hashing, native write and independent Python OpenEXR readback in
   its monitored process tree. RSS is sampled at intervals no greater than
   100 ms.
4. Each worker must have decoded float32 maximum absolute error `0`, the same
   input and decoded-pixel SHA-256, exact AP1/D60 metadata, negative/highlight
   preservation, finite samples and no new boundary clipping.
5. Each worker peak process-tree RSS must be at most 2,147,483,648 bytes, wall
   time at most 120 seconds and output size at most 1,073,741,824 bytes.
6. The two workers must agree on scientific payload, decoded pixels and
   metadata. Container byte identity between workers is required. Byte identity
   with the older Python writer is diagnostic only because producer/library
   header serialization may differ without changing the frozen master.
7. No external/project pixels, network during formal execution, GPU, target,
   reference or image-quality metric may be used. All owned build/output/temp
   artifacts are removed after the canonical report is written.

Any source, build, small-probe, pixel, metadata, resource, replay, atomicity or
cleanup failure closes this exact scanline implementation. Do not change block
height, compression, probe, compiler, thresholds, metadata or process topology
after observing a result.

## Claim ceiling

At most P248 can establish private local Windows x64 24MP resource feasibility
for one official-source native OpenEXR scanline writer and one procedural ACEScg
probe. It does not establish natural-image quality, arbitrary EXR support,
cross-platform parity, AP0/ST 2065-4 ACES interchange, renderer integration,
public package/schema/capability or product admission.
