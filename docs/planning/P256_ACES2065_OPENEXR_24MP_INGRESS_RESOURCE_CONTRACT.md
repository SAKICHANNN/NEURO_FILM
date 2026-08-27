# P256 ACES2065-1 OpenEXR 24MP ingress resource contract

## Question

Can the unchanged strict P251 ACES2065-1/AP0 OpenEXR ingress consume the exact
P255 24MP AP0/D60 master and return an owned ACEScg/AP1 `WorkingImage` with the
existing numerical contract while remaining below a frozen 2 GiB process-tree
RSS and 120 second wall-time ceiling?

P256 is an end-to-end scale/resource check. It does not change P251 or P255,
does not add a streaming reader, and does not establish image quality or a
product interface.

## Frozen parents and input

- P251's loader module, config and evidence bytes are exact and unchanged.
- P255's config, evidence, native writer, official OpenEXR/Imath sources and
  toolchain are exact and unchanged.
- Each complete controller independently builds the frozen P255 writer and
  creates its exact `4000x6000x3` AP0 master. The source must be 42,251,493
  bytes with SHA-256 `eeced074...f275` before any reader worker starts.
- The procedural source equations, AP1-to-AP0 matrix, AP0-to-AP1 matrix,
  ZIP compression, metadata, coordinate period and block sizes remain fixed.
- Network, project/external image pixels and target/reference pixels remain
  zero.

## Execution

Run two complete controllers from committed source and config. Each controller:

1. verifies all P251/P255 parent, source, wheel and toolchain identities;
2. builds P255 and creates one exact 24MP source in a private temporary root;
3. launches two fresh Python workers with the exact OpenEXR 3.4.15 wheel;
4. each worker calls the unchanged public
   `load_aces2065_openexr_working_image` once;
5. hashes the complete ACEScg output and checks it blockwise against the frozen
   procedural AP1 source without allocating a second full expected image;
6. records ownership, shape, provenance, negative/highlight preservation,
   source immutability, wall time and sampled process-tree RSS;
7. removes the source, wheel install, build and every owned temporary byte.

The controller performs no alternate decoder, no row/chunk tuning and no
reader implementation change after observing a result.

## Gates

- both complete controllers have the same stable scientific identity;
- four fresh reader workers have identical output hashes and stable facts;
- exact shape `4000x6000x3`, float32, owned/C-contiguous/writeable ACEScg/AP1
  scene-linear `WorkingImage` with unchanged P251 ingress metadata;
- maximum AP1 error at most `9.5367431640625e-7`, matching P251's frozen small
  roundtrip ceiling;
- finite output, preserved negative and above-one values, and zero new exact
  0/1 boundary samples relative to the frozen procedural AP1 source;
- P255 source bytes/hash and metadata remain exact and source bytes immutable;
- every reader worker process-tree RSS at most 2,147,483,648 bytes and wall
  time at most 120 seconds;
- all source/module/config/evidence/wheel/toolchain identities exact;
- zero network, external/project pixels and owned residue.

Resource measurements and independently rebuilt PE bytes are recorded but
excluded from the stable scientific identity. Any worker termination, memory
allocation failure, timeout, identity, numeric, ownership, metadata or cleanup
failure is a formal failure of this unchanged full-frame reader at 24MP.

## Stop rule and claim ceiling

On failure, close only the exact P251 full-frame read mechanism at 24MP. Do not
raise the resource/error gates, change matrices, retain fewer checks, lower
resolution, alter OpenEXR, or add chunking inside P256. A materially different
streaming/scanline intake may be preregistered separately.

A pass establishes only private Windows/Python 24MP strict-ingress resource and
mechanical conformance for one procedural P255 AP0 master. It does not prove
natural-image quality, arbitrary EXR/ACES conformance, SMPTE certification,
cross-platform parity, public dependency/API/package/schema/capability,
default renderer integration, automatic reference matching, film-stock
evidence or product admission. Candidate 3 remains closed at `2/3`.
