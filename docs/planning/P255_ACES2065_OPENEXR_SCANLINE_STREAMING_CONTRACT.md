# P255 ACES2065-1 OpenEXR scanline-streaming contract

## Question

Can the exact P248 official-source native ordered-scanline mechanism apply the
fixed P249/R1DT ACEScg AP1-to-ACES2065-1 AP0 transform per row and publish a
strict 24MP ACES2065-1 float32 ZIP OpenEXR master while preserving exact
decoded pixels, AP0/D60 container identity, bounded memory, deterministic
container bytes, failure atomicity, and the unchanged P248 AP1 parent?

P255 fills the specific AP0/ST2065-1 interchange gap left by P248.  It is not a
generic EXR writer, renderer integration, quality experiment, or product
admission.

## Parent bindings

- P248 is the retained 24MP Windows/MSVC ordered 16-scanline OpenEXR resource
  mechanism.  Its exact procedural ACEScg probe, OpenEXR 3.4.15/Imath 3.2.2
  sources, compiler/toolchain, ZIP compression, 2 GiB RSS ceiling and 120 s
  worker ceiling are unchanged.
- P249 independently binds the R1DT AP1-to-AP0 matrix, AP0/D60
  chromaticities/adoptedNeutral, `acesImageContainerFlag=1`,
  `colorInteropID=lin_ap0_scene`, and the exact small reference container.
- P251 is the strict inverse consumer intake and remains unchanged.
- No external or project image, target, reference, RAW or DNG pixel is read.

## Frozen implementation

The candidate is a new private native source derived from the existing P248
writer structure.  It may change only:

1. each generated ACEScg triplet is multiplied by the exact P249 float64
   AP1-to-AP0 matrix and rounded once to float32 before write;
2. header chromaticities become exact AP0/D60;
3. `acesImageContainerFlag=1` and `colorInteropID=lin_ap0_scene` are required;
4. temporary suffix and executable identity are separately versioned P255.

Image geometry `4000x6000x3`, procedural AP1 source equations, coordinate
period 1024, generation block 64, write block 16, ZIP compression, atomic
publish, official source/toolchain and process topology remain fixed.  P248
source and evidence bytes must not change.

## Prescore checks

Before the 24MP run:

- independently compute expected AP0 float32 rows from the unchanged AP1
  equations and the exact P249 matrix;
- run a `17x13` native probe twice and compare decoded AP0 pixels against both
  the independent computation and the exact P249 Python writer on the same
  AP1 values;
- require exact AP0/D60 metadata, flags and ID;
- reject invalid geometry/row block and prove injected pre-publish failure
  preserves an existing destination with no temporary residue;
- confirm every P248 parent identity and its retained 24MP output hash.

Any prescore failure stops before 24MP execution.  Numerical tolerance,
matrix, block size, probe, compression, metadata, compiler, resource ceiling
and process topology may not be changed after observing a result.

## Formal execution and gates

Run two complete controllers from the committed source/config.  Each controller
performs two fresh 24MP workers and one fresh small-control worker.  Require:

- exact source, parent evidence, source archive, wheel and toolchain identities;
- exact repeated 24MP container bytes and decoded AP0 float32 bytes;
- decoded maximum absolute error `0.0` against the independent frozen AP0 row
  computation;
- exact AP0/D60 chromaticities and adopted neutral,
  `acesImageContainerFlag=1`, `colorInteropID=lin_ap0_scene`, RGB float32 ZIP
  scanline structure, finite values, preserved negatives and highlights, and
  zero newly introduced exact 0/1 boundaries relative to expected AP0;
- exact small-probe parity with the P249 writer and repeat container;
- invalid-input rejection and failure-atomic create/replace behavior;
- every fresh worker process-tree RSS at most 2,147,483,648 bytes and wall time
  at most 120 seconds;
- zero network, external/project pixels and owned workspace residue;
- P248 source/evidence/config and retained AP1 output hashes unchanged.

Native PE bytes are recorded but are not a gate because P248 already proved
that independent MSVC build roots can differ in PE-container bytes while the
scientific output remains exact.  Resource measurements are reported and
excluded from the stable scientific identity.

## Stop rule and claim ceiling

Any source, parent, build, matrix, small-probe, pixel, metadata, resource,
replay, atomicity, cleanup or regression failure closes this exact AP0
scanline implementation.  Do not rescue it by tolerance, matrix-order, block,
compression, compiler, probe, metadata or resource-threshold changes.

A pass establishes only private Windows x64 24MP resource and mechanical
conformance for one procedural ACES2065-1/AP0 OpenEXR master.  It does not
prove natural-image quality, arbitrary EXR/ACES conformance, SMPTE
certification, cross-platform parity, public dependency/API/package/schema/
capability, renderer integration, automatic reference matching, film-stock
evidence or product admission.  Candidate 3 remains closed at `2/3`.
