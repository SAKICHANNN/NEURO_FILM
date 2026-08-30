# P321 — R1GX Sony ARQ four-shot no-copy intake

## Question

Can the consumer independently execute the exact source-locked R1GX
`decode_sony_arq` Git object on the frozen CC0 Sony ILCE-7RM3 composed ARQ,
reproducing the complete 5320x8000x4 uint16 RGBG stored-code array without
copying implementation or invoking the producer runner or LibRaw reference?

## Parent and dependency boundary

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- Dependency: the corrected R1GX V2 callable contract, execution lock and
  evidence lineage ending at producer commit `c24cb401`.
- P321 is an isolated no-copy intake. It does not integrate a default RAW
  loader and does not consume the four component ARW files.

## Frozen inputs

- Exact producer commits, Git blobs and artifact hashes in
  `configs/p321_r1gx_sony_arq_no_copy_intake_v1.json`.
- The exact repo-relative 340,873,216-byte composed ARQ source and its SHA-256.
- Exact callable ID, corrected V2 execution identity, producer contract and
  544-byte canonical fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity. Mutable producer HEAD
   is not a gate, and superseded R1GX V1 report identities are forbidden.
2. Materialize only the exact `sony_arq.py` Git object in an owned system
   temporary package and import it from there.
3. Independently decode the canonical fixture and exact composed ARQ in fresh
   forward/reverse consumer processes.
4. Require exact fixture/source/output hashes, 5320x8000x4 uint16 RGBG output,
   code bounds, owned writable C-contiguous output and source immutability.
5. Require one-byte truncation and mutable input to reject before output.
6. Require corrected producer contract/execution/evidence identities, byte-
   identical consumer reports and zero owned temporary residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GX one-camera/one-group Sony ARQ
four-sample stored-code callable. No ARW-to-ARQ writer, component-ARW intake,
alignment, fusion, crop, black subtraction, calibration, white balance,
demosaic, colour rendering, image quality, default loader, public package,
schema or capability, product mapping, stock evidence, automatic matching or
candidate-3 change.
