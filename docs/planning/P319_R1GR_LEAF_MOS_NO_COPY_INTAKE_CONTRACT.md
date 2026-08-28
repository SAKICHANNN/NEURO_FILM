# P319 — R1GR Leaf Aptus 22 MOS dual-storage no-copy intake

## Question

Can the consumer independently execute the source-locked R1GR
`decode_leaf_mos` Git objects on the frozen CC0 Leaf Aptus 22 Compression-99
SOF3 lossless-JPEG and Compression-1 uncompressed sources, reproducing both
complete uint16 mosaics without copying implementation or invoking the
producer runner/LibRaw reference?

## Parent and dependency boundary

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- `leaf_mos.py` depends on `lossless_jpeg.py`; both exact blobs and SHA-256
  identities are frozen by R1GR evidence/execution lock and must be imported
  together from the implementation commit.
- P319 is an isolated no-copy intake. It does not integrate a default RAW
  loader or continue adjacent MOS exploration.

## Frozen inputs

- Exact producer commits, Git blobs and artifact hashes in
  `configs/p319_r1gr_leaf_mos_no_copy_intake_v1.json`.
- Exact repo-relative compressed and uncompressed Aptus 22 MOS sources, byte
  counts, SHA-256 identities, storage modes, geometry and output hashes.
- Exact callable ID, producer contract and canonical fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity. Mutable producer HEAD
   is not a gate.
2. Materialize only exact `leaf_mos.py` and `lossless_jpeg.py` Git objects in
   an owned system temporary package and import them from there.
3. Read both exact sources, parse source-declared compression/storage facts and
   invoke the callable in forward and reverse order in fresh processes.
4. Require exact source/output hashes, uint16 HxW geometry/compression identity,
   code bounds, owned writable C-contiguous outputs and source immutability.
5. Require one-byte truncation of each exact source to raise `ValueError`
   before output. This additive control does not replace producer
   wrong-compression/IFD/tile/SOI/byte-order/row controls.
6. Require byte-identical canonical reports and zero owned temp residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GR one-camera/two-storage Leaf
Aptus 22 MOS stored-code callable. No generic MOS or Leaf RAW claim, other
lossless-JPEG profiles, crop, black subtraction, calibration, white balance,
demosaic, colour rendering, image quality, default loader, public
package/schema/capability, product mapping, stock evidence, automatic matching
or candidate-3 change.
