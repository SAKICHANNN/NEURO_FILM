# P320 — R1GS Hasselblad X2D dual-container no-copy intake

## Question

Can the consumer independently execute the source-locked R1GS
`decode_hasselblad_x2d` Git object on the frozen CC0 Hasselblad X2D 100C
two-3FR/two-FFF cohort, reproducing all four complete uint16 mosaics without
copying implementation or invoking the producer runner/LibRaw reference?

## Parent and dependency boundary

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- The exact standalone `hasselblad_x2d.py` blob and producer
  preregistration/source-lock/execution-lock/contract/fixture/evidence objects
  are frozen before consumer decode.
- P320 is an isolated no-copy intake. It does not integrate a default RAW
  loader or continue adjacent Hasselblad exploration.

## Frozen inputs

- Exact producer commits, Git blobs and artifact hashes in
  `configs/p320_r1gs_hasselblad_x2d_no_copy_intake_v1.json`.
- Exact repo-relative four-file X2D sources, byte counts, SHA-256 identities,
  container/compression modes, geometry and output hashes.
- Exact callable ID, producer contract and canonical fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity. Mutable producer HEAD
   is not a gate.
2. Materialize only the exact `hasselblad_x2d.py` Git object in an owned system
   temporary package and import it from there.
3. Read all four exact sources, parse source-declared container/compression
   facts and invoke the callable in forward and reverse order in fresh
   processes.
4. Require exact source/output hashes, uint16 HxW geometry/container/
   compression identity, code bounds, owned writable C-contiguous outputs and
   source immutability.
5. Require one-byte truncation of each exact source to raise `ValueError`
   before output. This additive control does not replace producer branch,
   predictor, byte-order, IFD, extent and row controls.
6. Require byte-identical canonical reports and zero owned temp residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GS one-camera/two-container/
four-file Hasselblad X2D stored-code callable. No generic 3FR, FFF or
Hasselblad RAW claim, crop, black subtraction, calibration, white balance,
demosaic, colour rendering, image quality, default loader, public
package/schema/capability, product mapping, stock evidence, automatic matching
or candidate-3 change.
