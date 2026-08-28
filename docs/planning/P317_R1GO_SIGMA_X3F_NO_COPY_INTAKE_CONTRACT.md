# P317 — R1GO Sigma X3F dual-codec no-copy intake

## Question

Can the consumer independently execute the source-locked R1GO
`decode_sigma_x3f` Git object on the frozen CC0 Sigma SD10 HUFFMAN_10BIT and
DP1s TRUE sources, reproducing both complete stored-code uint16 arrays without
copying implementation or invoking the producer runner/LibRaw reference?

## Parent and boundary

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- P317 is an isolated no-copy intake after R1GO's private exact two-codec pass.
- It does not integrate a default RAW loader, add X3F codecs or continue
  adjacent Sigma exploration.

## Frozen inputs

- Exact producer commits, Git blobs and artifact hashes in
  `configs/p317_r1go_sigma_x3f_no_copy_intake_v1.json`.
- Exact repo-relative SD10 and DP1s X3F sources, byte counts, SHA-256
  identities, source-declared codecs/geometries and output hashes.
- Exact callable ID, producer contract and canonical two-row fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity. Mutable producer HEAD
   is not a gate.
2. Materialize only the exact `sigma_x3f.py` Git object in an owned system
   temporary package and import it from there.
3. Read both exact sources, parse their source-declared codec contracts and
   invoke the callable in forward and reverse order in fresh processes.
4. Require exact source/output hashes, uint16 HxWx3 geometry/codec identity,
   code bounds, owned writable C-contiguous outputs and source immutability.
5. Require one-byte truncation of each exact source to raise `ValueError`
   before output. This additive consumer control does not replace producer
   wrong-codec/channel/predictor/overlap controls.
6. Require byte-identical canonical reports and zero owned temp residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GO Sigma SD10 HUFFMAN_10BIT and
DP1s TRUE stored-code callable. No generic X3F/Foveon or Sigma RAW claim,
Quattro, spatial resampling, crop, black subtraction, calibration, white
balance, colour rendering, image quality, default loader, public
package/schema/capability, product mapping, stock evidence, automatic matching
or candidate-3 change.
