# P318 — R1GP Samsung SRW multi-mode no-copy intake

## Question

Can the consumer independently execute the corrected, source-locked R1GP
`decode_samsung_srw` Git object on the frozen CC0 Samsung NX500 12/14-bit ×
normal/lossless cohort, reproducing all four complete stored-code uint16
mosaics without copying implementation or invoking the producer runner/LibRaw
reference?

## Identity correction and parent

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- Initial R1GP contract `6f59c657` mistyped the expanded oracle and formal
  execution commits. Producer correction `b75807f3...` changes only those
  identities; sources, core, fixture, reports, metrics, gates and claim remain
  unchanged.
- P318 is an isolated no-copy intake. It does not integrate a default RAW
  loader or continue adjacent SRW exploration.

## Frozen inputs

- Exact corrected producer commits, Git blobs and artifact hashes in
  `configs/p318_r1gp_samsung_srw_no_copy_intake_v1.json`.
- Exact four repo-relative NX500 SRW sources, byte counts, SHA-256 identities,
  bit depths, modes, geometries and output hashes.
- Exact callable ID, corrected contract and canonical producer fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity, including that both
   corrected commit IDs resolve and both superseded mistyped IDs reject.
   Mutable producer HEAD is not a gate.
2. Materialize only exact `samsung_srw.py` in an owned system temporary
   package and import it from there.
3. Read the four exact sources, parse source-declared bit-depth/mode contracts
   and invoke the callable in forward and reverse order in fresh processes.
4. Require exact source/output hashes, uint16 HxW geometry/bit-depth/mode
   identity, code bounds, owned writable C-contiguous outputs and source
   immutability.
5. Require one-byte truncation of every exact source to raise `ValueError`
   before output. This additive consumer control does not replace producer
   wrong-mode/bit-depth/cyclic-row/duplicate-tag controls.
6. Require byte-identical canonical reports and zero owned temp residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GP one-camera/four-mode Samsung
SRW stored-code callable. No generic SRW or Samsung RAW claim, crop, black
subtraction, calibration, white balance, demosaic, colour rendering, image
quality, default loader, public package/schema/capability, product mapping,
stock evidence, automatic matching or candidate-3 change.
