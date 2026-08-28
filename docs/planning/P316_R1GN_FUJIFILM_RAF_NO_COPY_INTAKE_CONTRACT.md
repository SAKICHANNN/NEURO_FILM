# P316 — R1GN Fujifilm RAF dual-layout no-copy intake

## Question

Can the consumer independently execute the corrected, source-locked R1GN
`decode_fujifilm_raf` Git object on the frozen CC0 Fujifilm S2Pro layout-0 and
S7000 layout-1 sources, reproducing both complete stored-code uint16 arrays
without copying implementation or invoking the producer runner/reference
decoder?

## Identity correction and parent

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- The original handoff contract/evidence named an unresolvable full
  implementation commit. Consumer stopped before source reads. Producer
  correction `11397b22...` binds the real commit
  `1cb9b9953250127b5d06f78e3e9047040aa46cf2` and its exact core blob; science,
  sources, fixture, reports, gates and claim are unchanged.
- P316 is an isolated no-copy intake. It does not integrate a default RAW
  loader or continue adjacent RAF exploration.

## Frozen inputs

- Exact corrected producer commits, Git blobs and artifact hashes in
  `configs/p316_r1gn_fujifilm_raf_no_copy_intake_v1.json`.
- Exact repo-relative producer S2Pro and S7000 RAF sources, byte counts,
  SHA-256 identities, geometries, layouts, stored byte orders and output
  hashes.
- Exact callable ID and canonical producer fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity, including that the
   corrected implementation commit resolves to the frozen core blob. Mutable
   producer HEAD is not a gate.
2. Materialize only the exact `fujifilm_raf.py` Git object in an owned system
   temporary package and import it from there.
3. Read the two exact sources, parse their source-declared layout contracts and
   invoke the callable in forward and reverse order in fresh processes.
4. Require exact source/output hashes, uint16 geometry/layout identity, code
   bounds, owned writable C-contiguous outputs and source immutability.
5. Require one-byte truncation of each exact source to raise `ValueError`
   before output. This additive consumer control does not replace the
   producer's duplicate/missing/nonterminal controls.
6. Require byte-identical canonical reports and zero owned temp residue.

Any failure closes the exact intake. No source, implementation object, output
identity, control, gate or claim may change after first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GN Fujifilm S2Pro layout-0 and
S7000 layout-1 stored-code callable. No generic RAF or Fujifilm RAW claim,
compressed RAF, SuperCCD spatial remapping, crop, black subtraction,
calibration, white balance, demosaic, colour rendering, image quality, default
loader, public package/schema/capability, product mapping, stock evidence,
automatic matching or candidate-3 change.
