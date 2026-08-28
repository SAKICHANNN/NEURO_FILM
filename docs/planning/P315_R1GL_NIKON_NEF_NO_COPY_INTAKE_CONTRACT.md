# P315 — R1GL Nikon NEF compression-34713 no-copy intake

## Question

Can the consumer independently execute the exact producer R1GL
`decode_nikon_nef` Git object on the frozen CC0 Nikon D6 12-bit and D3S
14-bit sources, reproducing both complete unpack-domain uint16 arrays without
copying implementation or invoking the producer runner/reference decoder?

## Distinction and parent

- Parent: mature RAW/DNG/HDR explicit decode infrastructure.
- P315 is an isolated no-copy intake of a versioned producer callable. It does
  not integrate a default RAW loader or continue adjacent Nikon exploration.
- The accepted domain is exactly the two source-bound classic-TIFF Nikon
  compression-34713 modes published by R1GL. Nikon HE, split-row curves,
  alternate compression/curve versions and all later image-processing stages
  remain outside the contract.

## Frozen inputs

- Exact producer commits, Git blobs and artifact hashes in
  `configs/p315_r1gl_nikon_nef_no_copy_intake_v1.json`.
- Exact repo-relative producer D6 and D3S NEF sources, their byte counts,
  SHA-256 identities, geometries, bit depths and expected output hashes.
- Exact callable ID and canonical producer fixture.

## Procedure and gates

1. Verify every immutable commit/blob/artifact identity. The producer's mutable
   current HEAD is deliberately not a gate.
2. Materialize only the exact `nikon_nef.py` Git object inside an owned system
   temporary package and import it from that package.
3. Read the two exact sources, parse their embedded contracts, and invoke the
   callable in forward and reverse order in two fresh consumer processes.
4. Require exact source and output hashes, exact uint16 geometry, finite-domain
   code bounds, owned writable C-contiguous outputs and source immutability.
5. Require one-byte truncation of each exact source to raise `ValueError`
   before output. This consumer control is additive and does not replace the
   producer's bit-depth/compression/curve-version controls.
6. Require byte-identical canonical reports, source immutability and zero owned
   temporary residue.

Any failure closes this exact intake. No source, implementation object, output
identity, control, gate or claim may change after the first consumer decode.

## Claim ceiling

At most private no-copy intake of the exact R1GL Nikon D6 12-bit and D3S
14-bit compression-34713 unpack callable. No generic Nikon NEF or lossless
claim, Nikon HE or split-table support, crop, black subtraction, calibration,
white balance, demosaic, colour rendering, image quality, default loader,
public package/schema/capability, product mapping, stock evidence, automatic
matching or candidate-3 change.
