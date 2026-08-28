# P311 — R1FU DNG per-profile gain-table no-copy intake

## Question

Can the consumer independently load the exact source-locked R1FU parser Git
object and reproduce its immutable main-plus-extra camera-profile to
ProfileGainTableMap2 association on the one exact Adobe DNG 1.7 sample, without
copying producer implementation or decoding image pixels?

## Parent and distinction

- Parent: the mature RAW/DNG/HDR explicit-infrastructure branch.
- Producer parent: R1FU `zhuise.dng-per-profile-gain-table-selection.v1`.
- P310 already intakes the R1FT PGTM2 storage decoder on samples 05–07. P311
  instead tests the R1FU container/profile association on sample 04. It does
  not repeat encoding parity, interpolation, rendering, or product integration.

## Frozen inputs

- Exact producer repository commits and Git blobs listed in
  `configs/p311_r1fu_dng_per_profile_gain_table_no_copy_intake_v1.json`.
- Exact Adobe DNG 1.7 `04_PGTM2_per_profile.dng`, 6,010,280 bytes, SHA-256
  `14a2afc29ad1c848a12390d2c30abd36f8adb29aab90866f66edbd33efd6e58e`.
- Only TIFF/profile metadata and the two PGTM payloads may be read. Image,
  preview, mask, SDK render, and output-pixel reads are zero.

## Procedure

1. Verify producer preregistration, evidence, parser, dependency, source and
   commit identities before execution.
2. Materialize the two exact parser Git blobs only inside one owned system
   temporary package; import from that isolated package and remove it after use.
3. Run the byte parser and path reader in opposite order in two fresh processes.
4. Canonicalize the immutable profile-set metadata, decoded gain hashes and
   failure controls. Forward and reverse reports must be byte-identical.
5. Verify the source is unchanged and no temporary residue remains.

## Frozen gates

- all artifact, commit, source and producer evidence identities exact;
- isolated no-copy Git-object import;
- exact canonical main then extra ordering, names, offsets, component hashes,
  PGTM payload hashes, decoded gain hashes and common Raw-IFD v1 map hash;
- frozen/slotted result objects and read-only decoded arrays;
- parse-bytes and read-path summaries exact;
- wrong source lock, swapped profile-table lock, missing profile lock,
  truncated source, invalid TIFF byte order and invalid extended-profile magic
  all raise `ValueError` before a profile set is returned;
- source immutability, zero image-pixel decode, exact replay and zero temp residue.

Any failure closes this exact intake. No source, lock, parser, control or gate may
change after the first sample04 parse.

## Claim ceiling

At most a private no-copy intake of one source-locked R1FU association parser on
one synthetic Adobe SDK sample. This is not arbitrary-DNG support, profile
selection UI, interpolation/stage application, image quality, a public package
or capability, a product mapping, stock evidence, automatic matching, or
candidate 3.
