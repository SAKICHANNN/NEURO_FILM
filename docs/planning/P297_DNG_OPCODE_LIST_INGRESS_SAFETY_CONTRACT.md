# P297 DNG OpcodeList ingress safety contract

## Scope

P297 is a private fail-closed safety correction for the exact P98 opt-in DNG
ForwardMatrix raster loader. It does not apply an opcode or claim arbitrary-DNG
support. Before camera-pixel decode, it parses `OpcodeList1` (51008),
`OpcodeList2` (51009), and `OpcodeList3` (51022) using the Adobe DNG 1.7.1
big-endian list envelope.

The loader may proceed only when every present list is structurally valid and
every opcode has the Adobe optional flag set. A required opcode, reserved flag,
truncated header/payload, trailing byte, impossible count, unsupported version,
or non-byte tag value rejects before `_decode_camera_linear_dng`. Optional-only
lists remain accepted with an explicit warning because the DNG specification
permits them to be skipped; this is not evidence that LibRaw applies them.

## Frozen authority and cohort

- Adobe DNG SDK 1.7.1 Build 2652 archive:
  `data/external/adobe_dng_sdk_1_7_1_2652/dng_sdk_1_7_1_2652_20260714.zip`,
  80,494,799 bytes, SHA-256
  `73499b47f4683e12120a234bd0946f02e52ab2ff9834bcbd0e9f8ab4f923360e`.
- SDK tag authority: 51008/51009/51022; opcode optional flag is bit 0.
- Exact P98 config SHA-256
  `5fb17f82cfecdae40a4c812f2ea9c26641e91bcab43e17c8a6a4bc94b2f03c35`.
- The five P98 source rows and their frozen WorkingImage float32 hashes remain
  unchanged. Metadata preflight found no `OpcodeList1`; four rows contain only
  optional opcodes and Blackmagic contains no opcode list.

## Frozen gates

1. Synthetic required opcodes in each of the three list tags reject before
   camera decode and produce no usable output.
2. Malformed count/header/payload/trailing/reserved-flag/version controls reject
   before camera decode.
3. Optional-only synthetic lists parse deterministically and add exactly one
   `optional_dng_opcodes_may_be_skipped` warning.
4. The exact five P98 sources retain their frozen pixel hashes and source bytes.
5. The formal real-file inventory matches the preregistered tag byte hashes,
   opcode IDs, versions, flags and counts.
6. Forward/reverse fresh-process scientific reports are byte-exact.

Any failure closes P297 without applying opcodes, changing pixel arithmetic,
replacing rows, relaxing gates, or claiming that LibRaw executed an optional
opcode. Claim ceiling: private exact-five no-silent-required-opcode safety only;
no full DNG renderer, image quality, default loader, package/schema/capability,
stock evidence, candidate-3, or product admission.
