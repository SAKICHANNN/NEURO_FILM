# P312 — R1FV DNG ProfileGainTableMap2 stage no-copy intake

## Question

Can the consumer independently execute the exact R1FV
`apply_profile_gain_table_map_v2` Git object on the frozen sample04 main and
extra maps, reproducing the producer's strict and support-overrange float32
outputs without copying implementation or rebuilding the Adobe SDK oracle?

## Distinction and parent

- Parent: mature RAW/DNG/HDR explicit arithmetic infrastructure.
- P310 covers PGTM2 storage decoding; P311 separately closed a per-profile
  association handoff identity. P312 uses exact profile names to select the two
  maps and tests only stage arithmetic.
- The producer's gamma=2 case is frozen as nondiscriminating because sample04's
  two intensity entries are identical at every spatial node. P312 must preserve
  that limitation and cannot claim general non-unit-gamma semantics.

## Frozen inputs

- Exact producer commits and Git blobs in
  `configs/p312_r1fv_dng_profile_gain_table2_stage_no_copy_intake_v1.json`.
- Exact Adobe sample04 source bytes and exact main/extra table-name locks.
- Exact source-free 13x17x3 procedural probe, image area `(7,11,20,28)`,
  exposure-weight gain `1.25`, six producer output hashes and five invalid
  controls.

## Procedure and gates

1. Verify every immutable commit/blob/evidence/execution-lock identity. The
   producer's mutable current HEAD is deliberately not a gate.
2. Materialize the exact stage and association modules only inside an owned
   system temporary package and import them from there.
3. Parse sample04 with exact name-to-table locks, select maps by exact name, and
   independently reconstruct the frozen source-free probe.
4. Execute main/extra strict and overrange cases plus the frozen gamma=2
   diagnostic in opposite orders in two fresh processes.
5. Require exact producer input/output hashes, float32 finite outputs, strict
   `[0,1]`, overrange preservation, owned writable C-contiguous outputs and
   input immutability.
6. Require nonfinite RGB, nonpositive exposure, mismatched image area,
   invalid gamma and cross-profile table substitution to reject before output.
7. Require byte-identical reports, source immutability and zero temp residue.

Any failure closes this exact intake. No source, probe, table, name, output hash,
control, gate or claim may change after the first consumer stage execution.

## Claim ceiling

At most private no-copy intake of exact R1FV selected-profile PGTM2 stage
arithmetic for the two synthetic Adobe sample04 maps. No general gamma claim,
raw decode, full DNG renderer, real-file or image quality, arbitrary map/DNG,
default-loader change, public package/schema/capability, product mapping, stock
evidence, automatic matching, or candidate 3.
