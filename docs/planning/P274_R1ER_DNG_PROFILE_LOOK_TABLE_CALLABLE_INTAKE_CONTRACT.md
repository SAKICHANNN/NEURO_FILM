# P274 — R1ER DNG ProfileLookTable callable intake contract

Status: `FROZEN_BEFORE_FIXTURE_DESERIALIZATION_OR_CALLABLE_IMPORT`

## Question

Can this consumer independently verify and execute the exact source-locked R1ER
`zhuise.dng-profile-look-table-callable.v1` from producer Git objects, without
copying the implementation into consumer source or rerunning the producer's
science?

## Frozen execution

- Verify every producer object by commit, Git blob, byte count and SHA-256.
- Materialize only the bound arithmetic core and callable in a fresh temporary
  package; never import the producer worktree and never add a consumer core/API.
- Validate the tiny canonical payload with the bound Draft 2020-12 schema.
- Execute the exact fixture and require byte-exact expected float64 output,
  direct-core parity, input/payload immutability and an owned writable
  C-contiguous float64 result.
- Exercise wrong rank, empty, nonfinite and out-of-domain inputs plus malformed
  dimensions/data/schema/encoding controls. Every failure must be `ValueError`
  before an output is returned.
- Run two fresh processes in forward/reverse control order. Require byte-exact
  scientific payload and zero temporary residue.

The stage contract is fixed as `ProfileHueSatMap -> ProfileGainTableMap or
ProfileGainTableMap2 -> exposure -> ProfileLookTable -> ProfileToneCurve`.
The callable input is finite nonempty `HxWx3` XYZ D50 representing linear ROMM
after exposure and before tone curve. The real CC0 Fuji 36x8x16 table remains
bound by hash only and is not duplicated into the tiny fixture.

## Stop rule and claim ceiling

Any source identity, schema, stage-order, domain, output, ownership, mutation,
invalid-control, replay or cleanup mismatch closes P274. No copy, tolerance
relaxation, loader integration or wrapper/runtime rescue is allowed.

Passing proves only private mechanical intake of one exact callable and fixture.
It is not a complete DNG renderer, real-file quality result, arbitrary profile
support, public package/schema/capability, stock evidence, automatic matching or
product admission. Candidate 3 remains closed at 2/3.
