# P254 R1DZ DNG ProfileHueSatMap overrange consumer-intake contract

## Question

Can Neuro-Film independently consume a prospectively published, versioned
producer callable for the already-frozen R1DR/R1DZ DNG ProfileHueSatMap
arithmetic, reproduce its canonical fixture exactly, and reject invalid or
ambiguous profile inputs without copying the producer implementation into this
repository?

P254 is a private mechanical handoff audit. It does not reopen the automatic
single-reference candidate counter, R1DU, or R1EA, and it cannot authorize a
full DNG renderer or product capability.

## Parent state and fixed facts

- Consumer P244/P245 currently reject ProfileHueSatMap-family DNGs before pixel
  decode rather than silently omitting the profile. Those guards remain the
  active behavior throughout P254.
- Producer R1DR established the metadata-chain order only as private arithmetic.
- Corrected producer R1DZ evidence is read from exact commit
  `d8bf6410f8b24e9700704dbcec8f05fe890453d8`, path
  `docs/evidence/R1DZ_DNG_PROFILE_HUESATMAP_OVERRANGE_RESULT.json`, SHA-256
  `6a5e8603123804be1ba8c223c6d5deeba428fedf760e101e141acafcc840ddd5`.
- R1DZ formal report is 4,081 bytes, SHA-256
  `772071979549ccf5be5b0d382884b777d0fad9a5ee6e6bb92a7c8619b390fec4`,
  scientific identity
  `b613ed6675081aedd0fb0f270b3d0a04d797bbc4b3999f671e53796e2e7a13dd`.
- Existing consumer metadata source
  `data/reference_color_match/p240_rawpixls_capture_metadata_v1/exif/1052.txt`
  is 15,658 bytes, SHA-256
  `e71b44097dfe38e8cbc1753932269014b7d21acc63698413f415780286d717fd`.
  Its two parsed table float32 hashes must remain
  `fa5d69874bc55aea4c9bf588709be515e121acccb5882cd63b6b6264553ce68b`
  and
  `65c161161f88a03b6728e47edff371998aec9435636b08a1b943802d48b0f142`.
- R1DZ explicitly recorded `payload_or_interface_published=false`. Therefore
  the old experiment source, producer worktree module, ignored outputs, or
  report-only basis are not admissible consumer interfaces.

## Two-stage prospective freeze

### Stage A: protocol freeze

This contract and its configuration are committed before any callable artifact
exists or any callable fixture is executed. Stage A may inspect only the fixed
facts above and producer Git metadata. It produces no arithmetic or product
result.

### Stage B: source-lock amendment

Before the first callable import, deserialization, or fixture output read, an
additive tracked amendment must bind all of the following:

1. producer publication commit, artifact path, Git blob, byte count and
   SHA-256;
2. artifact format/version and the exact public function names/signatures;
3. complete array dtype, shape, domain, table-axis order and stage-order
   semantics;
4. one-table and two-table interpolation-weight semantics;
5. `support_overrange` behavior, including negative-input handling and the
   encode/decode-overrange rule;
6. caller ownership, output ownership/contiguity, mutation policy and failure
   atomicity;
7. canonical fixture envelope, its exact input/table/output byte identities,
   and the producer report/evidence that binds them;
8. invalid controls for nonfinite RGB/table, malformed dimensions, negative
   scale factors, invalid interpolation weights, unsupported encoding/dynamic
   range, and invalid block/workspace capacity when applicable;
9. default-SDR unchanged identities and the relationship to R1DQ/R1DR;
10. producer rights and claim ceiling.

The amendment itself must be committed before P254 reads expected fixture
outputs. A missing, ambiguous, mutable, worktree-only, ignored, or report-only
artifact stops P254 as `NOT_READY_PRODUCER_HANDOFF_GAP`, not as a scientific
failure.

## Frozen consumer execution

After Stage B is valid:

1. Read the callable, fixture, manifest and producer evidence only through
   their exact Git objects. Do not import the producer worktree package.
2. Materialize the callable in a fresh isolated temporary module or install its
   exact offline artifact. Do not copy its implementation into `src/`.
3. Independently parse the exact P240 DJI metadata and prove both table hashes.
4. Reconstruct or deserialize only the canonical fixture specified by the
   producer handoff. No new RAW, DNG, target, reference, preferred render or
   project image may be read.
5. Exercise normal/reverse enumeration in two fresh processes and require
   byte-exact scientific payloads.
6. Remove every owned temporary module, install root and output. Retain only
   the canonical JSON report and tracked evidence.

## Frozen gates

- exact Stage A contract/config and Stage B source-lock identities;
- exact corrected R1DZ/R1DR evidence and default-SDR parent identities;
- exact local DJI EXIF and parsed Data1/Data2 hashes;
- exact callable artifact/fixture/manifest Git objects;
- exact canonical output bytes for all published fixture cases;
- float32 dtype, documented shapes, finite outputs, owned contiguous output and
  immutable inputs/tables;
- documented one/two-table interpolation and `support_overrange` semantics;
- every required invalid control rejects before returning output;
- consumer P244/P245 guards and existing default DNG behavior remain unchanged;
- zero network, zero new RAW/DNG/pixel/target reads, zero source copy and zero
  owned temporary residue;
- two fresh-process reports byte-exact.

Any mismatch closes the exact handoff. Do not substitute the producer
worktree source, reconstruct an interface from internal experiment code, tune
weights/tables/probes, relax equality, clip overrange values, weaken a guard,
or map the result to a public capability.

## Claim ceiling

At most P254 can establish private consumer-side reproducibility of one exact
versioned R1DZ/R1DR callable handoff on its fixed canonical fixture. It does
not prove complete DNG ordering, real-pixel quality, arbitrary camera/profile
support, R1DU/R1EA recovery, default-loader integration, a Neuro-Film public
API, package/schema/capability, film-stock evidence, automatic-reference
matching, or product admission. Candidate 3 remains closed at `2/3`.
