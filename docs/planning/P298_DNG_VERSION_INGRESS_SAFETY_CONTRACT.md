# P298 — DNG version ingress safety contract

## Parent and scope

P298 is a private ingress-safety child of the exact-five P98 DNG ForwardMatrix
path. It closes a container-identity gap left distinct from P297 OpcodeList
handling: a `.dng` suffix and profile tags are not sufficient evidence that a
file declares a supported DNG version.

This leaf does not add DNG rendering, profile stages, quality evidence, a
default loader, a public interface, or a product capability.

## Frozen authority

- Adobe DNG SDK 1.7.1 Build 2652 tag codes: `DNGVersion=50706` and
  `DNGBackwardVersion=50707`.
- Both tags, when present, are exactly four BYTE values.
- `DNGVersion` is required for this strict private path and must be in
  `[1.0.0.0, 1.7.1.0]`.
- Missing `DNGBackwardVersion` resolves to the DNG version with its final two
  components cleared, matching the SDK default.
- The resolved backward version must be at least `1.0.0.0`, no greater than
  `DNGVersion`, and no greater than the supported SDK version.

## Frozen cohort and controls

The five already-consumed P98 files are the only accepted real rows. Their
source bytes and P244 output hashes remain the oracle. Synthetic metadata-only
controls cover missing version, wrong type/count/value, pre-1.0 version,
future version, invalid backward version, backward greater than version, and a
future backward version. Every rejected control must stop before camera decode.

## Gates and stop rule

Pass requires exact authority identities, exact five-row version inventory,
all invalid controls rejected before camera decode, unchanged P98 pixels and
source bytes, finite outputs, and byte-exact forward/reverse scientific
reports. Any failure closes P298 without version normalization, row
replacement, broader DNG claims, or gate relaxation.

