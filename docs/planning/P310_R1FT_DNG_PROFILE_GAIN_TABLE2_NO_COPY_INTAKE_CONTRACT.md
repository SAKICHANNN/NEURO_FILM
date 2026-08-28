# P310 R1FT DNG ProfileGainTableMap2 No-Copy Intake Contract

## Question

Can the consumer independently hash, isolate, import and execute the exact
producer R1FT `ProfileGainTableMap2` parser Git object on the three official
Adobe DNG 1.7 v2 encoding samples without copying implementation, decoding
image pixels, or weakening the existing P257 predecode guard?

## Frozen inputs

- producer HEAD `4816b51df492018117aac6301ea068bf9806455e`;
- parser implementation commit `31ca3e378701755943cd457055215681c153e8e1`;
- R1FT evidence commit `f629bd755b3b4c66bfdd065f672850f38f109232`;
- R1FT preregistration commit `3e6deeca445fdb24aa1a377d94b5f5c73e241370`;
- exact Adobe samples 05/06/07 and their locked tag offsets, payload lengths and
  SHA-256 identities from committed R1FT evidence;
- exact expected decoded float32 gain hash
  `eac0fd9772c85219adfb17ccccbfd7386848a62817424bc69e1731844368790b`.

The consumer formal audit may read and hash the three source containers and
exact PGTM2 tag payloads only after this contract/config is committed. It must not
decode image samples, invoke the Adobe renderer, read sample08 as an oracle,
or import producer working-tree source.

## Excluded preformal invocation

One focused dry run executed after these exact contract/config/audit/test bytes
were written but before the freeze commit. It is excluded because the source
lock was not yet independently recoverable from Git. It caused no file change,
threshold/source/control/gate change, evidence publication or claim. The only
additive correction is this disclosure and the corresponding config status;
formal execution must start again from the committed freeze and audit heads.

## Gates

1. Every producer Git object, source container, TIFF tag and payload identity
   is exact.
2. The parser is imported only from an isolated temporary package reconstructed
   byte-for-byte from the locked Git object; no implementation enters consumer
   `src/`.
3. u8, u16 and f16 v2 payloads produce the exact frozen common float32 gain
   hash, common header facts, immutable result dataclasses and read-only arrays.
4. Invalid byte order, wrong endian interpretation, zero dimensions,
   unsupported data type, invalid gamma/bounds, non-finite spacing, truncation
   and trailing bytes all raise `ValueError` before a result is returned.
5. Forward/reverse fresh processes produce byte-identical canonical reports;
   sources remain immutable and owned temporary residue is zero.

Any failed gate closes P310 without modifying thresholds, payloads, parser,
source roles, P257 or the generic DNG loader.

## Claim ceiling

Private no-copy intake of one exact source-locked R1FT PGTM2 storage parser on
three synthetic official Adobe samples only. No interpolation, profile-stage
application, real-file or image quality, arbitrary DNG/map, default-loader,
public package/schema/capability, product mapping, stock evidence, automatic
matching or candidate-3 change.
