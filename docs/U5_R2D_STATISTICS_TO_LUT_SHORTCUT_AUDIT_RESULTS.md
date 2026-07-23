# U5.R2D statistics-to-LUT shortcut audit results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D`  
Decision: **properties pass; operator-identification shortcut rejected**

## Result

The isolated paper-compatible Lab descriptor passes every frozen numerical
property. This is not evidence that it identifies a film operator.

| Frozen check | Result | Gate |
|---|---:|---:|
| Descriptor length | 2,304 | exactly 2,304 |
| Lightness histogram mass error | 0 | <= 1e-12 |
| Raw chroma histogram mass error | 0 | <= 1e-12 |
| Pixel-permutation maximum error | 4.2633e-14 | <= 1e-12 |
| Red-versus-blue palette descriptor L2 | 111.6922 | > 0.1 risk witness |
| Two-LUT reference pixel error | 0 | <= 1e-12 |
| Two-LUT reference descriptor error | 0 | <= 1e-12 |
| Absent-blue probe RGB L2 | 1.0 | >= 0.1 counterexample |

The two audit runs are byte-identical at SHA-256
`7622e4a5c15374fb7c9095016c391b18cc9fd98ffd645dd8a3545e1db6928d58`.
The frozen config SHA-256 is
`4de4f3351cee383998319c3b0c54c1c4858a65888884c51c0947329e4d5a8283`
and the audited implementation commit is
`3b2a5026bcf1c0d4c25f94c2ee3074dddda976a9`.

## Interpretation

The representation is spatially agnostic in the narrow, testable sense that
permuting the same pixels does not materially change its output. It is not
content- or palette-agnostic: equal-size constant red and blue scenes are very
far apart in descriptor space.

More importantly, the same reference image and exactly the same reference
descriptor are compatible with two explicit LUTs that disagree strongly on a
colour absent from the reference. Therefore, a single unpaired scan cannot
uniquely identify a global digital-to-film operator from these statistics.
The descriptor may still be useful as:

- a content/palette-sensitive matching control;
- one input to a synthetic known-operator recovery experiment;
- a nuisance feature that a proposed film method must beat.

It must not be described as recovered stock response, paired evidence,
calibration or proof of semantic decoupling.

## Verification

- 10 descriptor tests pass, including an independent two-pixel lightness-bin
  reference;
- 16 focused descriptor/constrained-LUT tests pass;
- two formal audit runs are byte-identical;
- 726 complete CPU tests pass in 58.88 seconds;
- invalid range, shape, empty and non-finite inputs fail closed;
- no image, LUT pack, pretrained weight or stock pixel was downloaded.

## Branch

U5.R2D closes the direct statistics-to-operator-identification shortcut but
does not close bounded statistics-conditioned explicit operators. U5.R2D1 may
open only as a separately frozen, CPU-scale synthetic benchmark where the
ground-truth bounded operators are generated locally and known exactly.

R2D1 must include descriptor-only, source-content-only,
source-plus-target/difference, best-basic and identity controls. It must split
by operator family and palette support, evaluate held-out colour probes, and
report artifact/constraint failures independently from parameter or RGB
error. Real-film fitting, stock learning, LSM and production integration
remain forbidden.

The Ultimate Goal remains ACTIVE.
