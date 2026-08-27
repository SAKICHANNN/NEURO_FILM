# P284 libavif gain-map one-way PQ PNG contract

## Question

Can the exact three official libavif higher-headroom RGB16 renditions that pass
the forward P283/P87 boundary be published through the unchanged P89 BT.2100
PQ RGB16 PNG rail with exact signaling, decoded-sample identity, bounded
absolute-light quantization error and atomic behavior?

P284 is frozen after P283 failed its inverse gamut-domain gate and before the
first P284 output. It is intentionally one-way. It neither requires nor claims
roundtrip recovery of the original BT.709/PQ samples and therefore cannot
rescue or promote P283.

## Frozen inputs and transform

- Rebuild the exact P282 base or alternate-headroom rendition for each of the
  three P278 official fixtures; retained decoded images are never authorities.
- Validate exact CICP `[1,16,6]`, source/runtime hashes and the P283 forward
  adapter into the existing absolute-linear Rec.2020/D65 P87 profile.
- Publish those values unchanged through
  `save_absolute_rec2020_cdm2_to_pq_rgb16_png` using the P89/U1.4G fixed
  BT.2100-3 float64 formula, half-up RGB16 quantizer and `09 10 00 01` cICP.
- Decode the exact encoded samples through the inverse PQ diagnostic only to
  measure absolute-light quantization error. No gamut conversion, clipping,
  tone mapping, metadata synthesis or source-domain roundtrip is allowed.

## Frozen gates

1. P282, P283, P89 and U1.4G evidence/config/source/runtime hashes match;
2. all three rebuilt samples and exact higher-rendition roles match P283;
3. every P87 descriptor, ownership, range and input-immutability gate passes;
4. the publisher-returned RGB16 samples equal strict PNG readback and cICP is
   exactly `09 10 00 01`;
5. per-component unquantized code error is at most `0.5/65535 + 1e-12`;
6. decoded absolute-light maximum error is at most `0.07` cd/m2 for this frozen
   below-1000-nit cohort;
7. output is create-only, repeated bytes are exact and invalid/nonfinite/
   wrong-profile controls fail before publication;
8. source/runtime files remain immutable, formal outputs leave no scratch and
   forward/reverse normalized reports are byte exact.

Any failure closes P284 without changing inputs, profile, transform, quantizer,
error ceilings or container. In particular, clipping or a BT.709 inverse is
forbidden.

## Claim ceiling

Private exact one-way publication mechanics for three frozen official libavif
gain-map higher-headroom renditions through P87 and the existing P89 PQ RGB16
PNG rail only. No P283 roundtrip rescue, complete ISO 21496-1, arbitrary
AVIF/HEIF, HDR reconstruction quality, display validation, mastering/content-
light metadata, public API/dependency/package/schema/capability/product,
stock claim or candidate3 change.
