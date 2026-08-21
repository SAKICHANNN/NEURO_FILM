# P89 Ultra HDR absolute Rec.2020 to PQ PNG contract

## Question

Can the exact private P88 decoder-to-P87 chain convert its absolute linear
Rec.2020 values in cd/m2 to full-range BT.2100 PQ RGB16 samples and publish
them through the unchanged U1.4G PNG rail, without tone mapping, clipping,
invented mastering metadata or product admission?

P89 is frozen after P88 consumed the two fixtures and before the first P89 PQ
transform execution. It is a retrospective standards/mechanics integration
leaf, not fresh HDR-quality evidence.

## Standards and fixed arithmetic

- ITU-R BT.2100-3 (02/2025), Table 4 reference PQ EOTF and its inverse:
  <https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en>.
- PNG Third Edition (2025), full-range BT.2100 PQ `cICP` example
  `09 10 00 01`: <https://www.w3.org/TR/2025/REC-png-3-20250624/>.
- Input is the P87 profile
  `neuro-film.display-absolute-linear-rec2020-d65-cdm2.v1` and must be finite
  and inside `[0, 10000]` cd/m2. Values outside that interval are rejected;
  they are never clipped or tone mapped.
- For `Y = cd_m2 / 10000`, encode each RGB component independently as
  `E = ((c1 + c2*Y**m1) / (1 + c3*Y**m1))**m2`, using float64 and the exact
  BT.2100 constants `m1=2610/16384`, `m2=(2523/4096)*128`,
  `c1=3424/4096`, `c2=(2413/4096)*32`, `c3=(2392/4096)*32`.
- Quantize with `floor(E*65535 + 0.5)` to full-range RGB16. No dither,
  chromatic adaptation, gamut mapping, scaling or metadata synthesis is
  allowed.
- The inverse reference EOTF is evaluated in float64 for diagnostics and
  arithmetic validation only; it does not alter samples.

## Frozen inputs

- Exact P88 configuration and decoder executable, command and two mandatory
  CC-BY-4.0 fixtures remain unchanged.
- P87, P88 and U1.4G evidence files are hash-bound in the executable config.
- Every P89 run invokes the pinned decoder afresh, validates P87 afresh, then
  performs the fixed PQ transform and U1.4G publication. No retained decoded
  pixel file is an input authority.

## Frozen gates

1. all bound evidence/config/source/executable hashes match before execution;
2. both decoded payloads pass P87 unchanged and expose the exact absolute
   Rec.2020 profile;
3. scalar endpoint codes are exactly black `0` and 10000-nit white `65535`,
   and a fixed dense scalar probe is finite and monotone;
4. the vectorized output matches an independently written scalar float64
   oracle exactly on the fixed probe and both fixtures;
5. every unquantized code differs from its RGB16 code by at most
   `0.5/65535 + 1e-12`;
6. strict U1.4G readback preserves every native RGB16 sample and exact
   `09 10 00 01` signaling;
7. decoded-PQ diagnostic values are finite and stay inside `[0,10000]`;
8. source fixtures remain unchanged and output publication is create-only and
   failure-atomic;
9. nonfinite, negative, above-10000, wrong-shape and wrong-profile inputs fail
   before publication;
10. the existing production loader continues to reject both original gain-map
    JPEGs and the generated PQ PNGs;
11. fixture-order reversal and two fresh-process normalized scientific reports
    are byte exact;
12. no public package/schema/capability/default/product declaration changes.

Any failed gate closes P89 without changing the formula, constants,
quantizer, fixture set, decoder, metadata, range or loader boundary.

## Claim ceiling

Private exact two-consumed-fixture Windows pipeline compatibility from one
pinned libultrahdr v2.0.0 decoder through P87 absolute Rec.2020, the fixed
BT.2100-3 PQ inverse EOTF and the existing U1.4G RGB16 PNG rail only. No tone
mapping, arbitrary Ultra HDR/ISO 21496-1 support, HDR10/mastering/content-light
metadata, display validation, HDR quality, public package/schema/capability,
default loader, product or delivery admission.
