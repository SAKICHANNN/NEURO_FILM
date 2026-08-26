# P234 R1CZ UltraHDR still-safety contract

## Question

Does the exact locally persisted R1CZ paired/capture-time payload remain
boundary-safe and spatially non-amplifying when applied source-free to the two
exact already-consumed P88 UltraHDR absolute-Rec.2020 fixtures?

This is a bounded safety audit, not a new matching algorithm or quality claim.
It does not reopen the closed after-only candidate cycle.

## Frozen inputs and information flow

- Exact P233 capsule SHA `66050f58...859e`, decoded payload SHA
  `aa7fe040...28cb9` and bundle `2759d219...c3220f`.
- Exact P88 `apple_gainmap_new.jpg` and `apple_gainmap_old.jpg`, both
  `384x512`, under the retained CC-BY-4.0 source manifest.
- Exact official libultrahdr v2.0.0 commit `b2aacb36...07f8`; the decoder path
  is supplied at execution time and its binary SHA must be
  `cffdfcae...d5df0`.
- Exact P87 absolute Rec.2020/D65 cd/m2 ingress and 203-nit reference white.
- Bundle construction reads only the persisted payload. Application sources
  are decoded only after every identity and gate is frozen. No target image,
  paired reference, paired target, fit, network or repository media output is
  allowed.

Both image rows are mandatory. Reversing their execution order is the only
formal replay permutation.

## Frozen measurements and gates

For each row, compare the source and candidate in absolute Rec.2020 cd/m2:

1. output is finite and remains in `[0,10000]`;
2. no strictly interior source sample becomes exact 0 or 10000;
3. median per-pixel `L2(log1p(output)-log1p(source))` is at least `.01`;
4. horizontal-plus-vertical BT.2020 log-luminance spatial-gradient p95 ratio is
   at most `1.25`;
5. corresponding log-chroma spatial-gradient p95 ratio is at most `1.25`;
6. at most `.005` of spatial luminance edges exceed the source edge by more
   than `.25` log2 units;
7. every fixture passes every safety gate;
8. forward/reverse scientific payloads are exact;
9. payload-build application-source reads, target reads, network reads and
   repository media writes are zero.

Any failure closes this exact payload/fixture safety application. Do not alter
payload strength, decoder, fixtures, metric definitions or thresholds after
observing output.

## Claim ceiling

At most, P234 can retain a private target-blind safety D0 on two exact consumed
CC-BY-4.0 UltraHDR fixtures. It cannot establish grade correctness, aesthetic
quality, captured HDR truth, arbitrary-media support, a public package/schema/
capability or product admission.
