# P87 Ultra HDR absolute-Rec.2020 MatchView ingress contract

## Question

Can Neuro-Film independently canonicalize the exact public-C-API output of
Google libultrahdr v2.0.0 into a consumer-owned absolute-light Rec.2020
`MatchViewV1`, without changing the existing SDR matcher or inheriting a
producer product claim?

## Authority and fixed source facts

- Producer result: `R1BR_LIBULTRAHDR_V2_MATCH_VIEW_CANONICALIZATION_RESULT`.
- Producer commit: `ce75e9c47db566b31d14b5739413b1edbdca3fd8`.
- Producer evidence SHA-256:
  `85501b21f69cbb6ae8f169749a9b50f57420cfd6fbe3076e8eecf9be0f65eb48`.
- Official decoder: Google `libultrahdr` tag `v2.0.0`, commit
  `b2aacb366e1542cfc29605cb0d8a0ebd06bb07f8`.
- Input payload: little-endian, row-major RGBA binary16; linear BT.2100;
  nominal RGB range `[0, 10000/203]`; alpha must be exactly one.
- Consumer conversion: `float32(binary16 RGB) * float32(203.0)` in cd/m2,
  with no clipping, projection, gamut conversion, resampling or tone map.

The producer profile ID is an external alias only:
`zhuise.display-linear-bt2020-d65-absolute-cdm2-f32.v2`.  The consumer owns a
distinct profile ID:
`neuro-film.display-absolute-linear-rec2020-d65-cdm2.v1`.

## Scope

P87 may:

1. add the absolute Rec.2020 profile to the private `MatchViewV1` colour
   semantics table;
2. validate exact decoder version, source SHA-256, dimensions, payload length,
   finite/range/alpha facts and producer profile alias;
3. emit consumer-owned float32 pixel bytes, pixel hash, provenance hash and
   `MatchViewV1` identity;
4. prove deterministic forward/reverse and two-process replay on fixed
   arithmetic vectors;
5. keep the existing successor/product admission layer rejecting the external
   HDR profile and any unqualified HDR product declaration.

P87 must not:

- decode JPEG/Ultra HDR containers itself;
- accept arbitrary decoder versions, profile aliases, display boosts or white
  levels;
- change `WorkingImage`, SDR file ingress, the current shared-bundle matcher,
  product capabilities, public schemas or delivery defaults;
- infer image quality, mastering metadata, HDR10, display fitness or captured
  HDR truth;
- map R1BR/R1BS/R1BT to product, package, capability or consumer admission.

## Frozen gates

All gates must pass:

1. consumer profile semantics are absolute-linear Rec.2020/D65 and require
   positive finite reference white;
2. exact producer profile alias and decoder version are required;
3. source and decoded-payload SHA-256 values are independently bound;
4. dimensions and byte length are exact before array publication;
5. RGB is finite and within `[0, 10000/203]` and alpha is exactly one;
6. conversion is exact float32 multiplication by 203 with no clipping;
7. canonical output is row-major big-endian float32 RGB and hashes exactly;
8. input bytes remain unchanged;
9. invalid length, version, profile, source hash, nonfinite, range and alpha
   cases publish no result;
10. canonical/reverse sample order and two fresh-process reports are exact;
11. existing SDR MatchView and successor-admission regression tests pass;
12. no public product/capability/default declaration changes.

## Decision rule

- PASS retains only a private consumer ingress primitive and a versioned
  producer/consumer profile bridge candidate.
- Any failed gate closes P87 without changing scale, range, alpha policy,
  decoder version, profile IDs or admission rules.

## Claim ceiling

Private arithmetic and identity compatibility between pinned libultrahdr v2
linear RGBA16F output and a consumer-owned absolute Rec.2020 MatchView only.
No arbitrary-media, HDR quality, HDR10, display, public schema/package,
capability, shared-bundle, product or delivery claim.
