# P257 DNG ProfileGainTableMap ingress guard contract

## Question

Can the private opt-in P98 DNG ForwardMatrix raster path detect the standard
`ProfileGainTableMap` v1/v2 tags before pixel decode, preserve the exact
existing five-DNG outputs when those tags are absent, and fail closed instead
of silently omitting a profile stage it does not implement?

This is a product-integrity leaf. It does not apply the producer R1DY gain-map
arithmetic, reopen any prior DNG render result, or claim a complete Adobe DNG
render pipeline.

## Frozen authority and tag family

- Adobe DNG SDK 1.7.1 build 2652 `dng_tag_codes.h` is the tag-code authority.
- Guard exactly these codes:
  - `52525` `ProfileGainTableMap` (DNG 1.6, Raw IFD);
  - `52544` `ProfileGainTableMap2` (DNG 1.7, IFD0; supersedes v1).
- Presence of either tag is sufficient to reject. The guard must not require a
  valid table payload or both versions before refusing to ignore the stage.

## Frozen inputs and information flow

- Use exactly the five already-consumed P98 DNGs and their bound byte hashes.
- Use the unchanged P98 output SHA-256 values as the compatibility oracle.
- Bind the existing P244 HueSatMap guard and prove its six tag identities and
  rejection behavior remain unchanged.
- Read no new DNG, RAW, target, reference or preferred render.
- Build no gain table and apply no ProfileGainTableMap.
- The guard runs while TIFF metadata is open and before
  `_decode_camera_linear_dng` can be called.
- Synthetic tag-presence controls exercise both codes without constructing or
  decoding image samples.

## Frozen gates

1. Adobe SDK authority resolves exactly codes `52525` and `52544`;
2. all five P98 source sizes and hashes remain exact;
3. all five real files expose neither guarded tag;
4. each single-tag synthetic control rejects and names the exact tag;
5. a two-tag control reports both tags in numeric-code order;
6. every rejection path performs zero camera-raster decode calls;
7. all five P98 WorkingImage float32 pixel SHA-256 values remain exactly equal
   to the frozen P98 oracle;
8. warning codes and WorkingImage colour-state fields remain unchanged;
9. P244 HueSatMap tag identities and synthetic rejection behavior remain
   unchanged;
10. source bytes remain unchanged before and after execution;
11. canonical and reverse row-order scientific payloads are exact;
12. two fresh-process reports are byte-exact.

Any failure closes the exact guard. Do not apply R1DY arithmetic, infer gain
table ordering, normalize binaries, relax a replay gate, replace a row, or
rescue with a warning-only policy after scoring.

## Claim ceiling

At most P257 can establish a private opt-in no-silent-drop guard for the exact
Adobe DNG `ProfileGainTableMap` v1/v2 tags and prove that five existing P98
files retain byte-identical scene-linear Rec.2020 outputs when those tags are
absent. It cannot establish gain-map application, full DNG profile conformance,
arbitrary DNG support, vendor/Adobe rendering parity, photographic or
colorimetric quality, default-loader integration, a public schema/capability,
film-stock evidence or product admission.
