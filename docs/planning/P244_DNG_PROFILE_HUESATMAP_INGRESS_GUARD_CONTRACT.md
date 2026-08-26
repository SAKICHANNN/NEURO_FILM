# P244 DNG ProfileHueSatMap ingress guard contract

## Question

Can the private opt-in P98 DNG ForwardMatrix raster path detect every standard
`ProfileHueSatMap`-family tag before pixel decode, preserve the exact existing
five-DNG outputs when those tags are absent, and fail closed instead of
silently omitting a render profile it does not yet implement?

This is a product-integrity leaf. It does not reopen P243's failed fresh-build
identity gate and does not claim that the HueSatMap operator, profile-table
interpolation, or a complete Adobe DNG render pipeline is integrated.

## Frozen authority and tag family

- Adobe DNG SDK 1.7.1 build 2652 `dng_tag_codes.h` is the tag-code authority.
- Guard exactly these codes:
  - `50937` `ProfileHueSatMapDims`;
  - `50938` `ProfileHueSatMapData1`;
  - `50939` `ProfileHueSatMapData2`;
  - `51107` `ProfileHueSatMapEncoding`;
  - `52537` `ProfileHueSatMapData3`;
  - `52551` `ProfileDynamicRange`.
- Presence of any one guarded tag is sufficient to reject. The guard must not
  require a complete or internally consistent table before refusing to ignore
  it.

## Frozen inputs and information flow

- Use exactly the five already-consumed P98 DNGs and their bound byte hashes.
- Use the unchanged P98 output SHA-256 values as the compatibility oracle.
- Read no new DNG, RAW, target, reference or preferred render.
- Build no LUT and apply no HueSatMap.
- The guard runs while TIFF metadata is open and before
  `_decode_camera_linear_dng` can be called.
- Synthetic tag-presence controls exercise all six codes without constructing
  or decoding image samples.

## Frozen gates

1. all five P98 source sizes and hashes remain exact;
2. all five real files expose none of the six guarded tags;
3. every single-tag synthetic control rejects and names the exact tag;
4. a multi-tag control reports all present tags in numeric-code order;
5. the rejection path performs zero camera-raster decode calls;
6. all five P98 WorkingImage float32 pixel SHA-256 values remain exactly equal
   to the frozen P98 oracle;
7. warning codes and WorkingImage colour-state fields remain unchanged;
8. source bytes remain unchanged before and after execution;
9. canonical and reverse row-order scientific payloads are exact;
10. two fresh-process reports are byte-exact.

Any failure closes the exact guard. Do not apply the P243 arithmetic directly
to linear Rec.2020, infer a ProPhoto conversion, choose/interpolate Data1/2/3,
normalize binaries, relax a replay gate, replace a row, or rescue with a
warning-only policy after scoring.

## Claim ceiling

At most P244 can establish a private opt-in no-silent-drop guard for the exact
Adobe DNG ProfileHueSatMap tag family and prove that five existing P98 files
retain byte-identical scene-linear Rec.2020 outputs when that family is absent.
It cannot establish HueSatMap application, full DNG profile conformance,
arbitrary DNG support, vendor/Adobe rendering parity, photographic or
colorimetric quality, default-loader integration, a public schema/capability,
film-stock evidence or product admission.
