# P261 DNG ProfileToneCurve ingress guard contract

## Question

Can the private opt-in P98 DNG ForwardMatrix raster path detect the standard
`ProfileToneCurve` tag before pixel decode, preserve the exact existing
five-DNG outputs when the tag is absent, and fail closed instead of silently
omitting a profile stage it does not implement?

This is a product-integrity leaf. It does not consume or reimplement the
producer R1DX tone-curve arithmetic, reopen any prior DNG render result, or
claim a complete Adobe DNG render pipeline.

## Frozen authority and tag

- Adobe DNG SDK 1.7.1 build 2652 `dng_tag_codes.h` is the tag-code authority.
- Guard exactly code `50940`, `ProfileToneCurve`.
- Presence in any traversed IFD is sufficient to reject. The guard must not
  require a structurally valid curve payload before refusing to ignore it.

## Frozen inputs and information flow

- Use exactly the five already-consumed P98 DNGs and their bound byte hashes.
- Use the unchanged P98 output SHA-256 values as the compatibility oracle.
- Bind the existing P244 HueSatMap and P257 GainTableMap guards and prove their
  tag identities and rejection behavior remain unchanged.
- Read no new DNG, RAW, target, reference or preferred render.
- Build or apply no tone curve.
- The guard runs while TIFF metadata is open and before
  `_decode_camera_linear_dng` can be called.
- Synthetic tag-presence controls exercise code `50940` without constructing
  or decoding image samples.

## Frozen gates

1. Adobe SDK authority resolves `ProfileToneCurve` exactly to code `50940`;
2. all five P98 source sizes and hashes remain exact;
3. all five real files expose no `ProfileToneCurve` tag;
4. the synthetic control rejects and names `ProfileToneCurve(50940)` and its
   IFD path;
5. rejection performs zero camera-raster decode calls;
6. all five P98 WorkingImage float32 pixel SHA-256 values remain exactly equal
   to the frozen P98 oracle;
7. warning codes and WorkingImage colour-state fields remain unchanged;
8. P244 HueSatMap and P257 GainTableMap tag identities and synthetic rejection
   behavior remain unchanged;
9. source bytes remain unchanged before and after execution;
10. canonical and reverse row-order scientific payloads are exact;
11. two fresh-process reports are byte-exact.

Any failure closes the exact guard. Do not apply R1DX arithmetic, infer tone
curve ordering, normalize binaries, relax a replay gate, replace a row, or
rescue with a warning-only policy after scoring.

## Claim ceiling

At most P261 can establish a private opt-in no-silent-drop guard for the exact
Adobe DNG `ProfileToneCurve` tag and prove that five existing P98 files retain
byte-identical scene-linear Rec.2020 outputs when the tag is absent. It cannot
establish tone-curve application, full DNG profile conformance, arbitrary DNG
support, vendor/Adobe rendering parity, photographic or colorimetric quality,
default-loader integration, a public schema/capability, film-stock evidence or
product admission.
