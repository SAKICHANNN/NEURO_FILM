# P262 DNG ProfileLookTable ingress guard contract

## Question

Can the private opt-in P98 DNG ForwardMatrix raster path reject the complete
standard `ProfileLookTable` tag family before pixel decode, preserve the exact
five P98 outputs when the family is absent, and confirm the guard on one
already-retained real DNG carrying the stage?

This is the terminal profile-stage product-integrity guard leaf. It implements
no look-table arithmetic and makes no DNG rendering or image-quality claim.

## Frozen authority, source and tags

- Adobe DNG SDK 1.7.1 build 2652 `dng_tag_codes.h` is the tag-code authority.
- Guard exactly `50981 ProfileLookTableDims`, `50982 ProfileLookTableData` and
  `51108 ProfileLookTableEncoding` in every traversed IFD.
- Bind the already-retained raw.pixls.us Google Pixel 4a DNG, repository row
  4176, 13,586,704 bytes, SHA-256
  `11b38186ee3455376291bcdb3049728a3f9dcd94b9e82ff76855d9f3430e44be`.
- Read only its TIFF metadata; perform zero pixel decode, copy, download or
  render. It must expose dims/data and may omit encoding.

## Frozen compatibility and gates

1. SDK authority resolves all three tag codes exactly;
2. the real source hash/size remains exact and exposes exactly dims/data from
   the guarded family;
3. the public opt-in loader rejects that real source before camera decode;
4. each synthetic single-tag control rejects with exact name/code/IFD and zero
   decode calls; a multi-tag control reports numeric-code order;
5. all five P98 rows are look-table-free and retain exact WorkingImage,
   warning, colour-state and source identities;
6. P244 HueSatMap, P257 GainTableMap and P261 ToneCurve guards remain exact;
7. forward/reverse scientific payloads and two fresh reports are exact.

Any failure closes the exact guard. Do not apply a look table, substitute a
profile, relax ordering or replay gates, replace the real row, or rescue with a
warning-only policy.

## Claim ceiling

At most P262 can establish private predecode rejection for the exact Adobe DNG
`ProfileLookTable` family, confirm it on one retained real DNG, and preserve the
five P98 no-table outputs. It cannot establish look-table application, complete
DNG rendering, arbitrary DNG support, quality, default-loader integration,
package/schema/capability, film-stock evidence or product admission.
