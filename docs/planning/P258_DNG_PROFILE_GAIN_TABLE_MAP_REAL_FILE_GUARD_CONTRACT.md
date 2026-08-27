# P258 DNG ProfileGainTableMap real-file guard contract

## Question

Does the unchanged P257 no-silent-drop guard reject one exact real CC0 Samsung
Galaxy S21 Ultra DNG carrying `ProfileGainTableMap` v1 before any camera-raster
decode, while preserving the source bytes and deterministic diagnostics?

This is a real-file safety confirmation. It does not decode LinearRaw samples,
apply a gain table, reproduce producer R1EB/R1EG, or expand the P98-supported
DNG cohort.

## Frozen source and authority

- Reuse the already-retained P7H source at
  `data/external/rawpixls_p7h_v1/raw/samsung_galaxy_s21_ultra.dng`.
- Source identity is frozen at 37,770,480 bytes and SHA-256
  `5baeb5f0c4a125647df03b82c2d38b0f02784393bd39eff1a5b23d8f5308dc5e`.
- The exact P7H source row records raw.pixls.us repository id 6728, its
  source URL and CC0 repository binding; no network request or new copy is
  allowed in P258.
- Adobe DNG SDK 1.7.1 build 2652 `dng_tag_codes.h` remains the tag authority:
  `ProfileGainTableMap=52525`, `ProfileGainTableMap2=52544`.
- Producer R1EB is only a cross-project provenance fact that independently
  identified this same source hash as a Samsung LinearRaw/PGTM file; no
  producer implementation or pixel result is consumed.

## Frozen information flow

1. Bind the exact P257 config/evidence/current guard source and exact P7H
   source row before reading the DNG metadata.
2. Traverse TIFF IFD metadata only and require exactly `52525` among the P257
   guarded family, with `52544` absent.
3. Patch only `_decode_camera_linear_dng` with a counting failure sentinel and
   call the unchanged public P98 opt-in loader.
4. Require `DngForwardRasterError` to name every actual guarded tag and report
   its IFD path before the sentinel can run.
5. Hash the source again after the call. Persist no pixel, preview, metadata
   extract, DNG copy or media artifact.

## Frozen gates

1. P7H config/source path/size/hash and CC0/source URL row are exact;
2. P257 config/evidence/implementation identities are exact;
3. SDK tag codes resolve exactly to `52525` and `52544`;
4. the real file exposes exactly `ProfileGainTableMap(52525)` from the guarded
   family and no `ProfileGainTableMap2`;
5. the rejection diagnostic names every actual guarded tag and IFD path;
6. camera-raster decode calls equal zero;
7. source bytes remain unchanged;
8. network requests, copied files, partial files and pixel artifacts equal
   zero;
9. forward/reverse scientific payloads are exact;
10. two fresh-process complete reports are byte-exact.

Any failure closes P258 without downloading a replacement, invoking a mirror,
decoding pixels, applying R1DY/R1EB arithmetic, changing P257, relaxing a gate
or adding another PGTM file.

## Claim ceiling

At most P258 can confirm the private P257 predecode refusal on one exact real
CC0 Samsung DNG carrying `ProfileGainTableMap` v1. It cannot establish gain-map
application, LinearRaw decoding, Adobe renderer parity, full or arbitrary DNG
support, photographic/colorimetric quality, default-loader integration,
package/schema/capability, film-stock evidence or product admission. This leaf
ends the PGTM guard confirmation family.
