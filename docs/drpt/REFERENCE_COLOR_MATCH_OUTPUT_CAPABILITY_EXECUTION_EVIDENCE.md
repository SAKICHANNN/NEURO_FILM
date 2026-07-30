# P166 — advertised output-capability execution evidence

## Result

The runtime capability payload exactly matches the frozen public v1 inventory.
All nine advertised tuples execute twice:

- sRGB 8-bit: `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`;
- sRGB 16-bit: `.png`, `.tif`, `.tiff`;
- relative BT.2020 SDR 16-bit: `.png`.

All 18 outputs have the declared JPEG/PNG/TIFF format, bit depth and ICC/CICP
profile kind. Within every tuple, output bytes, recipe and normalized report
reproduce exactly. All 18 safety decisions remain `identity-fallback`, with
zero stage/backup/temporary residue.

The canonical capability payload, frozen config, runner and raw report
SHA-256 identities are `c055dc60...3f3a6`, `57f4b914...b0dbb`,
`66fe5be8...5a44f` and `a70dffd0...c0967`.

## Interpretation

`neuro-film.reference-file-output-capabilities.v1` is executable rather than a
declarative superset on the tested local Windows/Python implementation. The
extension aliases `.jpeg` and `.tif` are covered explicitly.

## Boundary

This does not admit arbitrary input codecs or profiles, preserve unspecified
metadata, implement HDR/RAW/video, establish target-device runtime, promote a
non-identity algorithm or prove product readiness.
