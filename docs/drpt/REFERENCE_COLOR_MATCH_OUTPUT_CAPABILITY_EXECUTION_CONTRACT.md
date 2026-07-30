# P166 — advertised output-capability execution contract

## Question

Does every exact tuple advertised by
`neuro-film.reference-file-output-capabilities.v1` execute through the real
file transaction with its declared rail, bit depth and extension?

## Frozen matrix

The matrix contains all nine advertised extension tuples:

- sRGB 8-bit: `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`;
- sRGB 16-bit: `.png`, `.tif`, `.tiff`;
- BT.2020 SDR 16-bit: `.png`.

Each tuple runs twice at 64-by-48 against product source `f2ea6f7`.
The public capability payload itself must exactly match the frozen inventory.
Output format/depth/profile, output/recipe/normalized-report replay, identity
fallback and zero staging residue are mandatory.

## Boundary

This checks the existing local Windows/Python capability advertisement. It
does not add or validate arbitrary input codecs/profiles, HDR/RAW/video,
metadata preservation beyond the named output profile, target devices,
non-identity quality or product readiness.
