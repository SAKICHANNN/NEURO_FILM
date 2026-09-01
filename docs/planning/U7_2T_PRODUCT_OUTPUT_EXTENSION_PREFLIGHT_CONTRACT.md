# U7.2T product output-extension preflight contract

## Question

Does the first-class `--product-look` CLI reject an output extension that the
existing 8-bit encoder cannot publish before it reads the input image or
performs a render?

The encoder supports `.png`, `.jpg`, `.jpeg`, `.tif` and `.tiff`, but the CLI
currently discovers an unsupported 8-bit suffix only when `save_srgb8` runs
after input decode and colour/effect rendering. This is a bounded product
preflight defect. It is not a request to add a format or change pixels.

## Frozen change

- Apply the repair only when `--product-look` selects the existing
  `safe-rich-product-v1` path and `--output-bit-depth 8` is active.
- Before profile loading, transaction setup or input decode, require the output
  suffix case-insensitively to be one of `.png`, `.jpg`, `.jpeg`, `.tif` or
  `.tiff`.
- Reject an absent or unsupported suffix through the argument parser with one
  stable message naming the suffix.
- Keep the existing 16-bit validation and every historical/research invocation
  unchanged. Do not add WebP/AVIF/HEIF or infer a format from content.

## Frozen gates

1. no suffix plus `.webp`, `.avif`, `.heif`, `.bmp` and `.gif` reject with
   parser status `2` before input decode;
2. each rejection names the unsupported suffix, does not disclose/read the
   missing input sentinel and publishes no image, recipe, layers or metrics;
3. `.png`, `.jpg`, `.jpeg`, `.tif` and `.tiff`, including uppercase spellings,
   cross the preflight and reach the deliberately missing input;
4. the same unsupported suffixes in legacy non-product mode retain their
   current late encoder behavior;
5. the exact three U7.2O default product outputs and normalized recipe semantics
   remain unchanged under adjacent regression;
6. focused and parent-chain tests pass and the worktree contains no owned
   runtime residue.

## Claim ceiling and stop rule

A pass establishes only product-only predecode extension validation for the
existing private Look Approximation CLI. It does not add an output format,
package, installer, release, calibrated stock response or physical-film claim.
Any gate failure closes this exact repair without adding formats, changing
encoders, weakening transactions or altering legacy behavior. Stop adjacent
output-option wrapper expansion after closure.
