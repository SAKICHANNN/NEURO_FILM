# U7.2O first-class product Look Approximation CLI contract

## Question

Can a user select one of the three available product Look Approximations with
one explicit CLI option, without knowing an internal profile path or combining
legacy flags, while producing exactly the same image, recipe and transaction
semantics as the current `safe-rich-product-v1` path?

The authoritative catalog and renderer already exist, but the documented CLI
still demonstrates the historical compatibility path. A real product render
currently requires the user to discover and combine `--use-render-profile`,
the internal `safe_rich_product_v1.json` path and `--style`. This is a product
entry-flow defect, not a colour-algorithm or film-stock experiment.

## Frozen change

- Add `--product-look` with exactly the currently available product look IDs:
  `velvia_50`, `portra_400` and `ektar_100`.
- Resolve it before any input decode to the unchanged `safe_lab` engine, exact
  repository `safe-rich-product-v1` profile, `--use-render-profile` semantics
  and matching internal style ID.
- Reject combinations with `--style`, `--use-render-profile`, an explicitly
  supplied `--render-profile`, a non-safe-Lab engine, or
  `--list-product-looks`. Do not silently choose between competing controls.
- Preserve all existing product validations, bounded `--look-amount`, recipe,
  layer, metrics and create-only bundle behavior. Do not add a new profile,
  schema, output format or renderer.
- Update the README primary quick start to use `--product-look`, describe all
  three names as film-inspired / Look Approximation, and retain the exact
  calibrated-stock and physical-film claim prohibition.

## Frozen gates

1. all three product look IDs at amounts `0`, `.5` and `1` produce byte-exact
   images and normalized recipe semantics versus the pre-existing explicit
   `--use-render-profile --render-profile ... --style ...` invocation;
2. recipe profile identity remains `safe-rich-product-v1`, the selected style
   is exact, and the output claim remains Look Approximation only;
3. image-only and full recipe/layers/metrics publication retain U7.2N
   create-only and rollback behavior;
4. conflicting product/legacy/profile/list/engine controls reject before
   `load_working_image` and publish no artifact;
5. unavailable `generic_bw`, historical styles, unknown and empty values are
   not accepted as `--product-look` values;
6. legacy CLI invocations and `--list-product-looks` bytes remain unchanged;
7. the README quick-start command is executable and names the claim boundary;
8. forward/reverse fresh-process reports are byte-identical and owned residue
   is zero.

## Claim ceiling and stop rule

A pass establishes only a first-class private CLI entry for the existing
deterministic film-inspired / Look Approximation product path. It does not
establish calibrated stock response, physical-film reproduction, stock
separation, population preference, a public package/installer, release
readiness or cross-platform parity. Any failure closes this exact CLI entry;
do not rescue it by changing pixels, profiles, catalog availability, output
claims, amount semantics, recipes or transaction gates.
