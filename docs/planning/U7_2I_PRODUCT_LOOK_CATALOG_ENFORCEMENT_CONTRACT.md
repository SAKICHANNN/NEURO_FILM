# U7.2I Product Look Catalog Enforcement Contract

## Defect

`safe-rich-product-v1` requires an explicit `--style`, but the CLI currently
accepts any style embedded in the profile.  The profile also carries historical
research/replay styles (`hp5`, `tri_x_400`, `vision3_250d`, `vision3_500t`,
`portra_800`).  Those names are absent from the authoritative product catalog,
yet they reach input decoding instead of failing at product selection.

## Frozen change

- When and only when `profile_id == "safe-rich-product-v1"`, require the
  explicit style to be an authoritative product-catalog row whose availability
  is `available`.
- Reject catalog-blocked `generic_bw`, historical named B&W, Vision3,
  Portra 800, unknown and empty styles before input decoding or output creation.
- Preserve the historical non-product profiles and their omitted Velvia default.
- Preserve exact bytes and normalized recipe semantics for the three available
  colour Look Approximations: Velvia 50, Portra 400 and Ektar 100.

## Stop rule and claim ceiling

This leaf changes selection authorization only.  It does not alter colour
parameters, stock evidence, calibration, separation, algorithms, profiles,
schemas or release status.  After the catalog gate and regression evidence are
complete, stop adjacent selection-wrapper expansion.
