# U7.2J Product Recipe Catalog Enforcement Contract

## Confirmed defect

U7.2I enforces the authoritative product look catalog at the CLI, but the
shared recipe builder and verifier still authorize any style embedded in a
profile.  A caller can therefore construct or replay a
`safe-rich-product-v1` recipe for historical `hp5`, `tri_x_400`, Vision3 or
Portra 800 styles even though those identities are absent from the available
product catalog.

## Frozen repair

- Centralize one private product-profile style authorization check in the
  recipe contract module.
- Apply it before recipe construction and before recipe input-file hashing or
  decoding during verification/replay.
- Source the decision from the authoritative product look catalog; do not copy
  an independent allow-list.
- Reject non-catalog and unavailable rows with `RenderContractError`.
- Preserve exact recipe/replay behavior for Velvia 50, Portra 400 and Ektar
  100 under the product profile.
- Preserve historical HP5 and other internal styles under non-product profiles.

## Stop rule and claim ceiling

After builder and verifier/replay enforcement plus behavioral regression are
complete, stop this selection-authorization family.  No colour parameter,
pixel algorithm, stock evidence, calibration, separation, public schema or
release state may change.
