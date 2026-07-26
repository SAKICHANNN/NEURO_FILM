# SF2.9R ColorReference multi-family IT8 reference contract

Date frozen: 2026-07-26

Node: `ULT > RF0.4 > SF2.9R`

Status: frozen before measurement acquisition

## Question

Can five publicly downloadable IT8/ISO 12641 transmission-target reference
archives provide scanner-independent, content-free evidence useful to the
stock-first programme?

This is a source/schema and measurement-feasibility audit. It is not a
stock-response fit, a training corpus or permission to invert target
measurements into a photographic look.

## Why this leaf is discriminating

The source provides direct measured target reference data rather than
community photographs or scanner RGB. It may therefore remove scene content
and scanner software as shortcuts. However, IT8 targets are manufactured for
scanner calibration: their production may deliberately drive different film
materials toward common aim colours. A classifier or large colour difference
could then measure target manufacture, batch, dye family or process residuals,
not the appearance operator of a camera-exposed stock.

The audit must preserve that negative interpretation and may formally return
`useful_physical_prior_only` or `target_calibration_confounded`.

## Frozen lane

Acquire exactly the five latest public family archives visible on the source
page:

| Archive | Declared target family | Bytes |
|---|---|---:|
| `E240220.zip` | Ektachrome K3 family; exact stock unknown | 293,776 |
| `V240219.zip` | Fujichrome Velvia RVP 50 V3 family | 293,115 |
| `N230513.zip` | mixed newer Fujichrome N3-compatible family | 294,186 |
| `A240301.zip` | Agfa RSX/CT Precisa A3 family | 292,423 |
| `F240222.zip` | mixed older Fujichrome F3-compatible family | 294,516 |

Expected total is exactly 1,468,016 bytes. The hard transfer ceiling is 8 MiB.
No scan image, product purchase, email request or unlisted batch is allowed in
this leaf.

## Required audit

1. Verify HTTP status, frozen byte count, hash and ZIP CRC.
2. Inventory every member and parse the reference formats deterministically.
3. Record patch identifiers, measurement condition, colourimetric/density
   fields, spectral wavelengths and units.
4. Measure common-patch connectivity without treating row order as identity.
5. Determine whether each archive gives exact stock, emulsion/batch, process
   and target-production evidence.
6. Determine whether the data are common-input responses or calibrated target
   outputs designed around common IT8 aims.
7. Separate possible uses:
   scanner/profile test, film spectral prior, batch/process nuisance,
   stock-family evidence, or operator fitting.

## Gates and branches

- CRC, parse, common-patch support or spectral-support failure closes the leaf.
- Mixed-family labels cannot be upgraded to exact stocks.
- A common patch ID is not a common scene exposure unless the production input
  and recorder/process chain are identified.
- Deliberate aim-value calibration closes stock-style/operator fitting even if
  family labels are numerically separable.
- Lack of a standard data licence keeps use internal and forbids
  redistribution/training.
- Only an independently lineaged common-input exposure design with exact stock
  and rights evidence could open a separately frozen operator leaf.

## Claim ceiling

At most this leaf may establish bounded internal schema,
spectral-measurement, batch/family and target-manufacturing-nuisance evidence.
It cannot establish a camera-scene pair, exact multi-stock response, stock
style, calibrated simulation, operator fitting permission, latent mode or
production evidence.
