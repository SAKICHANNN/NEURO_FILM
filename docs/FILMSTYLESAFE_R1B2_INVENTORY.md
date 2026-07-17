# U5.R1B2 A0 inventory scaffold

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B2`

Decision: **provisional A0 inventory + synthetic operator card scaffold pass**

## Delivered

- `configs/filmstylesafe_r1b2_a0_inventory_v1.json` — seven membership cards covering source, strength controls 53/55/56, ID11 regression, one synthetic failure operator card, and one legitimate-local hard negative
- `scripts/audit_filmstylesafe_r1b2_inventory.py` — fail-closed validation under the R1B contract
- Extended `tests/test_filmstylesafe_r1b.py`

## Explicit limits

Exact/perceptual hashes in this inventory are **placeholder digests** for schema and leakage plumbing. They are not bound to real U4/RF2.C0/ID11 file bytes yet. No pixels were generated. Hidden splits remain empty. No participants or training.

The synthetic operator card `explicit-highlight-chroma-island-v0` is card-only (`implementation_status=card_only_no_pixels`).

## Verification

Inventory audit passes; 7 targeted R1B tests and 386 full CPU tests pass.

## Next

`U5.R1B3`: bind real parent-scene IDs and hashes from existing A0-only artifacts, or implement the first explicit synthetic failure operator under a frozen parameter card—still without hidden splits or recruitment. Parallel product work on remaining U1 colour-state gaps remains legal.
