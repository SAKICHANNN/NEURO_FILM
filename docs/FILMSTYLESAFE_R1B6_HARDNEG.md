# U5.R1B6 legitimate-local bloom/halation hard-negative

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B6`

Decision: **bounded bloom/halation hard-negative executed and bound; A0 inventory fully bound**

## Delivered

- Operator: `explicit-bounded-bloom-halation-hardneg-v0` wrapping deterministic
  `halation_layer` + screen composite
- Frozen config: `configs/filmstylesafe_r1b6_hardneg_operator_v0.json`
- Runner: `scripts/run_filmstylesafe_r1b6_hardneg.py`
- Output (gitignored): `outputs/filmstylesafe/r1b6/a0_hardneg_halation_001.png`
- Inventory: `configs/filmstylesafe_r1b6_a0_inventory_v1.json` (9/9 bound)

## Evidence hashes

| Field | Value |
|---|---|
| input_hash (u41-01) | `b551eeccf6ec90c6b6b52e09174705d81360afb0d83734f802bcd17cc43c4bb5` |
| output_hash | `87f111a20a6a47823bc4294cf636eb4057db8a106f75f4ce4ede48e83db2514a` |
| parameter_hash | `834a15ce888f0798fa4318684bfa4a37f07565828a691098779904d408236966` |

Formal applications are deterministic. Mean RGB delta vs source is about
`(+0.18, +0.05, +0.00)` with highlight-biased warm red/orange bloom.

## Autonomous visual evidence (not population preference)

Cursor inspected the hard-negative PNG. Geometry, prayer flags and scene
structure remain intact. There is no neon chroma-island or ID11-style red
speckle cluster. The change is a subtle warm highlight bloom consistent with
intentional film-like halation. This member is labeled
`legitimate_local_hard_negative` / `bounded_bloom_or_halation` and must not be
auto-scored as a severe chromatic artifact. Gates are not lowered.

## A0 suite status after R1B6

| Role class | Bound? |
|---|---|
| source / strength / ID11 regression | yes |
| synthetic solid + HF chroma failures | yes |
| RF2.C0 external style control | yes |
| legitimate-local hard-negative | yes (this leaf) |
| unbound placeholders | **none** |

## R1C readiness note

The executed A0 inventory is now leakage-auditable and fully bound under the
frozen R1B contract. This **opens U5.R1C planning** for conventional-metric
failure study / prospective SCIS design on A0 development material.

Still forbidden without new authority: hidden A1/B3 population, external human
recruitment, SCIS training, and any risk/population claim.

## Claim ceiling

A0 hard-negative prototype only. Not stock truth, not product promotion, not a
severe-risk guarantee.

## Next

Preferred: start `U5.R1C` planning/contract freeze for conventional-metric
failure study on the bound A0 suite. Parallel `U1.2/U1.4/U1.5` remain legal.
