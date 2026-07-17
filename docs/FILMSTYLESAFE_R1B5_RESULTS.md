# U5.R1B5 HF-speckle synthetic and RF2.C0 control binding

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B5`

Decision: **HF-speckle synthetic v1 executed and RF2.C0 Ektar/fixed-e0 bound as A0 external style control**

## Delivered

- Operator: `explicit-highlight-chroma-speckle-v1` in
  `src/eval/filmstylesafe_synthetic.py`
- Frozen config: `configs/filmstylesafe_r1b5_synthetic_operator_v1.json`
- Runner: `scripts/run_filmstylesafe_r1b5_synthetic.py`
- Output (gitignored): `outputs/filmstylesafe/r1b5/a0_synth_speckle_hf_001.png`
- Inventory: `configs/filmstylesafe_r1b5_a0_inventory_v1.json`
- Contract role extension: `external_style_control` in
  `configs/filmstylesafe_r1b_contract_v1.json` / `src/eval/filmstylesafe_r1b.py`

## Evidence hashes

| Field | Value |
|---|---|
| synthetic input_hash (u41-01) | `b551eeccf6ec90c6b6b52e09174705d81360afb0d83734f802bcd17cc43c4bb5` |
| synthetic output_hash | `73af7d4e49c3d28a714f172a974714551360497033eb9c07493f990163012085` |
| synthetic parameter_hash | `0cdb2e526108993a56428086abdbec9edbfcd7f1b879f76912e464bcbe2db4d8` |
| RF2.C0 Ektar/fixed-e0 @01 | `d23ddd7510c137b3116f7f647a83a5fa25675ddb1a6f0d1af22d5c40f35d182e` |

Formal operator applications are deterministic (identical RGB arrays).

## Autonomous visual evidence (not population preference)

**Synthetic HF v1:** Cursor inspected the PNG. Relative to R1B4's solid magenta
disk, v1 produces a circular region of discrete magenta/pink dots over the
upper-right sky/tree highlights. Geometry and prayer-flag structure remain
intact. This is a clearer high-frequency chroma-island stress case, but the
pattern is still denser and more regionally clustered than the fine ID11
highlight speckles. Gates are not lowered.

**RF2.C0 Ektar/fixed-e0 @01:** Cursor inspected the bound control. The Look
Approximation is warm/dense relative to the source, with no confirmed severe
geometry, face, text, banding or neon-speckle failure on this sample. This
reaffirms the RF2.C0 A0-only external-control boundary: not stock truth and not
product integration.

## Membership status

| Member | Status |
|---|---|
| source / 53/55/56 / ID11 / R1B4 solid synth | bound (retained) |
| `a0-synth-speckle-hf-001` | bound (new) |
| `a0-rf2c0-ektar-fixed-e0-01` | bound (new) |
| `a0-hardneg-halation` | still unbound placeholder |

## Claim ceiling

A0 development inventory only. No hidden-split population, participant study,
SCIS training, stock authenticity, product integration of spektrafilm, or
severe-risk guarantee.

## Next

Preferred: bind a legitimate-local hard-negative member, then review whether
the executed A0 suite is leakage-safe enough to open `U5.R1C` planning.
Parallel `U1.2/U1.4/U1.5` colour-state product leaves remain legal. Optional
finer-density speckle ablations may continue without lowering gates.
