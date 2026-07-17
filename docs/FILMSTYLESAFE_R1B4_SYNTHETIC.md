# U5.R1B4 explicit synthetic failure prototype

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B4`

Decision: **first explicit non-generative synthetic failure operator executed**

## Delivered

- Operator: `src/eval/filmstylesafe_synthetic.py` —
  `explicit-highlight-chroma-island-v0`
- Frozen config: `configs/filmstylesafe_r1b4_synthetic_operator_v0.json`
- Runner: `scripts/run_filmstylesafe_r1b4_synthetic.py`
- Output (gitignored): `outputs/filmstylesafe/r1b4/a0_synth_speckle_proto_001.png`
- Bound inventory: `configs/filmstylesafe_r1b4_a0_inventory_v1.json`

## Evidence hashes

| Field | Value |
|---|---|
| input_hash (u41-01) | `b551eeccf6ec90c6b6b52e09174705d81360afb0d83734f802bcd17cc43c4bb5` |
| output_hash | `c04a9237d0023cbeb26924be496997cef82e146f0700e97007c1d64cf45929ad` |
| parameter_hash | `af21e9710ddca57b1ae46da5f42a51046c4c9833ec26359f43c4c2d5e2f3094c` |

Two formal operator applications are deterministic (identical RGB arrays).

## Autonomous visual evidence (not population preference)

Cursor inspected the prototype PNG. On the snowy u41-01 balcony scene the
operator produces a large, highly saturated magenta/pink circular blotch in the
upper-right highlight field rather than fine neon speckles. Geometry and
prayer-flag structure appear otherwise intact. This is useful as an obvious
chromatic severe stress case, but it is **not** yet a faithful ID11-style
speckle surrogate. Refinement of spatial grain/speckle structure is future work;
gates are not lowered after seeing the image.

`visual_adjudication_status`: recorded as autonomous visual evidence for A0
development only.

## Claim ceiling

A0 synthetic stress prototype only. No hidden-split population, participant
study, SCIS training, stock claim, or severe-risk guarantee.

## Verification

391 full CPU tests pass; focused synthetic + R1B tests pass.

## Next

`U5.R1B5`: refine the synthetic family toward speckled/high-frequency chroma
islands and/or bind RF2.C0 Ektar/fixed-e0 as an additional A0 control member.
Parallel U1 colour-state product leaves remain legal.
