# U5.R2AH1D Group-Invariant Reference Operator Development Results

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AH1D`  
Decision: `close_frozen_development_gate_failure`

## Outcome

The preregistered hierarchical Deep Sets predictor does **not** recover a
usable reference-only operator on the generated development population.
Two independent local-CUDA processes produce byte-identical reports at
SHA-256 `c49817dfaae3aba84efe2c6da72707320ccfe760308350b39743e806dfc984d1`.
The formal decision repeats byte-identically at
`3d0cd45c6371a62a0e1360dae1ad18abb0bdce47d69e0e95ca3a7fc4497f608d`.

This closes AH1 without width, depth, step-count, loss, adversary, seed,
threshold, calibration or post-hoc strength rescue. The untouched AH1C
different-family confirmation and all W1 reserved seeds remain unread.

## Fixed execution

| Item | Frozen value / observed fact |
|---|---|
| Software commit | `1dce72949ca98db73126991328969feebe911fa9` |
| Config SHA-256 | `9b1128700b0e1c4f0312d5e8c6762ff662828efee39c3e5c319cf9eddcaf9ab6` |
| Model | 59,020-parameter hierarchical Deep Sets |
| Renderer | deterministic bounded O0, never neural final RGB |
| Training | deterministic CUDA float32 AdamW, seed 30131, 5,000 steps |
| Hardware | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Run A / Run B | 1,551.2 s / 1,463.2 s |
| Parameter-state SHA-256 | `582d4dd0a13d80e9b67cdcdae2abd8ae9c5382bff5f87225dfcbf183f94d83a1` |
| Training population SHA-256 | `7af6dcd4d8922db5f1f685824f3b68cec3662bf051d7274d9c52fd0b89c6aede` |
| Development population SHA-256 | `e19bcff89ac24e168e73e55ba81ee78cb93225c498bbd0128c5c5d45401d0682` |

The reports are ignored runtime evidence under
`outputs/u5_r2ah1_group_invariant_reference_operator_development_v1/`.
The machine-readable adjudication is
`configs/u5_r2ah1_group_invariant_reference_operator_development_decision_v1.json`.

## Frozen gates

| Gate | Result | Frozen requirement | Decision |
|---|---:|---:|---|
| Four-reference operator RMSE median | 0.04001 | <= 0.05 | pass |
| Four-reference operator RMSE p90 | 0.09723 | <= 0.08 | **fail** |
| Median improvement over identity | 7.25% | >= 25% | **fail** |
| Median improvement over global mean | 4.89% | >= 25% | **fail** |
| Median improvement over W1 ridge | 39.66% | >= 10% | pass |
| Same-look replicate RMSE median | 0.01004 | <= 0.04 | pass |
| Identity-reference RMSE median | 0.00833 | <= 0.01 | pass |
| Content probe BA, worst | 27.10% | <= 22.5% | **fail** |
| Nuisance probe BA, worst | 25.64% | <= 35% | pass |
| Fixed-content different-look accuracy | 12.5% | >= 75% | **fail** |
| 53/55/56 minimum direction cosine | 0.98356 | >= 0.95 | pass component |
| 53/55/56 strength Spearman | -1.0 | >= 0.9 | **fail** |
| Output range | [0.05957, 0.94129] | within [0, 1] | pass |
| Minimum sampled Jacobian determinant | 0.14513 | >= 0.02 | pass |
| Maximum sampled Jacobian norm | 2.03263 | <= 8 | pass |
| Inverse maximum error | 3.35e-8 | <= 2e-5 | pass |
| Serialization / set permutation | exact / exact | exact / exact | pass |

## Interpretation

The model is structurally safe because it only predicts bounded O0 parameters,
and its nuisance suppression, repeatability, identity rejection and
same-operator replication pass. Those facts do not establish operator
recovery.

The recovery sits close to the global and identity baselines, retains
decodable content, cannot distinguish looks when content is fixed, and
reverses the intended continuous strength ordering of the `53/55/56` negative
control. This is an averaged shortcut representation, not evidence that a
larger reference encoder is warranted.

AH1 is stronger negative evidence than W1 because it fails even with exact
synthetic same-operator groups, independently crossed content/nuisance and
known O0 truth. It therefore forbids treating arbitrary final photographs as
enough supervision to identify their generating colour operator.

## Claim ceiling and branch

This is generated-development evidence only. It is not a real reference-only
operator, digital-to-film identification, film-stock model, latent mode,
calibration, authenticity, preference or product result. No photograph,
current film pixel, visual stage or production integration opens.

O0 remains independently retained as the safe explicit renderer. Ultimate
remains active and must select a genuinely different, evidence-authorized
algorithm or data leaf rather than tuning AH1.
