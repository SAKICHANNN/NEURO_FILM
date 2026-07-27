# U5.R2Z0 FilmSet Within-Recipe Case-Retrieval Results

Date: 2026-07-27

Decision: **ClassNeg Oracle only; Velvia no qualifying Oracle**

## Reproducibility and boundary

- software commit:
  `c5b7c04f775b4be0db1a4281278e2a5b6c3c5c23`;
- config SHA-256:
  `fd689811440d824f92f99580461c039002942d5393478168ca8e40288679cd88`;
- two complete reports are byte-identical at
  `ffec19f4bbc5068014deef448bf272c10d279b4e869818d831a34be699d7c6dc`;
- both stderr logs are empty;
- the local CUDA device is the NVIDIA GeForce RTX 5070 Ti Laptop GPU with
  PyTorch `2.11.0+cu128`;
- 24 development and 16 confirmatory identities were used;
- exactly 120 aligned input/ClassNeg/Velvia payloads were hash checked;
- no unselected payload was read, the final 628 manifest contributed zero
  parsed payload rows, and no full-raster output or visual shortlist was
  generated.

Each run fits one shared and 24 per-case bounded O0 operators for each recipe,
for 50 fits total. Development fit/evaluation coordinates are disjoint.
Confirmatory targets only score the fixed bank and define the evaluator
Oracle; neither retrieval selector sees them.

## Primary results

| Recipe | Eligible cases | Shared mean RMSE | Oracle mean RMSE | Oracle gain / wins | Bootstrap lower | Spatial NN mean RMSE | Oracle gap closure | Branch |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ClassNeg | 24/24 | `.015363` | `.012795` | `16.72% / 75.0%` | `.000941` | `.020720` | `-208.62%` | `case_bank_oracle_only` |
| Velvia | 24/24 | `.016928` | `.015273` | `9.78% / 68.75%` | `.000321` | `.022866` | `-358.58%` | `case_bank_no_oracle_value` |

ClassNeg passes all three frozen Oracle gates exactly at or above their
boundaries: at least 10% mean improvement, at least 75% wins and a positive
paired-bootstrap 95% lower bound. The bank therefore contains held-out
post-hoc value under this paired software-recipe evaluator.

The preregistered selector fails decisively. Spatial photometric nearest
neighbour is worse than the shared O0 (`.020720` versus `.015363`) and closes a
negative `208.62%` of the shared-to-Oracle gap. Global photometric nearest
neighbour is also worse than shared and closes `-167.93%`. Spatial retrieval
does beat random (`.021330`) and its deterministic shuffled assignment
(`.022961`), but those controls are themselves worse than the shared champion;
that comparison cannot promote retrieval.

Velvia misses both primary Oracle gates: `9.78%` is below 10% and `68.75%` is
below 75%. Its positive bootstrap interval does not override the frozen
magnitude and win-rate requirements. The case-routing branch closes for this
recipe control.

## Recipe-specific negative control

The post-hoc Oracle from the wrong recipe bank is much worse:

| Target recipe | Correct Oracle mean | Best wrong-bank mean | Correct improvement | Correct wins |
|---|---:|---:|---:|---:|
| ClassNeg | `.012795` | `.061494` | `79.19%` | `100%` |
| Velvia | `.015273` | `.058185` | `73.75%` | `100%` |

This shows that the fitted banks are not interchangeable generic colour
operators. It does not produce an inference-time selector and does not convert
Capture One recipe evidence into physical-film evidence.

## Structural evidence

All shared and eligible O0 flows pass every frozen structural gate:

| Recipe | Minimum determinant | Maximum Jacobian norm | Maximum inverse error | Maximum coefficient norm | Sampled output range | Replay |
|---|---:|---:|---:|---:|---:|---:|
| ClassNeg | `.19392` | `1.92329` | `4.34e-8` | `1.13441` | `[.04923,.96463]` | `0` |
| Velvia | `.34792` | `1.98156` | `8.06e-8` | `1.25960` | `[.03252,.96783]` | `0` |

The result is an information/selection finding, not numerical collapse,
out-of-cube rendering, orientation reversal or excessive flow capacity.

## Interpretation

Z0 separates two questions that previous reference-matching discussions often
conflated:

1. **Does a bank contain useful alternatives?** For ClassNeg, yes under a
   held-out paired Evaluator Oracle.
2. **Can raw input appearance retrieve the useful alternative?** The frozen
   global and spatial photometric nearest-neighbour selectors do not. They are
   substantially worse than one shared bounded operator.

The user's intuition that a photograph might benefit from a previous
case-specific colour operator therefore retains a narrow research path, but
not through simple "looks similar in scene colour and luminance" retrieval.
A distinct next hypothesis may test asymmetric query/operator applicability:
whether a simple development-only hard reranker can predict which fixed
operator is safe and useful for a query. It must compare against the shared
champion, random, shuffled and the failed photometric selectors before any
trained router or visual candidate is allowed.

## Branch consequences

- Retain the ClassNeg bank only as a paired evaluator Oracle and research gap.
- Reject the frozen global/spatial photometric hard selectors.
- Close case routing for Velvia under Z0.
- Do not retune descriptors or gates, add capacity, train a router, generate
  visuals or access the final 628 as a Z0 rescue.
- W2F1 output-only reference recovery remains closed.
- Current real-film pixels, stock learning, LSM, calibration and production
  integration remain closed.

`ClassNeg` and `Velvia` here are Capture One software-recipe domains. They are
not physical stocks, processes, scanners or identified digital-to-film
operators.
