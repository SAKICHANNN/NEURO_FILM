# U5.R2L1 full-frame strength-preflight results

Status: **complete — renderer replay passes; exact policy gate fails and route
closes**

Node: `ULT > U5 > U5.R2 > U5.R2L1`

## Result

The deterministic algorithm trial-renders the fixed density-cyan operator at
`s0.65`, measures the full RGB8 frame with the frozen endpoint metric, and
hard-falls back to `s0.50` above `0.005`.

The implementation itself replays perfectly:

| Gate | Result |
|---|---:|
| archived `s0.50` RGB8 matches | 41 / 41 |
| archived `s0.65` RGB8 matches | 41 / 41 |
| R2L0 Oracle assignment matches | 40 / 41 |
| R2L0 selected-output RGB8 matches | 40 / 41 |

Both formal reports are byte-identical at SHA-256
`e1aa4271a077bfc8110ee2a691708336edff8bae45c3247fe97290c09f451a4b`.

## Single frozen-boundary disagreement

Stress sample 20 is the only mismatch:

- R2L0 sampled clipping: `0.0049247049`, selecting `s0.65`;
- R2L1 full-frame clipping: `0.0050015475`, selecting `s0.50`.

The full-frame value exceeds the frozen ceiling by about `0.0000015475` as a
fraction, but the contract requires all 41 assignments exactly. Nearness does
not permit relaxing the gate.

## Decision

Close the exact full-frame preflight. Do not change the threshold, epsilon,
strengths or sampling after seeing this result, and do not train a predictor to
imitate a policy that failed its simplest deterministic gate.

R2L0 remains valid only as evidence that heterogeneous strength headroom exists
under its exact sampled evaluator. It is not a deployable policy.

The reusable `src/roll2film/adaptive_density_strength.py` primitive remains
research infrastructure because it is deterministic, explicit, hard-routed
and fully tested. It is not integrated into the production renderer.

## Verification

- seven focused tests pass;
- both fixed candidate banks replay 82/82 decoded RGB8 arrays exactly;
- two failure reports are byte-identical;
- full local CPU suite: **868 passed** in 62.81 seconds.

## Claim ceiling

B0 deterministic replay and negative boundary-sensitivity evidence for one
fixed film-inspired density Look Approximation. No accepted inference policy,
universal safety, learned routing, preference, digital-to-film
identification, named-stock response, calibration, authenticity or production
claim.
