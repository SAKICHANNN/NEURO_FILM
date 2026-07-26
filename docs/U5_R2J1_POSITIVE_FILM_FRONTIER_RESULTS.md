# U5.R2J1 positive-film real-image frontier results

Date: 2026-07-26

Node: `ULT > U5 > U5.R2 > U5.R2J1`

Decision: **close; automatic strength did not become visual value**

## Automatic result

Both 1,025-render passes and both metric passes are byte-identical. Seven of
25 fixed candidates pass the inherited automatic gates. The three frozen
shortlist representatives are:

| Candidate | Style DE76 | Non-basic residual | Worst gold/stress new clip |
|---|---:|---:|---:|
| `cross_bias_like__s35` | 11.877 | 7.771 | 0% / 0% |
| `warm_highlight_like__s35` | 12.746 | 6.609 | 0% / 0% |
| `cyan_shadow_warm_highlight_like__s35` | 13.101 | 6.417 | 0% / 0% |

This confirms that the J0 response family can create strong, non-basic,
bounded real-image changes. It does not establish that those changes look
better.

## Blind visual result

The three deterministic rounds contained input, safe-rich, the retained R2E1
density challenger and all three J1 representatives. Choices were recorded
before opening the private mapping.

| Round | Winner | Runner-up |
|---:|---|---|
| 1 | retained R2E1 `cyan-shadow/warm-highlight s0.50` | safe-rich |
| 2 | retained R2E1 `cyan-shadow/warm-highlight s0.50` | safe-rich |
| 3 | retained R2E1 `cyan-shadow/warm-highlight s0.50` | safe-rich |

No J1 candidate enters the top two in any round. The J1 looks are either too
close to the milder control or introduce less attractive warm/green bias. The
extra automatic residual is therefore not evidence of better film style.

## Full-resolution safety

All 27 shortlisted J1 gold renders were inspected at full resolution. No
confirmed severe corruption, banding, posterization, geometry failure or
unintended clipping is present. In particular, all three ID11 renders are
smooth and contain no recurrence of the historical red-speckle regression.

The family closes for lack of visual gain, not for artifact failure.

## Reproducibility

- evaluator commit: `cec5cc7355bb764d9e64182dabda57e40ceeeb97`;
- render manifest SHA-256, both passes:
  `a3bd882ea1b21c0426759bde181dcaf376b7aaafb624ab3c25dfa8aff51c213e`;
- automatic report SHA-256, both passes:
  `07d616c1da23a58fecb720f23b2ddc32bdbabc1a869367a0b4f4cdb13a9c3023`;
- exact blind choices, sheet hashes and mapping hash are frozen in
  `configs/u5_r2j1_positive_film_frontier_decision_v1.json`.

## Branch

Retain J0 as a reusable clean-room representation and negative algorithm
control. Do not tune the same synthetic bank, increase capacity or compose it
with the winner merely to chase this development set. R2E1 remains the
strongest artifact-clean B0 challenger; safe-rich remains the current product
fallback. No stock, calibration, preference-population, training, router or
production permission opens. The Ultimate Goal continues.
