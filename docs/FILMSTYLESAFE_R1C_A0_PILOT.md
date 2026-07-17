# U5.R1C A0 conventional-metric failure pilot and SCIS v0

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1C`

Decision: **conventional-metric gap confirmed on A0 proxy labels; SCIS v0 remains an unpromoted candidate**

## Delivered

- Contract: `configs/filmstylesafe_r1c_contract_v1.json`
- Implementation: `src/eval/filmstylesafe_r1c.py` (PSNR/SSIM/mean-abs/ΔE76 + explicit SCIS v0)
- Runner: `scripts/run_filmstylesafe_r1c_a0_pilot.py`
- Ignored report: `outputs/filmstylesafe/r1c/a0_metric_pilot_report.json`
- Decision: `configs/filmstylesafe_r1c_decision_v1.json`

## A0 proxy-label pilot (not human population)

| Cohort / score | Sensitivity @ zero FPR |
|---|---:|
| Best conventional vs all non-severe | 0.333 (1/3) |
| Best conventional vs hardneg+external | 0.333 (1/3) |
| SCIS v0 vs all non-severe | 0.333 (1/3) |
| SCIS v0 vs hardneg+external | 0.667 (2/3) |
| Hard-neg SCIS below all positives | yes |

Observed member pattern:

- Global fidelity metrics treat strong owner-anchor style (53/55/56) as more
  “different” than sparse HF chroma speckles (`mean_delta_e76` 0.75 for HF
  synth vs 12.4 for scheme 53).
- SCIS v0 fires on the solid magenta island and stays above the bloom
  hard-negative for every proxy-severe member, but the sparse HF operator and
  strong global style still break a perfect zero-FPR gate.

## Adjudication

**Facts:** conventional global fidelity metrics fail to catch chromatic severe
proxy cases at zero FPR on this bound A0 suite. SCIS v0 is directionally useful
for large smooth-region islands and hard-negative ranking, but not yet a
complete detector for sparse speckles under strong style controls.

**Inferences:** the FilmStyleSafe metric-gap hypothesis is supported for A0
development. SCIS remains a candidate, not a safety gate.

**Hypotheses for R1C2:** multi-scale / lower residual thresholds, style-robust
global-relation residuals, and explicit sparse-speckle features.

Gates are not lowered. No A1 population, recruitment, training or risk claim
opens.

## Claim ceiling

A0 proxy-label development evidence only.

## Next

Preferred: `U5.R1C2` refine SCIS v0 for sparse speckles, or parallel product
`U1.2/U1.4/U1.5`. Goal remains ACTIVE.
