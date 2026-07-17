# U5.R1C2 SCIS v0.1 sparse/style-robust refinement

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1C2`

Decision: **SCIS v0.1 reaches perfect hardneg/external sensitivity; still not a gate**

## Change

`scis_v0_1` adds:

1. low-frequency ab residual removal (style-robust HF residual);
2. multi-threshold island search;
3. sparse-density term for fine speckles.

## A0 proxy result (full-resolution pilot)

| Score / cohort | Sensitivity @ zero FPR |
|---|---:|
| Conventional best vs hardneg+external | 0.333 |
| SCIS v0 vs hardneg+external | 0.667 |
| **SCIS v0.1 vs hardneg+external** | **1.000** |
| SCIS v0.1 vs all non-severe (incl. 53/55/56) | 0.333 |

Hard-negative remains below all proxy-severe members under both SCIS versions.

## Adjudication

SCIS v0.1 closes the intended hard-negative / external-control gap on this A0
suite and beats conventional metrics there. Strong owner-anchor style still
breaks a universal zero-FPR threshold, so SCIS is **not** promoted to a safety
gate. No A1 population, recruitment or training opens.

## Next

`U5.R1C3` style-control exclusion / calibration protocol, or parallel U1
product leaves. Goal remains ACTIVE.
