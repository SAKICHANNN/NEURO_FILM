# U5.R2D1 synthetic known-operator recovery results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D1`  
Decision: **canonical information passes; unpaired interaction closes**

## Result

The generated benchmark contains 384 exact operators and 768 observations.
There is zero operator-ID overlap across fit, validation, confirmation and
stress. All three truth families are materially non-identity:

| Family | Median truth-versus-identity uniform RGB RMSE |
|---|---:|
| matrix-only | 0.07369 |
| tone-only | 0.07791 |
| combined | 0.10183 |

Every projected prediction remains finite, in range, neutral-axis preserving,
monotone and positive-determinant. Numerical validity therefore does not
explain the accuracy differences.

## Confirmation comparison

| Representation | Median RGB RMSE | Captured style | Median Delta E76 |
|---|---:|---:|---:|
| global mean | 0.06897 | 0.177 | 10.516 |
| source-content only negative | 0.07094 | 0.201 | 10.809 |
| target-style only | 0.12336 | -0.645 | 18.523 |
| source/target interaction | 0.15512 | -0.948 | 22.049 |
| canonical-reference delta oracle | 0.05638 | 0.298 | 9.398 |
| matched-query delta oracle | 0.05716 | 0.295 | 8.816 |
| shuffled-operator negative | 0.10250 | -0.311 | 15.831 |

The canonical-reference delta oracle improves over target-only by 54.29% and
over the global mean by 18.25%. It passes the frozen canonical-information
gate, although it captures only 29.8% of the true style effect and is not a
complete recovery solution.

The deployable unpaired interaction is 125% worse than the global mean and
25.7% worse than target-only. It fails both interaction gates and the 50%
captured-style gate. Source-only and shuffled controls do not spuriously pass.

## Interpretation

The result supports a narrow but important conclusion:

> Exact knowledge of how a reference changed relative to its neutral source
> contains recoverable operator information; a transformed unpaired reference
> alone does not separate that operator from its scene palette in this
> benchmark.

This is evidence that a canonicalization or matched-control layer can matter.
It is not evidence that the project currently has a valid real-photo
canonicalizer. The exact pre-transform reference used by the oracle is
unavailable for real film scans.

Adding a larger network to the failed interaction lane is not authorized. It
would add capacity without adding the missing information and would be at high
risk of learning the palette shortcuts already exposed by U5.R2D.

## Visual and reproducibility evidence

The non-binding worst-case Hald sheet shows large global tone/hue direction
errors for interaction predictions, without geometry changes or numerical
glitches. This agrees with the quantitative failure; it does not provide a
film-aesthetic judgement.

- two final reports are byte-identical at
  `6fd2b4bc84b18d4a38b7daf6118d67fb6730520129485bf1eea62846c7eebdd5`;
- config SHA-256:
  `fc1939a394d30e6c84a017095ca47ce84abf6ee6a7a6b62de68cc83fbc4c83ec`;
- manifest SHA-256:
  `76c71291d7b6309da20c691d301a71609137879655663adeb43fc01e13fde9be`;
- implementation commit:
  `118d72b18122a3cd1717b1bcee0f28f46632eb92`;
- visual sheet SHA-256:
  `4d75a5ad6497e7d0dea84d7d1da5e07a6ecf0c18d82b7453fe7cb1939c08a3c5`;
- 21 focused and 737 complete CPU tests pass.

## Branch

Open U5.R2D2 only as a separately frozen canonicalizer-sensitivity and hard
retrieval study. It must:

- replace unavailable exact deltas with multiple explicit, imperfect
  canonicalization hypotheses;
- compare hard operator/case retrieval against regression;
- vary the neutral reference pool and palette matcher;
- declare the method unidentified if conclusions change across plausible
  canonicalizers;
- retain the global mean/identity fallback and numerical severe veto.

No current stock pixels, operator fitting, neural RGB generation, LSM or
production integration opens. The Ultimate Goal remains ACTIVE.
