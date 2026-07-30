# U6.P8BQ Fixed-Arm Evaluator-Oracle Diagnostic

## Decision

`censored_no_router`

This leaf does not reopen U6.P8BP. It reuses only hash-bound, already frozen
autonomous winner observations and confirms that the native Standard arm is
the compiled P7F `gauged_spatial_4000` challenger.

## Result

| Population | Fixed AO6 | Physical | Per-source observed Oracle | Gain | Physical-selected sources | Censored |
|---|---:|---:|---:|---:|---:|---:|
| P7F development | 13/27 | 14/27 | 20/27 | +7 | 7/9 | 0/27 |
| P8BP fresh | 7/27 | 5/27 | 9/27 | +2 | 2/9 | 15/27 |

The development population contains strong descriptive heterogeneity. On the
fresh population, the observed per-source Oracle improves the AO6 winner count
by two votes and selects the physical arm for Blackmagic and Phase One.
However, B0 wins 15 of 27 rounds. Those rounds reveal neither the AO6/physical
runner-up order nor a pairwise preference, so imputing them would fabricate
evidence.

The frozen numerical gain and support checks pass, but the complete-pairwise
gate fails. No selector or router opens; AO6 remains the simpler global colour
champion. A future routing study would first require a separately
preregistered, uncensored pairwise design on a new eligible population, not
extra rounds on P8BP.

## Reproducibility

- contract: `configs/u6_p8bq_fixed_arm_oracle_diagnostic_v1.json`
- implementation: `src/eval/physical_fixed_arm_oracle.py`
- runner: `scripts/run_u6_p8bq_fixed_arm_oracle.py`
- tests: `tests/test_u6_p8bq_fixed_arm_oracle.py`
- two report SHA-256 values:
  `fd1c13bc1670e97dbad6f246a825fe96c48165d7a497e5437c1d137a8f483b36`
- stable evidence ID:
  `1ccf41424a748665a19eaa8645988af786a4e047c41c7b35c18671fd0166af30`

Claim ceiling: retrospective autonomous winner-only evaluator-Oracle
diagnostic. It is not complete pairwise Oracle evidence, population
preference, stock response, calibration, or product promotion.
