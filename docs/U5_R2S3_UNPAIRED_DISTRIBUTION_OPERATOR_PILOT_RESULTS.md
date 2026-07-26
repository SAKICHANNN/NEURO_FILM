# U5.R2S3 Unpaired Distribution/Operator Pilot Results

Date: 2026-07-27

Decision: **distribution fail — close both fixed unconditional objectives
without optimizer or capacity rescue**

## Reproducibility

- software commit:
  `c460c2af4818f6c2810909d17839e274a8d58b22`;
- config SHA-256:
  `862a1c20e2ced3dc18ad7e5039d6656312d4f8d144d528290d0f4f9175b08c93`;
- two complete CUDA reports are byte-identical at
  `b9d9fd8753b8b7e83c4fb584029e59c287f0a4484d4c661fbe63f2bc71215a55`;
- reserved confirmation seed `28203` was not accessed;
- the complete CPU suite passes `932/932`.

Each method fits eight independently generated styles twice, from disjoint
neutral and styled observations. No source/target pair or hidden operator is
visible to optimization.

## Automatic results

| Method | Held-out distribution improvement | Oracle RMSE median / p90 | A/B operator RMSE | Style retention | Initial -> final fit loss |
|---|---:|---:|---:|---:|---:|
| RFF-MMD-192 | 11.80% | .11591 / .15423 | .07902 | .6450 | .0004269 -> .0001848 |
| SW-24 | 3.36% | .11444 / .16447 | .13138 | .8730 | .019866 -> .002610 |
| Frozen gate | at least 30% | at most .07 / .10 | at most .04 | .6--1.4 | diagnostic only |

Both optimizers substantially reduce their in-fit objectives. Neither method
passes the held-out distribution gate, hidden-operator gates or independent
A/B-fit stability gate.

## This is not optimizer collapse

Every structural gate passes:

| Method | Min determinant | Max Jacobian norm | Max inverse error | Max coefficient norm |
|---|---:|---:|---:|---:|
| RFF-MMD-192 | .20430 | 2.3777 | `1.70e-8` | 1.3712 |
| SW-24 | .05627 | 3.4177 | `1.66e-7` | 1.5728 |

Both operators also preserve the cube range, replay exactly and pass the
frozen style-retention interval. The experiment therefore produced valid,
visibly nonidentity-capable explicit operators; the unconditional
distribution objectives simply selected transforms that neither generalize
well enough to the independent distributions nor recover a stable hidden
operator.

Training loss is not evidence of the desired mapping. Many bounded maps can
reduce a marginal-distribution discrepancy, and the frozen objectives do not
contain enough correspondence information to select the generated
photographic operator.

## Branch decision

The formal branch is `distribution_fail`, rather than
`distribution_pass_operator_fail`, because neither method reaches 30%
held-out distribution improvement. S3 is closed without:

- additional optimization steps or a larger grid;
- new projections, bandwidths or loss selection;
- access to confirmation seed `28203`;
- a visual shortlist;
- project photographs or film pixels.

No visual review is warranted: this is a synthetic identifiability experiment
and the automatic scientific gate failed before any image claim.

## Next distinct question

S4 is now ready under its already-frozen contract. It asks whether one shared
bounded flow becomes identifiable when several independently generated
source/target condition distributions have **known correct correspondence**:

```text
same hidden style
  + several generated content conditions
  -> pooled versus correctly corresponding versus shuffled conditions
  -> one bounded explicit flow
```

That condition identity is generated truth. It is not inferred from RGB,
content, CLIP, uploader or scanner metadata. Even a positive S4 result would
only establish a synthetic identifiability mechanism; real use still requires
an independently observed, rights-cleared auxiliary condition that passes the
stock/source/content gates.
