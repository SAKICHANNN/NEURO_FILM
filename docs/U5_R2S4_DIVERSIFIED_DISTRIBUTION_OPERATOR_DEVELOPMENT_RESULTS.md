# U5.R2S4 Diversified-Distribution Operator Development Results

Date: 2026-07-27

Decision: **conditional distributions match, hidden operator fails**

## Reproducibility

- software commit:
  `c03c321b9fc642e2e092d59e20dd1b145b96192d`;
- config SHA-256:
  `2347a1dd10c6680f687ea2fd4db9593cfd43e54c87fea2df63a7fb5eeaa5a5b8`;
- two complete CUDA float32 reports are byte-identical at
  `25ad02460cf248416d84ab9cb8260bdd35554f27d710920c343c98a0f0955b57`;
- the reserved confirmation seed was not accessed;
- no image or visual shortlist was generated.

Every method sees identical independently sampled source and styled
observations. Only the generated condition identity differs: pooled discards
it, conditional uses the correct correspondence, and shuffled applies the
frozen non-identity permutation `[1, 3, 0, 2]`.

## Automatic results

| Method | Held-out conditional improvement | Oracle RMSE median / p90 | A/B operator RMSE | Style retention | Initial -> final fit loss |
|---|---:|---:|---:|---:|---:|
| Pooled RFF-MMD-192 | 64.94% | .11744 / .12388 | .02568 | .5880 | .0008194 -> .0001700 |
| Correct-condition RFF-MMD-192 | 71.36% | .10751 / .11900 | .03153 | .6458 | .0020569 -> .0003294 |
| Shuffled-condition RFF-MMD-192 | 24.79% | .14331 / .14769 | .04288 | .6982 | .0041950 -> .0027180 |
| Frozen gate | at least 30% | at most .07 / .10 | at most .04 | .6--1.4 | diagnostic only |

Correct condition correspondence is real signal in this generated design. It
passes held-out conditional-distribution improvement, independent-fit
stability and style-retention gates; the shuffled control fails its
operator/stability negative control as required.

It still does not identify the hidden operator. Median and p90 oracle errors
miss their frozen `.07/.10` gates. Correct conditions improve median oracle
error by only 8.45% over pooled, below the required 25%. They improve it by
24.97995% over shuffled, also below the frozen 25% gate. The latter is not
rounded upward or promoted post hoc.

## Structural validity

The primary flow is numerically healthy:

- output range: `.05321` to `.95901`;
- minimum Jacobian determinant: `.15076`;
- maximum Jacobian spectral norm: `2.51170`;
- maximum inverse error: `6.83e-8`;
- maximum coefficient-vector norm: `1.57310`;
- replay and permutation-operator errors: exactly zero.

Every range, Jacobian, norm, inverse, replay, coefficient and style gate
passes. This is not numerical collapse. It is an information/objective result:
the conditional objective learns a coherent bounded map that improves
distribution correspondence without recovering the generated photographic
operator accurately enough.

## Branch decision

The formal branch is
`primary_matches_distribution_but_operator_fails`.

S4 closes without:

- additional optimization, loss tuning or model capacity;
- relaxed gates or a post-hoc 24.98% pass;
- confirmation-seed access;
- a visual shortlist;
- project photographs, film pixels or real-condition claims.

The result supports the weaker statement that known, corresponding auxiliary
distributions can materially reduce marginal ambiguity. It does not establish
that a real dataset supplies valid conditions, that conditions are independent
of content/source shortcuts, or that unpaired digital-to-film transport is
identified.

## Next distinct question

U5.R2U1D is ready under its already-frozen contract. It tests whether
hierarchical colour coupling can construct more informative synthetic
pseudo-pairs while fitting only the already bounded O0 explicit flow:

```text
same generated condition truth
  -> random / pooled-HCC / correct-condition-HCC / shuffled-condition-HCC
  -> bounded pair-fitted flow
  -> hidden-operator and A/B stability gates
```

These are constructed correspondences, never observed pair truth. A U1 pass
would remain synthetic mechanism evidence and would not open current pixels,
stock learning, calibration or production integration.
