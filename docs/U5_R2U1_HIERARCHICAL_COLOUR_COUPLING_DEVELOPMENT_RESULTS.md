# U5.R2U1 Hierarchical Colour Coupling Development Results

Date: 2026-07-27

Decision: **constructed pair fit passes; hidden operator fails**

## Reproducibility

- software commit:
  `ed1dbb5784b2519be8d56ff30961bee778995ced`;
- config SHA-256:
  `53a986860734c3d89b1c2ab84a26adf3263a78e7ad968e61d689a0ed025ebaee`;
- two complete CUDA float32 reports are byte-identical at
  `b0e9a64198414768a23ec183f1d87d016e1304f27314e05b69fc90ba6eda71b3`;
- stdout and stderr are empty;
- the reserved confirmation seed was not accessed;
- no image or visual shortlist was generated.

All methods fit the same bounded O0 flow. HCC creates pseudo-pairs by
recursively matching RGB-sign octants; those pairs are a transport prior, not
observed correspondence truth.

## Automatic results

| Method | Pair-fit improvement | Held-out distribution improvement | Oracle RMSE median / p90 | A/B operator RMSE | Style retention |
|---|---:|---:|---:|---:|---:|
| Random within correct condition | 53.46% | 66.85% | .07276 / .08348 | .03995 | .7507 |
| Pooled HCC | 79.20% | 66.80% | .08681 / .09639 | .02923 | .6568 |
| Correct-condition HCC | 84.72% | 65.75% | .08791 / .09699 | .04176 | .7058 |
| Shuffled-condition HCC | 60.59% | -10.97% | .12893 / .13160 | .04149 | .8525 |
| Frozen primary gate | diagnostic | diagnostic | at most .07 / .10 | at most .04 | .6--1.4 |

Correct-condition HCC strongly fits its constructed pairs and is 31.81%
better than shuffled HCC on median hidden-operator error. It nevertheless is
20.82% worse than the simple random correct-condition baseline and 1.27%
worse than pooled HCC. Its median oracle error and independent A/B replicate
error both fail.

The discrepancy is the result: a convincing pseudo-pair objective can be
optimized without identifying the generating operator. Pair-fit improvement
is therefore not a valid promotion proxy.

## Structural validity

The primary fitted flow remains numerically healthy:

- output range: `.04498` to `.96293`;
- minimum Jacobian determinant: `.09783`;
- maximum Jacobian spectral norm: `2.52969`;
- maximum inverse error: `9.62e-8`;
- maximum coefficient-vector norm: `1.43938`;
- replay and permutation-operator errors: exactly zero.

Every range, Jacobian, norm, inverse, replay, coefficient and style gate
passes. This is an identifiability failure, not a numerical collapse.

## Branch decision

The formal branch is `operator_fails_despite_pair_fit`.

U1 closes without:

- deeper coupling, alternative pairing, extra optimization or capacity;
- relaxed gates or post-hoc threshold changes;
- confirmation-seed access;
- a visual shortlist;
- project photographs, film pixels or real-pair claims.

The result rejects HCC pseudo-pairs as an operator-identification mechanism
under this frozen design. It does not claim that every hierarchical coupling
or every source of real paired evidence is impossible.

## Next distinct question

U5.R2W1D is ready under its already-frozen contract. Unlike marginal
distribution matching or constructed pseudo-pairs, it supplies a known
look identity across several unrelated generated contents during development
and tests whether an output-only reference representation can recover that
known bounded operator:

```text
same known bounded look across unrelated contents
  -> single versus four output-only references
  -> fixed retrieval and bounded parameter regression
  -> seen/unseen operator, nuisance, strength-path and paired-upper-bound gates
```

W1 remains synthetic mechanism evidence. It does not open film pixels, stock
learning, calibration or production integration.
