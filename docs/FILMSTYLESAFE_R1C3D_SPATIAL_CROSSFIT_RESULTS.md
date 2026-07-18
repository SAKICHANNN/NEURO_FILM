# U5.R1C3D spatial cross-fit absorption diagnostic results

Date: 2026-07-18

Decision: **fail — close the current conditioned-SCIS route**

## Result

The fixed 4×4 neighbour-exclusion affine cross-fit reproduces the R1C3 parent
weak-pass, then remains at 2/3 proxy positives under the unchanged all-negative
zero-FPR rule.

| Readout | Result |
|---|---:|
| maximum non-severe score | 63.9936 |
| maximum negative | RF2.C0 external style control |
| positives detected | 2/3 |
| ID11 | 164.5868 |
| solid synthetic island | 129.2571 |
| sparse HF synthetic | 1.2444 |
| hardneg/external below every positive | no |

Spatially excluding each predicted tile and its one-tile neighbourhood did not
recover the sparse HF member. The frozen spatial self-contamination explanation
is therefore insufficient. The result does not identify the true mechanism and
does not validate or invalidate the A0 proxy label.

## Reproducibility

- contract config SHA-256: `5acf3c29...e67f40`;
- implementation: `38470c5`;
- two complete reports are byte-identical at SHA-256 `b845fabe...d22f78`;
- 15 focused tests pass;
- 647 complete CPU tests pass;
- compile and diff checks pass.

Ignored reports remain under `outputs/filmstylesafe/r1c3d/`.

## Branch decision

No further canonicalizer, capacity, grid, threshold or style-control exclusion
is allowed on this A0 route. SCIS v0/v0.1 and conditioned variants remain
descriptive research metrics only. The product retains the independent severe
artifact veto and deterministic regression tests; it must not treat this
failed detector research as permission to accept artifacts.

A future hidden A1 or new transform-family programme would require its own
data/authority/contract. Ultimate remains active and returns to an independent
explicit product leaf.
