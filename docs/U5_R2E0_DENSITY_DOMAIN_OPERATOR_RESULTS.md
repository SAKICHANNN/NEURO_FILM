# U5.R2E0 density-domain explicit operator results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2E0`  
Decision: **numerical/diversity primitive pass; visual frontier still required**

## Result

The clean-room linear-RGB negative-to-print chain passes every frozen
mechanical gate for all five descriptive witnesses. It is deterministic,
bounded, non-affine, locally orientation-preserving on the audit grid and
exact under partitioned execution and serialized replay.

| Witness | Identity RGB RMSE | Best affine residual RMSE | Minimum Jacobian determinant | Minimum directional derivative |
|---|---:|---:|---:|---:|
| neutral-density reference | 0.36470 | 0.16847 | 2.116e-5 | 0.00354 |
| warm-dense-like | 0.39456 | 0.16468 | 5.715e-6 | 0.00149 |
| cool-soft-like | 0.34349 | 0.15457 | 6.381e-5 | 0.00322 |
| cyan-shadow/warm-highlight-like | 0.37411 | 0.16812 | 5.924e-6 | 0.00189 |
| cross-processed-like | 0.41601 | 0.15562 | 1.236e-6 | 0.00183 |

All output samples remain in `[0,1]`. Endpoint, strength-zero,
strength-one, partition and replay maximum errors are exactly zero.

## Diversity

The ten pairwise uniform-grid RGB RMSE values range from `0.07319` to
`0.14704`, comfortably above the frozen `0.015` minimum. The witnesses
therefore span genuinely distinct explicit colour/tone directions rather than
five strength values on one path. This is a representation result only:
visible diversity does not establish film plausibility or safety.

## Reproducibility

- two formal audit reports are byte-identical at
  `82d63ad5fc1c548cc4240d2e8f51fb3c3990d54819bdbe7b0a46850d0f21dcc3`;
- config SHA-256:
  `fffb390eb7bb01789b770964da6cea081393ce09e0019dcd7d4a43098d3bf337`;
- implementation commit:
  `500df31c70e0e8bb2aefeaf6c6dd7d760690e96b`;
- 748 complete CPU tests pass.

## Interpretation and boundary

This result establishes one original, replayable explicit operator family
whose structure is richer than a global affine transform. It does not show
that any witness resembles a real stock, that the parameterization is
physically accurate, that strong settings avoid severe artifacts, or that
users prefer it. The names are visual descriptions, not exposure, process,
scanner or emulsion labels.

No external implementation, profile, LUT, output or stock response was copied
or used as teacher truth. Real-film operator fitting, training, LSM and
production integration remain closed.

## Branch

Open U5.R2E1: freeze and run the five witnesses over bounded strengths on the
existing U5.R2B gold/stress inputs. Apply the unchanged severe-artifact-first
ordering, compare against safe-rich and the retained margin-4 anchor56
challenger, preserve the ID11 red-speckle regression, and admit at most three
automatic survivors to blinded visual review. A complete E1 failure rejects
the current witnesses without invalidating the reusable primitive.

The Ultimate Goal remains ACTIVE.
