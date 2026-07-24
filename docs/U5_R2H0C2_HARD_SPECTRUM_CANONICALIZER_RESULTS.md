# U5.R2H0C2 hard measured-spectrum canonicalizer results

Date: 2026-07-24
Decision: **retain hard Top-1 at Delta E76 <=1.0; external replication required**
Parent: `U5.R2H0C1`

## Outcome

The simplest cross-scene hard measured-spectrum retrieval materially improves
prediction of the known CAVE overlap-witness target over smooth bounded
spectral reconstruction when it is restricted to a close D65-Lab neighbour.
The frozen selection rule retains threshold `1.0`, with smooth fallback
outside that radius.

This result directly supports the algorithmic idea "this colour is close to a
measured case, so use that case's explicit spectrum" rather than averaging all
cases. It still operates on CAVE cell colours and a synthetic datasheet-prior
target. It is not a learned Velvia transform or real-photo product evidence.

Formal report:
`outputs/u5_r2h0c2_hard_spectrum_canonicalizer/formal/report.json`, SHA-256
`f074c0c3551b4798f25697b6a912eb195356aceb606b86f00a77d12573cdca45`.

## Integrity and population

- 5,951 queries across 31 retained scenes;
- every bank excludes the complete query scene;
- zero same-scene nearest neighbours;
- feature is D65 CIE Lab only;
- target, smooth and retrieved spectra share one immutable overlap witness;
- two full evaluations are hash-identical;
- query CSV records scene/cell and retrieved scene/cell lineage for all rows.

The known target witness is nontrivial numerically: its Delta E76 from identity
is 13.76 median / 25.93 p95. The comparison is not passing because the target
look is nearly identity.

## Baselines

| Policy | Error Delta E76 median | p95 | mean | maximum |
|---|---:|---:|---:|---:|
| identity | 13.7589 | 25.9324 | 14.3868 | 54.7418 |
| smooth bounded reconstruction | 2.5813 | 14.9031 | 4.4159 | 31.7227 |
| raw hard Top-1, no fallback | 2.4771 | 10.6132 | 3.6366 | 50.5642 |

Raw hard retrieval improves the aggregate tail but has an unacceptable 50.56
worst case. It is diagnostic only, exactly as preregistered.

## Frozen threshold frontier

| T | coverage | win rate | win LCB | median reduction | reduction LCB | hard selected median / p95 | full-policy p95 | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 22.69% | 86.00% | 78.27% | 70.64% | 64.24% | 0.237 / 1.602 | 14.903 | eligible |
| 1.0 | 36.13% | 76.60% | 67.14% | 62.05% | 46.52% | 0.339 / 2.975 | 14.883 | **selected** |
| 2.0 | 56.75% | 68.52% | 59.30% | 34.58% | -0.40% | 0.780 / 6.710 | 13.996 | reject: reduction bootstrap |
| 3.0 | 72.66% | 65.98% | 57.81% | 13.97% | -11.45% | 1.390 / 7.937 | 12.763 | reject: point + bootstrap reduction |

Both smaller thresholds pass every gate. The frozen rule chooses the highest
coverage eligible threshold, so `T=1.0` is retained without tuning. Wider
radii improve aggregate p95 through greater coverage, but their across-scene
median benefit is not stable enough. This is positive evidence for a hard OOD
boundary, not for dense blending.

At `T=1.0`, selected smooth error is 0.894 / 3.939 median/p95, while hard error
is 0.339 / 2.975. The full fallback policy is 2.438 / 14.883; its overall p95
improvement is small because 63.87% of queries correctly retain the smooth
fallback.

## Numerical caveat

Across target, smooth and hard outputs, raw linear sRGB spans
`[-0.1198, 0.9198]`; 5.24% of channel values are outside gamut. This evaluator
does not clip them. A future renderer requires a separate bounded output
mapping and severe-artifact frontier before any visual promotion.

## Meaning for Ultimate

This result is the strongest mechanism evidence so far for case-based,
non-averaging spectral canonicalization:

- a real measured neighbour can preserve the nonlinear witness direction;
- hard selection outperforms one universal smooth spectrum on close cases;
- widening retrieval too far loses group-stable benefit;
- a deterministic smooth fallback is necessary for OOD colours.

It does not yet answer which complete photograph is similar, because H0C2
uses colour-space proximity only. It also does not validate the witness against
film scans. Content retrieval and mode routing remain separate future questions
under the existing architecture.

## Branch decision

Retain one internal research candidate:

`hard measured-spectrum Top-1, cross-source/group, Delta E76 <=1.0, otherwise smooth fallback`.

Open `U5.R2H0C3` only for external measured-spectrum source/replication audit.
The candidate must be frozen before external evaluation. A failure on the
external source closes broad empirical-prior claims; it cannot be rescued by a
neural network or threshold retuning.

No project photograph, film pixel, visual shortlist, product integration,
stock/calibration claim, preference claim or LSM work opens.

## Reproducibility

- config SHA-256: `9c300be85cb85918bbf66c38ec1031cacb24fcc52ccfb7c96b7e69a6a404c86f`;
- parent config SHA-256:
  `b878aa9e2de93b5f633df084078a12d0ea204d78c6c8ecdc2b67cb03cccfb9d1`;
- software commit: `e51f34f8200ba92fc36ec34c62b984404f5804f6`;
- transfer snapshot: `c7a6862f271b3a958c69a5c020f57064a04cc074`;
- exact runs: 2/2;
- focused H0A/H0C1/H0C2 tests: 11 passed;
- full CPU suite: 786 passed in 67.87 seconds.

## Claim ceiling

One-source, leave-one-scene-out prediction of a synthetic datasheet-prior
witness target using explicit hard measured-spectrum retrieval with a frozen
OOD fallback.
