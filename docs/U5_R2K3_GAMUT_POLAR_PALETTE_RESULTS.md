# U5.R2K3 Gamut-Polar Palette Results

## Decision

**Closed on the Jacobian-norm gate.** The clean-room gamut-polar
factorization passes every range, inverse, neutral-axis, endpoint, orientation,
diversity, non-affinity and replay gate. One frozen
cyan-shadow/warm-highlight witness reaches maximum local Jacobian spectral norm
`11.198`, above the frozen `8.0` budget.

The fixed bank does not open a real-image frontier. Its parameters and gates
must not be weakened after this result.

## Reproducibility

| Item | Value |
|---|---|
| Research principle | ACES 2.0 invertible polar gamut compression design |
| ACES formula/code/constants copied | none |
| Software commit | `43efb5bbe32c1de175e5d076fe882ba9da45971b` |
| Config SHA-256 | `02634f1f95aa2e397492b0cad1bfcf38335eaa732b09f1ed23df9625dc47f647` |
| Report SHA-256 | `5def5f9101fa81a40b82d2986c813ff417e615a3b83dfe43446bb645bae15d96` |
| Repeated reports | byte-identical |
| Full cube / Jacobian points | `15,625 / 729` |
| Focused tests | 4 passed |

## Structural results

All five fixed witnesses:

- map the full sampled RGB cube into exact `[0,1]` with no output clip;
- preserve black, white and the complete neutral axis;
- roundtrip through the analytic inverse below `6.6e-15`;
- have strictly positive sampled Jacobian determinants;
- replay exactly after serialization and across partitioned execution.

| Witness | RGB RMSE from identity | Best-affine residual | Min det(J) | Max norm | Inverse max error |
|---|---:|---:|---:|---:|---:|
| identity | ~0 | ~0 | 1.0000 | 1.000 | `3.48e-15` |
| cyan shadow / warm highlight | .06458 | .05591 | .04342 | **11.198** | `2.92e-15` |
| warm dense | .10393 | .07410 | .05443 | 7.697 | `4.00e-15` |
| cool soft | .07550 | .05209 | .04600 | 6.127 | `6.59e-15` |
| cross palette | .07403 | .06132 | .00915 | 7.110 | `4.00e-15` |

Minimum pairwise witness RGB RMSE is `.04501`, comfortably above the `.015`
diversity floor. All non-identity witnesses exceed the `.03` identity-distance
and `.008` non-affine-residual floors.

## Interpretation

The result validates the central representation idea:

- an exact RGB-cube ray boundary can replace post-hoc gamut clipping;
- tone, exposure-layered hue and hue-specific chroma can produce large,
  distinct and non-affine palette changes;
- neutral-axis preservation and invertibility can remain structural.

It also finds the important failure mode before rendering photographs. A
smooth, orientation-preserving map can still be locally too aggressive.
Exposure-dependent Möbius hue trajectories can magnify small source-colour
differences even when outputs stay in gamut. This is a plausible precursor to
neon-edge instability or posterization-like stress behavior, so positive
determinant and bounded output are insufficient safety evidence.

## Binding branches

Forbidden:

- reduce or retune the failed witness after seeing the Jacobian grid;
- relax the norm budget or omit the cyan witness;
- clip, project or smooth the output as a rescue;
- render this fixed bank on the gold/stress images;
- describe the operator as ACES-conformant or film-measured.

Allowed:

- retain the coordinate implementation as a research primitive/negative
  control;
- design a separately motivated palette family with an analytic derivative
  budget frozen before evaluation;
- continue stock-data acquisition and the retained R2E1 product challenger.

## Claim ceiling

Clean-room data-independent numerical and diversity evidence. The gamut-polar
factorization is bounded, invertible, neutral-safe and strongly non-affine, but
one frozen palette exceeds the local Jacobian-norm budget. It establishes no
ACES conformance, measured film response, stock identity, calibration,
authenticity, preference or production promotion.
