# U5.R2AK1 Time-Dependent Cube-Flow Capacity Results

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AK1`  
Decision: **closed — structurally safe, but less parameter-efficient than both stationary controls**

## Integrity

- frozen config SHA-256:
  `cab133017f7e84586dbab209403d9743a9a41242a699045f8b2dbac66c3fa560`;
- formal software commit:
  `ca155ec438d2cf93da0d2c1b9cf6bc5e4ceb6e47`;
- child report SHA-256:
  `d410076b01691dc58e4adabfa92059291cddcd3d73e439062dfa50a8165649f0`;
- repeat-decision SHA-256:
  `fa35d413a4a034040fd1fe12d88502ea6f0da4229b081a964712981d0a3dedda`;
- two fresh processes independently reconstructed every fit and produced
  byte-identical 6,773-byte canonical reports;
- the target pair passes the pre-fit noncommutation gate at RGB RMSE
  `0.0176158`.

The implementation uses deterministic float64 CPU optimization. The fitted
parameters define an explicit boundary-vanishing velocity field; a neural
network never emits pixels. A mathematically equivalent vectorized trilinear
sampler was verified against the reference value and gradient paths before
formal execution.

## Result

The 243-parameter quadratic-time `K=3` candidate passes absolute fidelity and
every structural gate on both target orders:

| Target | Candidate RMSE | Stationary K4 | Stationary K5 | Gain vs K4 | Ratio to K5 |
|---|---:|---:|---:|---:|---:|
| A then B | 0.0023466 | 0.0013491 | 0.0009169 | -73.94% | 2.559x |
| B then A | 0.0025862 | 0.0013777 | 0.0009043 | -87.72% | 2.860x |

The candidate's frozen absolute RMSE ceiling is `0.008`, so both directions
are accurate in isolation. They fail the two capacity-efficiency gates:

- neither improves at least 25% over the 192-parameter stationary K4 flow;
- neither stays within 1.10x of the 375-parameter stationary K5 flow.

This is not an optimizer, integration, or safety failure. Candidate minimum
finite-difference Jacobian determinants are `0.58349` and `0.53692`; maximum
spectral norms are `3.43099` and `3.23316`; inverse errors are below
`3.42e-9`. Range, endpoints, coefficient bound, serialization and partition
checks all pass.

## Interpretation

Time-dependent vector fields are mathematically valid explicit renderers, but
this frozen temporal basis does not buy useful capacity at its lower spatial
resolution. The stationary controls approximate each complete staged map more
accurately despite having fewer or comparable parameters. The result therefore
does not support carrying this exact `K=3`, three-control quadratic-time family
into film-inspired fitting.

The frozen branch forbids adding grids, stages, controls, optimizer steps or a
different target after seeing this result. The reusable implementation and
negative result remain as mechanism evidence. No photograph was rendered and
no external code, checkpoint, image or film pixel was used.

Ultimate remains active and moves to a distinct color-selective explicit
operator question: a clean-room fixed analytic named-colour partition with
bounded monotone curves. That future leaf must not reuse the original
NamedCurves canonicalizer, attention, code, weights or color-naming asset and
must remain synthetic until separately gated.

## Claim ceiling

`repeat-exact clean-room synthetic negative capacity evidence for one fixed
quadratic-time cube-preserving explicit colour flow`.

There is no stock, calibration, unpaired-identification, preference,
photographic-safety or production claim.
