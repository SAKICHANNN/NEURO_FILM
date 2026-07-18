# U1.6G3 Halation Field-DAG and Lifetime Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G3`

**Decision:** pass the static planner and keep effect integration closed.

## Reproducibility

- implementation commit: `5fd5c6b98a916195aee77b41135229832f557496`;
- planner: `halation-field-dag-v1`;
- config: `configs/u1_6g3_halation_field_dag_v1.json`;
- config SHA-256: `0652882a1e012dc495909e443fa3082fd2c32edaef386c63e5d583a18f9b8333`;
- cases: `4000x6000` (24MP) and `10000x10000` (100MP arithmetic
  only), tile 512, default diffusion 1.0, derived linear source;
- no image pixels, effect execution, download, fitting or training occurred.

## Graph identity and inventory

The implementation parses current `effects.py` in tests rather than trusting
only hard-coded counts.

| Family | Nodes | Blur calls | Percentiles | Finite-halo | Global-grid |
|---|---:|---:|---:|---|---|
| physical-colour-v1 | 41 | 8 | 2 | local_abs, red_near, red_mid, green_near, green_mid | local_mean, red_tail, red_glare |
| density-v1 | 31 | 6 | 1 | local_abs, near, mid | local_mean, tail, glare |

Both graphs preserve source normalization, background mean/deviation,
visibility/skin controls and distinct colour/density terminal semantics. Lazy
stream dependencies propagate liveness transitively: coarse contexts and
percentile scalars are not released after only their first direct consumer.

## Formal resource arithmetic

All byte values below exclude caller-owned external arrays and required returned
outputs from the planner-owned RAM peak.

| Case/family | External | Required output | Peak context | Peak scalar | Peak finite workspace | Peak planner RAM | Fingerprint |
|---|---:|---:|---:|---:|---:|---:|---|
| 24MP colour | 288,000,000 | 384,000,000 | 18,164,000 | 16 | 3,902,528 | 18,164,016 | `d3d26ca1...` |
| 24MP density | 288,000,000 | 384,000,000 | 12,791,340 | 8 | 4,091,472 | 12,791,348 | `3062ca34...` |
| 100MP colour | 1,200,000,000 | 1,600,000,000 | 75,685,556 | 16 | 3,902,528 | 75,685,572 | `685307ca...` |
| 100MP density | 1,200,000,000 | 1,600,000,000 | 53,305,124 | 8 | 4,091,472 | 53,305,132 | `f5859243...` |

The 100MP rows are integer/resource-plan evidence only. In particular, the
1.2GB input and 1.6GB returned layer demonstrate why these rows cannot support
an end-to-end memory or performance claim.

## Readiness decision

Every plan correctly returns `integration_ready=false` and names:

- `coordinate_exact_gradient_window`;
- `row_chunked_global_stage_builder`.

The global-grid nodes are reported as unresolved workspace nodes until the
second capability exists. Maximum downstream output input halo is one pixel,
coming from the gradient-dependent visibility/source paths.

## Verification

- 27 focused DAG tests passed;
- 51 combined U1.6G1/G2/G3 tests passed;
- 458 complete CPU tests passed in 20.70 seconds;
- malformed vocabulary/storage, duplicate/unknown nodes, cycles,
  use-before-produce, invalid geometry/diffusion/capabilities and overflow-risk
  shapes fail closed;
- existing effect, renderer, profile, recipe and CLI paths remain unchanged.

## Branch and claim ceiling

U1.6G3 passes as a faithful static graph/lifetime/resource planner. It opens
only the two missing execution prerequisites, beginning with coordinate-exact
gradient windows. It does not execute either effect, prove pixel parity,
physical realism, streaming decode, stock/calibrated response, bounded total
memory or 100MP readiness.
