# U1.6G2 Exact Streaming Percentile Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G2`

**Decision:** pass the independent exact reduction primitive; do not integrate
physical or density halation yet.

## Reproducibility

- implementation commit: `746c4f3842dd9495a89a743bc793ada924360e09`;
- algorithm: `float32-radix-percentile-v1`;
- config: `configs/u1_6g2_exact_streaming_percentile_v1.json`;
- config SHA-256: `66c19df1c46dca2fa058457690925a2f6e099c468aa56a96d9c29f9a55f24a29`;
- formal stream: 1,048,613 deterministic lognormal float32 values, seed 91,
  65,537 values per chunk;
- requested percentiles: 99.7 and 99.8;
- reference: `numpy.percentile(method="linear")`.

## Formal result

| Percentile | Selected ranks | Reducer | NumPy | Float64 bytes equal |
|---:|---:|---:|---:|---|
| 99.7 | 1,045,466 / 1,045,467 | 17.288356315612788 | 17.288356315612788 | yes |
| 99.8 | 1,046,514 / 1,046,515 | 20.609730636596556 | 20.609730636596556 | yes |

Both passes produced stream SHA-256
`7f5e600638f3828efcec100e8d783b947ab6af6acb30d56b898167f5b391e247`.
The persistent histograms consumed 1,572,864 bytes: one 65,536-bin high table
and two selected high-bucket low tables. This size depends on requested rank
buckets, not pixel count. The local two-pass smoke took about 25 ms; that is a
diagnostic, not a 100MP or end-to-end performance claim.

## Verification

- 11 focused tests passed;
- 49,793 random finite IEEE-754 float32 bit patterns matched NumPy byte-for-byte
  at 107 percentile points;
- duplicate-heavy, negative, signed-zero, min/max/tiny, irregular chunk and
  chunk-boundary cases passed;
- changed second-pass bytes, count mismatch, float64, NaN/Inf and invalid
  percentile inputs fail closed;
- 24 combined U1.6G1/G2 tests passed;
- 431 complete CPU tests passed in 20.78 seconds.

## Decision and next branch

U1.6G2 passes as the exact global-percentile prerequisite. It does not modify
the existing `np.percentile` effect path. U1.6 remains active and may now freeze
a scratch-field DAG/lifetime plan that composes U1.6G1 and U1.6G2 without hidden
full-resolution fields. Colour and density effects still require separate
versioned integration and visual/severe-artifact gates.

## Claim ceiling

This result proves exact NumPy-linear percentile values for repeatable finite
float32 streams with bounded radix-histogram memory. It does not prove effect
parity, physical realism, renderer integration, streaming decode, stock or
calibrated response, bounded total renderer memory or 100MP readiness.
