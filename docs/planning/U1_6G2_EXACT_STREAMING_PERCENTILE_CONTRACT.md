# U1.6G2 Exact Streaming Percentile Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G2`

**Status:** frozen / implementation ready

## Purpose

Physical and density halation currently normalize source fields with global
99.7/99.8 percentiles. U1.6G2 freezes a repeatable two-pass reduction that can
recover exact float32 order statistics without sorting or retaining the whole
field in RAM. It remains independent of effect integration and U1.6G1.

## Algorithm

The candidate is `float32-radix-percentile-v1`.

1. Accept a repeatable chunk-factory, declared element count and one or more
   finite percentiles in `[0, 100]`.
2. Convert each finite float32 value to an order-preserving uint32 key: negative
   bit patterns are complemented; non-negative patterns flip the sign bit.
3. Pass one counts the high 16 bits in 65,536 uint64 bins and records a SHA-256
   over the exact row-major float32 byte stream.
4. Resolve the floor/ceiling ranks required by NumPy's `method="linear"` rule.
5. Pass two counts the low 16 bits only for the high buckets containing those
   ranks, and repeats the element-count and byte-stream hash audit.
6. Recover exact selected float32 values from their radix keys and interpolate
   in float64 using `(n - 1) * q / 100`.

The method is exact for finite IEEE-754 float32 values under the frozen linear
quantile definition. It is not an approximate value-range histogram.

## Scope and invariants

Allowed changes are a new reduction module under `src/filmfx/`, public exports,
focused tests and deterministic synthetic evidence. Existing effect code,
normalization defaults, renderer and CLI remain unchanged.

The implementation must:

- use at most one 65,536-bin high histogram plus one low histogram per distinct
  selected high bucket;
- accept 1-D or arbitrary-shaped finite float32 chunks and ignore chunk
  boundaries while preserving byte-order audit;
- require the factory to produce the identical stream twice;
- reject empty, non-float32, non-finite, count-mismatched or changed streams;
- preserve duplicate values, signed zero and negative ordering correctly;
- return immutable result metadata including algorithm version, count,
  percentiles, selected ranks, pass hash and histogram bytes.

No source field, sorted run or full-resolution percentile cache may be retained.

## DoR

- U1.6G0 identifies percentiles as an independent global prerequisite;
- U1.6G1 has passed and is not modified by this leaf;
- clean pre-contract HEAD is `4184db1`;
- no effect integration, renderer, download or training process is active.

## DoD and gates

1. bit-exact equality to `np.percentile(..., method="linear")` on frozen random,
   duplicate-heavy, negative, signed-zero, extreme-float and irregular-chunk
   float32 cases for 0/1/50/99.7/99.8/100 percentiles;
2. chunk-boundary changes do not change results when the concatenated stream is
   identical;
3. changed second-pass bytes, count mismatch, dtype, NaN/Inf, invalid q and
   non-repeatable factories fail closed;
4. histogram memory is reported and independent of pixel count;
5. repeated execution is byte-identical;
6. focused tests and the complete CPU suite pass.

## Branches

- **Pass:** retain the reducer and open scratch-field DAG/lifetime planning;
  effect-family integration remains separately gated.
- **NumPy parity failure:** repair the rank/key/interpolation definition; do not
  loosen exact equality or silently adopt an approximate percentile.
- **Repeatability failure:** reject the source factory; do not accept a stale
  percentile context.
- **Resource failure:** close this implementation and retain current full-frame
  effect behavior.

## Claim ceiling

A pass proves only exact bounded-histogram percentile reduction for repeatable
finite float32 streams. It does not prove full physical-halation staging,
legacy effect parity, physical accuracy, renderer integration, stock response,
streaming decode, bounded total memory or 100MP performance.
