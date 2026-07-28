# P157 Chunked Lab-Ingress Memory Contract

## Purpose

P156 moved the 6 MP whole-file peak from full-frame gamut rendering to
reference fitting and persistent style arrays. P157 tests one bounded
implementation claim: converting display-linear RGB to Lab in exact 128-row
slices and reusing that validated Lab must reduce the whole-file peak without
changing any durable result or safe-Lab semantics.

The functional parent is P156 implementation
`240e6e6a81129c5bca298915d7caf9f013b74d15`. The measurement baseline includes
its durable evidence only:
`bfb36d6065145bf21ec418c16d64483b1226e6eb`.

## Frozen implementation boundary

The candidate may add one private row-kernel module and change only reference
fit/render callers and focused tests. It must:

1. preallocate one float32 Lab result and fill it with the existing
   `linear_rgb_to_lab` function over at most 128 rows;
2. validate gamut over the same bounded rows;
3. reuse the validated reference Lab for recipe statistics;
4. reuse the validated source Lab for style application;
5. preserve the existing full-image safe-Lab context and style operator.

Recipe schema, algorithm identity, policy defaults, style/gamut math, producer
contracts and the Neuro-Film main renderer are forbidden changes. In
particular, P157 may not row-slice the current Gaussian luma-detail operation.

## Frozen evidence

The same deterministic 3000-by-2000 PNG inputs and baseline/candidate/candidate/
baseline order used by P156 apply. A pass requires:

- exact output, recipe and normalized semantic-report identities;
- exact default identity fallback and zero cleanup residue;
- focused bit-exact equivalence for both working spaces and non-divisible
  heights;
- at least `67,108,864 B` median process-tree RSS reduction;
- candidate/baseline median RSS ratio no higher than `0.94`;
- candidate/baseline median worker-wall ratio no higher than `1.05`.

Thresholds are frozen before implementation and measurement. Failure remains a
valid result; thresholds will not be lowered.

## Claim ceiling

A pass proves only a local Windows/Python bounded Lab-ingress kernel for the
reference-match consumer. It does not prove total high-resolution readiness,
main/native/device parity, RAW/HDR/video support, producer compatibility,
A1/A4/A5, visual quality or product admission.
