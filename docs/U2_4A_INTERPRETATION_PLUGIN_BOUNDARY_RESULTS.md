# U2.4A interpretation plugin boundary results

Date: 2026-07-18

Decision: **pass — retain isolated software boundary; real interpretations remain pending**

## Implemented boundary

`src/inference/interpretation.py` provides a pure request/plugin/result API with
explicit interpretation ID, operator ID, input/output colour domains, evidence
scope, production eligibility and claim ceiling. Execution validates registry
identity, exact domain match, float32 HxWx3 finite bounded input/output, shape,
input immutability, output non-aliasing and immutable JSON-safe metadata.

Synthetic-test-only plugins cannot be marked production eligible. Unknown,
unregistered, mismatched or ineligible requests fail before returning output.

## Synthetic witnesses and controls

All three witnesses live only in the test module:

- slide direct scan: exact identity copy;
- B&W developer scan: exact neutral-axis projection;
- colour-negative neutral scan: synthetic `1-rgb` inversion with maximum
  double-application roundtrip error below `1e-6`.

They are software witnesses, not image-quality candidates or physical models.
Tests cover duplicate registration, wrong interpretation/operator pairing,
wrong colour domain, production escalation, invalid dtype/shape/finite/range,
attempted input mutation, aliased/wrong/nonfinite/out-of-range output and
mutable/ndarray metadata.

## Reproducibility

- contract config SHA-256:
  `60cd8f71671f1292771f06fd500a0ffe8cd6fb448801ed4fba7fb8763b4a1be2`;
- implementation commit:
  `909394c7910f4e7c76f12529e11ddd17ad2c2f4e`;
- 23 boundary tests and 50 focused/adjacent tests pass;
- 698 complete CPU tests pass;
- compile and diff checks pass;
- renderer, v1 profile schema, v1 recipe schema and tracked profile do not
  reference the boundary and retain their prior hashes.

## Branch decision and claim ceiling

Retain U2.4A as an isolated extension boundary. U2.4 remains pending because no
real negative/slide/B&W interpretation operator or evidence-qualified profile
exists. No renderer integration, production profile, physical interpretation,
digital-to-film operator, stock authenticity or calibration claim opens.

