# SF2.3 Commons-union shared-author connectivity results

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.3`

Decision: **insufficient shared-author connectivity; close this union before live pages or pixels**

## Frozen evidence

- software commit: `44cf22fd9a5a59edd848973af61903067637dddb`;
- config SHA-256: `92fd5c47...e74d2`;
- report SHA-256: `0e29ef44...59914`;
- decision SHA-256: `ffcd15fc...961d`;
- two complete executions produced byte-identical report and decision hashes;
- all three snapshot hashes and all three source-config hashes match the
  frozen contract;
- input contract errors and all four cross-stock identity-overlap classes are
  zero.

The formal audit retains 294 strict scene rows across 19 exact-stock categories
and 30 conservatively normalized author strings. It does not merge apparent
aliases or infer author identity from uploader.

## Support and graph result

Only two stocks pass the reused per-stock gate of at least eight strict rows,
five authors and no author above 60%:

| Stock | Strict rows | Authors | Largest author share | Gate |
|---|---:|---:|---:|---|
| Kodak Ektar 100 | 26 | 8 | 30.77% | pass |
| Kodak UltraMax 400 | 51 | 8 | 50.98% | pass |

Several other categories have many rows but fail independent-source support:
Industrial 100 has 24 rows/two authors/87.5% dominance; ProImage 100 has
30/two/86.67%; Vision3 50D has 29/two/96.55%; Vision3 500T has
35/two/97.14%. Row volume therefore does not repair author confounding.

The two eligible stocks share only one normalized author, `toomore chiang`.
The frozen graph requires at least two independent authors per edge, three
eligible stocks, three edges and a cycle. Consequently:

- raw eligible-stock edges: one;
- retained two-author edges: zero;
- graph components eligible for preflight: zero.

The development census's eight multi-stock strings are not confirmatory
evidence: most connect stocks that independently fail the source-support gate,
and no eligible edge has leave-one-author-out redundancy.

## Interpretation and binding branch

This result does not prove that Wikimedia Commons as a whole lacks useful
stock connectivity. It establishes that the current three immutable snapshots,
under their frozen exact-stock/right/scene filters and conservative author
identity, cannot support the next controlled experiment.

Therefore:

- do not merge `svetlov artem` with `artem svetlov` or any other unverified
  aliases after seeing the result;
- do not lower the two-author, component or dominance gates;
- do not open a live-page preflight or download additional pixels from this
  union;
- do not fit operators, train models or begin LSM from these rows;
- retain the audit as source-connectivity negative evidence and continue
  Ultimate through another independently frozen data/source leaf or the
  deterministic product path.

The claim ceiling is current-snapshot metadata connectivity only. There is no
stock response, appearance, identifiability, operator, latent-mode, `S1/S2`,
calibration, authenticity or product claim.
