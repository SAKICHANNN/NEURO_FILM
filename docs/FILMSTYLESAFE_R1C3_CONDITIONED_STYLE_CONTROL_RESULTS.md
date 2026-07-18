# U5.R1C3 conditioned style-control results

Date: 2026-07-18

Decision: **weak-pass; no candidate selected or promoted**

## Result

The parent R1C2 baseline reproduced exactly. Replacing repeated connected-label
full-image scans with one `bincount` aggregation leaves its report byte-identical
at SHA-256 `a48b1d37...a4780` while making the frozen pilot executable inside the
local time budget.

| Candidate | Zero-FPR threshold, all 5 negatives | Positives detected | Decision |
|---|---:|---:|---|
| robust affine | 57.1735 | 2/3 | weak-pass only |
| robust quadratic | 53.2482 | 1/3 | fail |

Neither candidate reaches the frozen 3/3 promotion gate. The simplest full-pass
rule therefore selects nothing.

## Member mechanism

- scheme 53 remains the maximum negative for both candidates;
- affine detects ID11 (`145.8475`) and the solid synthetic island (`128.8120`);
- affine misses the sparse HF synthetic (`1.2429`);
- quadratic detects only the solid island (`128.2937`), while ID11 falls to
  `30.8372` and sparse HF to `1.3590`;
- because RF2.C0 external style scores `39.5606` under affine, the missed sparse
  HF case also violates the hard-negative/external-below-all-positives control.

Thus the weak-pass is not evidence that legitimate global style was generally
removed. The observed pattern instead supports a narrower hypothesis: a
same-pair colour fit can absorb source-state-correlated sparse failures, and
the quadratic capacity makes that problem worse. This is an A0 inference, not
a validated causal finding.

## Reproducibility

- contract config SHA-256: `b08a0463...76f814`;
- implementation: `f0fb467c19291ce806ec8a5e01d6308491aa6cf3`;
- two full reports are byte-identical at SHA-256 `b37541b0...ddec39`;
- 13 focused tests pass before the formal runs;
- 645 complete CPU tests pass;
- compile and diff checks pass.

Ignored evidence remains under `outputs/filmstylesafe/r1c3/`; no member,
threshold or report was changed after result inspection.

## Branch decision

The contract permits one separately frozen minimal mechanism diagnostic after
a 2/3 weak-pass. The next legal question is whether spatial cross-fitting —
fitting the affine relation outside each held-out block — prevents a sparse
artifact from teaching its own canonicalizer while preserving legitimate
global style. Quadratic/higher capacity is closed.

No conditioned SCIS candidate, safety gate, hidden A1 population, recruitment,
training, product veto or risk claim opens.
