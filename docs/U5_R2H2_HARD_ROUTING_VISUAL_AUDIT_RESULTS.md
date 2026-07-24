# U5.R2H2 exact hard-routing visual audit results

Date: 2026-07-24  
Node: `ULT > U5 > U5.R2 > U5.R2H2`  
Decision: **close the exact hard-1NN visual policy**

## Integrity and replay

- software commit: `3de4ef38bccb9df854cd120a5dd4da95070900d4`
- config SHA-256: `fa7084b3f245da55f6fe9e0946c8a04cb44c752d0fee0cd9c0d67b9c5f3121d5`
- replay-manifest SHA-256: `c35c335db244e59c405529762552f5485593a715672f96e815a9cad4f6e405c9`
- presentation-index SHA-256: `7e140e9b6144c46d319db2cded173ae3eda4fe2c0b2f2b441006839d97db50a3`
- blind scores were frozen before key access at SHA-256
  `8be8ee65b3fa21586194995ade9a4deaccac98b11574c9febfca3649bb418e0a`
- blind-key SHA-256: `0aadd3343ae7ace0a0b3158d7e72f42accf3144c9c86ce5619afb7335608a00f`
- severe-review SHA-256: `d10ad048d7c06ddd53defc921f729693182c943f2be6adae5e4657017fb309fb`
- adjudication SHA-256: `7b8bad1c16cb58a018fe8159740545c8afa6db811721d5b854ed481ca7ca5947`

The replay verifies 41 source hashes, both complete candidate banks, all H1
assignment membership, ten anchor predictions, 27 density predictions and
four density fallbacks. No RGB was regenerated.

## Severe-artifact result

All 41 routed full-frame outputs and central crops were inspected. Suspicious
rows were opened at original resolution.

- confirmed routed severe failures: **0/41**;
- new severe failures caused by an intervention: **0/10**;
- blind A/B severe marks: **0**.

ID 24 contains faint chroma noise in a dark sky and is visually inferior under
the anchor route, but it is not a confirmed severe colour block, band,
geometry failure, or ID11-scale red-speckle failure. It remains a negative
quality example rather than a severe veto.

## Blind result

All ten actual interventions entered every round; there was no post-hoc
shortlist.

| Round | Routed wins | Global wins | Ties | Score | Frozen 7/10 gate |
|---|---:|---:|---:|---:|---:|
| full fit | 6 | 4 | 0 | +2 | fail |
| central 60% detail | 6 | 4 | 0 | +2 | fail |
| reordered/re-sided full fit | 6 | 4 | 0 | +2 | fail |

The decoded direction is perfectly stable across the three repeated
presentations:

- routed anchor visually wins on `11, 18, 20, 23, 28, 39`;
- density global visually wins on `06, 12, 24, 40`.

The positive-score gate passes in all three rounds, but the stricter
win-or-tie coverage gate fails at 6/10 rather than the frozen 7/10. The exact
policy therefore closes. The threshold, descriptor, metric labels, neighbour
count, bank and fallback are not changed.

## Interpretation

H1's content-space signal was not imaginary: the routed operator is visually
better for a stable majority of its interventions and remains severe-clean.
However, metric-derived style-plus-residual labels and this 71-D source
descriptor do not support a sufficiently reliable visual policy. Four errors
are stable across full-frame and detail views, so treating the near-pass as a
promotion would be post-hoc gate relaxation.

The two global operators remain valid B0/A0 research looks. H2 does not
invalidate content-conditioned explicit operators in general; it closes this
exact H1 policy and forbids immediate capacity expansion on the same exposed
41 rows.

Maximum claim remains autonomous A0 finite-bank evidence. There is no human or
population preference, held-out generalization, stock/mode, calibration,
authenticity, production, or paper claim. Ultimate remains active.
