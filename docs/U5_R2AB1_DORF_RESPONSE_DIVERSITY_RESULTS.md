# U5.R2AB1 — DoRF response diversity results

## Decision

**`insufficient_diversity`; close the DoRF response-bank route.**

Two full reports from software commit
`a6895ca72de977d6776af3a04e3d4d60f3939338` are byte-identical at SHA-256
`ee64b659c00bea9dc5855f121fbdff5e4f388ac4a71a5112819a00cb1f00d985`.
The frozen config hash is
`7d030b200f7bd47b6f2586301123906fbde646e5000753785197a65797797c29`;
report payload hash is
`f3a5e5f65c46145d146ec379fa2725b6fb63594b2bc794f727fe89f24479f655`.

## What passed

All 46 strict RGB triplets are strong on the synthetic grid:

- style gate: 46/46;
- joint-basic residual: 46/46;
- shared-curve residual: 37/46;
- per-channel-power residual: 42/46;
- neutral chroma: 39/46;
- derivative floor/cap: 43/46 and 46/46;
- raw finite range: 46/46.

Across candidates, median style Delta E76 ranges from 13.08 to 43.11
(population median 25.27). Median residual from the shared response ranges
from 0.54 to 10.07, and residual from per-channel power ranges from 0.81 to
31.00. Twenty-seven candidates pass every individual frozen gate.

This is evidence that normalized one-dimensional channel responses can create
a pronounced, non-basic appearance. It is not evidence of cross-channel film
colour physics: exact independent monotone curves reproduce every candidate
by construction.

## Why the bank failed

The frozen bank required the closest two individual survivors to remain at
least `0.02` output RMSE apart. Observed minimum separation is only:

`0.0010158953030260727`

for:

- `agfacolor-futura-400CD`
- `agfacolor-hdc-400-plusCD`

The survivor-count gate passes (27 versus required three), but the pairwise
diversity gate fails. This is the intended protection against treating nearly
the same operator direction or response strength as multiple modes, analogous
to the project's 53/55/56 negative-control principle.

Seven positive-film-labelled candidates also fail neutral chroma, and three
fail the derivative floor. These failures are not repaired with a neutral
gauge, smoothing or clamp because the contract forbids post-result rescue.

## Consequence

- Do not construct a DoRF stock bank.
- Do not train a selector/router.
- Do not render project photographs to choose a favourite.
- Do not alias or deduplicate names after seeing results.
- Do not call these curves calibrated stock responses or digital-to-film
  transforms.

At most one curve may remain as a descriptive historical response prior in a
future separately justified synthetic control. AB2 does not open. Ultimate
continues through a distinct evidence-authorized algorithm or data leaf.
