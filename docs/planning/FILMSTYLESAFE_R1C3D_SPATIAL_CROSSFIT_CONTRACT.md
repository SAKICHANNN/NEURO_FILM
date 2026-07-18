# U5.R1C3D spatial cross-fit absorption diagnostic contract

Date: 2026-07-18

Node: `ULT > U5.R1 > U5.R1C3D`

Status: frozen before implementation

## Parent evidence and question

R1C3 affine weak-passes at 2/3 because the sparse HF synthetic falls to
`1.2429`; quadratic is worse and is closed. The next and only authorized
mechanism question is:

> Does a same-pair affine fit absorb a localized, source-state-correlated
> failure because the affected pixels participate in their own canonicalizer?

## Frozen diagnostic

Retain the exact R1C3 affine basis, ridge, Huber delta, IRLS count, score and
eight A0 members. Divide every image into a deterministic 4×4 spatial tile
grid. For each output tile:

1. exclude that tile and every tile within Chebyshev distance one from fit
   samples;
2. fit the affine source-Lab-to-candidate-ab relation on the remaining pixels;
3. predict only the held-out tile;
4. stitch the 16 predictions without overlap or interpolation;
5. compute the unchanged conditioned SCIS score once on the stitched residual.

Maximum fit samples remain 65,536 per held-out tile. Each fit must retain at
least 4,096 pixels. Tile boundaries are integer `linspace` boundaries covering
every pixel exactly once.

## Controls and decision

- reproduce R1C3 self-fit affine scores before cross-fit interpretation;
- all 53/55/56 style controls remain non-severe negatives;
- strict threshold remains `score > max(all five non-severe scores)`;
- full mechanism pass requires 3/3 positives and hard-negative/external below
  every positive;
- otherwise the diagnostic fails and conditioned-SCIS calibration closes;
- repeated reports must be byte-identical;
- focused/full tests, compile and diff checks must pass.

## Branches

- pass: record A0 evidence that local self-contamination caused the R1C3 miss;
  open only a separately frozen resolution/new-transform-family design;
- fail: close the conditioned-SCIS route and keep existing metrics descriptive;
- invalid: repair coverage, sample, baseline, determinism or evidence defects
  and rerun the unchanged contract.

## Forbidden fallback and claim ceiling

No quadratic/higher basis, different grid, tuned exclusion radius, altered
threshold, dropped control, new member, neural feature, training, hidden A1,
recruitment or production integration after seeing results.

Claim ceiling: A0 mechanism diagnostic for spatial self-absorption only; never
human confirmation, a validated detector, population risk evidence or a
product gate.
