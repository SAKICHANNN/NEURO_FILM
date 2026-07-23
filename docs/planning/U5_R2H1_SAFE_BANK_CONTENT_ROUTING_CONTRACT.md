# U5.R2H1 safe-bank content-routing feasibility contract

Date frozen: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2H1`  
Status: **frozen before descriptor or router results**

## Question

Does the user's desired pattern - “this input resembles cases for which this
colour treatment works better” - have measurable value when routing only
between two already artifact-clean, project-owned explicit operators?

This is an A0 development feasibility study. It is not stock learning,
preference learning, confirmatory evidence, or production routing.

## Parent evidence and known-at-freeze facts

The immutable bank contains:

- `anchor56_chroma_margin4_challenger`;
- `cyan_shadow_warm_highlight_like__s50`.

Both have passed their frozen aggregate automatic gates and all-nine
full-resolution severe review. The post-hoc metric comparison already observed
before this contract is frozen:

- on 41 samples, residual and style winners agree for 30;
- residual winner counts are 15 anchor / 26 density;
- style winner counts are 18 anchor / 23 density;
- on the nine gold samples, a raw `style + non-basic` Oracle has about 9.1%
  median utility gain over the better single density operator.

These are disclosed development observations, not fresh confirmatory results.

## Immutable data

Use the exact 41-source U4.1 A0 manifest and the exact R2B/R2E1 reports and
render manifests. Verify every source and output hash. No generated output is
used to compute input descriptors.

All 41 rows remain A0 development. The historical gold/stress split is
reported but cannot be called confirmatory because both result banks have
already been inspected.

## Separate content representation

Compute only from the unstyled source image, after EXIF transpose and
deterministic 64x64 RGB resize:

- 16-bin luma histogram;
- nine luma quantiles;
- eight-bin saturation histogram;
- 4x4 luma cell means;
- 4x4 chroma-magnitude cell means;
- global RGB mean and standard deviation.

This representation is content space only. It never defines a film mode,
stock identity, source label, scanner label, or output transform.

Fit standardization inside each leave-one-out training fold. No PCA,
normalization or threshold may use the held-out row.

## Three disclosed Oracle views

For each row compare immutable operator metrics:

1. non-basic residual;
2. style Delta E76;
3. raw composite `style + non-basic`.

If the absolute utility difference is below `0.5`, label the row unassigned
for that view. Do not force a winner. Report coverage and class balance.

The composite view is the primary development routing target, but the route
may advance only if its conclusions are not contradicted by the two component
views. None is a human preference label.

## Frozen routers

Evaluate exact leave-one-out predictions:

1. training-fold majority class;
2. standardized Euclidean hard 1-NN;
3. standardized Euclidean hard 3-NN with distance-weighted vote;
4. L2 logistic regression, `C=1`, fixed seed `20260723`.

No neural network, embedding, image output, semantic detector, augmentation,
feature search, hyperparameter tuning, soft style blend, or per-image strength
is allowed.

For 1-NN and 3-NN record selected training IDs. Final application is always one
hard existing operator; the router never predicts RGB or parameters.

## Negative controls and gates

For each view report:

- assigned coverage;
- balanced accuracy and ordinary accuracy;
- per-class recall;
- median achieved utility;
- best-global utility;
- Oracle utility;
- fraction of global-to-Oracle mean regret closed.

Run 1,000 deterministic label permutations through the complete LOO
standardization/router path. The p-value is
`(1 + count(null balanced accuracy >= observed)) / 1001`.

The content-routing hypothesis passes only if:

- the composite Oracle median gain over best global is at least 5%;
- both classes have at least eight assigned rows;
- one non-majority router has balanced accuracy at least 65%;
- permutation p-value is at most 0.05;
- it closes at least 30% of mean global-to-Oracle regret;
- its direction is not contradicted by both component views.

Otherwise close learned content routing for this two-operator A0 bank. Do not
increase capacity or add semantic embeddings.

## Claim ceiling and branches

Maximum claim:

`A0 development evidence for or against hard content-based selection between
two already safe deterministic operators under autonomous metric-derived
labels`.

A pass permits only a separately frozen full-resolution replay/visual
diagnostic. A fail retains the two global operators and closes routing.

No stock, mode, authenticity, calibration, owner/population preference,
training-data generalization, production, or paper claim opens. Ultimate
remains active under every result.
