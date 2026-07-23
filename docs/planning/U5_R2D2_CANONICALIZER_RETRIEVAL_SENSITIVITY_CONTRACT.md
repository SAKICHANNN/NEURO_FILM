# U5.R2D2 canonicalizer and hard-retrieval sensitivity contract

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D2`  
Status: **frozen before implementation or results**

## Parent evidence

U5.R2D1 rejects ordinary unpaired source/reference interaction on exact
generated operators: confirmation RGB RMSE is 0.15512, versus 0.06897 for the
global mean. The exact canonical-reference delta reaches 0.05638 and improves
54.29% over target-only, but captures only 29.8% of true style.

This leaf asks whether any explicit approximation to that unavailable exact
delta retains the gain, and whether the user's proposed hard “this resembles
that case” retrieval is more robust than parameter regression.

## Fixed parent data

Reuse the exact U5.R2D1:

- 384 operator manifest and 768 observations;
- operator-group fit/validation/confirmation/stress splits;
- matrix-only, tone-only and combined truth;
- palette-support shifts and 17-cube evaluation;
- numerical validity and severe-veto contract.

No parent gate, seed, split or result is changed.

## Independent neutral-control bank

Generate only neutral RGB clouds, never transformed targets:

- eight independent clouds for each of the nine frozen palette families;
- seed namespace disjoint from operator generation and evaluation clouds;
- no operator parameter, transformed pixel or split label in the bank;
- exact bank manifest and hash;
- the same bank is frozen for every method.

This is a synthetic analogue of a rights-cleared neutral digital pool. It is
not a real-film canonicalizer and cannot create paired evidence.

## Frozen canonicalization hypotheses

Each hypothesis estimates a neutral reference for the transformed style
reference and forms:

`LabStatistics(transformed reference) - LabStatistics(estimated neutral)`

1. `query_as_neutral`: use the different-palette query image as the neutral
   control. This is practical but deliberately weak.
2. `nearest_raw_lab`: hard Top-1 neutral-bank retrieval by raw 2,304-D Lab
   descriptor distance.
3. `nearest_basic_normalized_quantiles`: hard Top-1 retrieval by 33 quantiles
   per RGB channel after per-channel median/IQR normalization. This removes
   first-order exposure/WB/contrast but not palette shape.
4. `fixed_gaussian_neutralization`: per-channel mean/std-map the transformed
   reference to the frozen balanced neutral template, then use that image as
   the estimated neutral.
5. `palette_oracle_nearest`: restrict quantile retrieval to the known
   synthetic palette family. This is a content-label oracle, not deployable
   film evidence.
6. `exact_raw_reference_oracle`: reuse the exact pre-transform reference. This
   is the parent upper control.

All distance normalization is fit on development-fit observations or the
neutral bank only. Confirmation cannot select features, thresholds, pools or
canonicalizers.

## Frozen prediction policies

For every signature:

- `ridge`: standardized PCA(48) plus the parent alpha grid;
- `hard_case_top1`: standardized PCA(48), nearest development observation,
  exact retrieved operator parameters, no averaging;
- `sparse_case_top3`: inverse-distance Top-3 parameters followed by bounded
  projection.

Hyperparameters and PCA are development-fit/validation only. Hard Top-1 is the
primary test of case retrieval; Top-3 is a smoothing sensitivity, not a
dense-mixture default.

## Additional controls

- global mean and parent exact oracle;
- shuffled operator-group targets;
- neutral-bank retrieval same-palette accuracy;
- maximum neutral-bank item share;
- pairwise disagreement between practical canonicalizers in projected
  parameter space and rendered 17-cube RGB;
- leave-one-palette-family-out neutral-bank sensitivity.

The same embedding cannot be described as film-mode space. This is synthetic
operator signature retrieval only.

## Frozen gates

1. Parent manifest/config/report hashes and zero group leakage reproduce.
2. Every projected policy is valid; any severe numerical veto rejects it.
3. Exact-oracle best RMSE must reproduce within 5% relative of U5.R2D1
   `0.056384`.
4. A **practical canonicalizer** counts as useful only if its best policy:
   - improves confirmation median RGB RMSE at least 10% over global mean;
   - captures at least 30% of true style;
   - does not fail any family by exceeding global RMSE by more than 10%.
5. Robust canonicalization requires at least three of the four practical
   hypotheses (`query`, `raw-Lab`, `normalized-quantile`, `Gaussian`) to be
   useful independently.
6. Their median pairwise rendered-operator disagreement must be <= 0.03 RGB
   RMSE on the 17-cube. Otherwise the operator conclusion is
   `canonicalizer_sensitive/unidentified`.
7. Hard-case value requires Top-1 to beat ridge by at least 5% for at least
   one independently useful practical canonicalizer, without worsening any
   family by more than 10%.
8. Sparse Top-3 may advance only if it beats Top-1 by at least 3% and remains
   within the same validity/stability gates.
9. Palette-oracle success without practical success is a
   content-connectivity result, not an operator result.
10. Stress and leave-one-palette results are interpreted only after
    confirmation gates freeze the branch.

## Branches

- no practical canonicalizer useful: close canonicalizer-conditioned learning
  and do not add model capacity;
- only palette oracle useful: neutral content connectivity is the missing data
  requirement;
- practical methods disagree: mark unidentified and close product routing;
- stable practical canonicalizer plus Top-1 win: open a separately frozen
  bounded hard FilmCase retrieval challenger;
- stable regression but no retrieval win: retain bounded prediction research,
  not the “similar case” hypothesis;
- any real-film use still requires stock evidence, rights, identifiability and
  leakage gates independently.

## DoD

- contract/config committed before code;
- neutral bank, canonicalizers and retrieval have independent unit references;
- model preprocessing remains development-only;
- two reports and manifests are byte-identical;
- full CPU tests and worst-case Hald visual audit pass;
- evidence propagates without changing production;
- scoped commits are pushed and the Ultimate Goal remains ACTIVE.
