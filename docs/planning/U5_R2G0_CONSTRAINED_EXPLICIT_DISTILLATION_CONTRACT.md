# U5.R2G0 constrained explicit distillation contract

Date frozen: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2G0`  
Status: **frozen before any G0 fit or render**

## Question

Can the immutable, strongly styled but unsafe U5.R2F1 two-LUT transforms be
approximated on a fixed synthetic RGB cube by a small, bounded,
orientation-preserving explicit operator without the style collapse observed
under U5.R2F2 uniform contraction?

This is a method-control question. It is not a film-stock, calibration,
unpaired-operator-identification, or production experiment.

## Parent evidence

- F1: reference conditioning is material and 6/9 raw reference policies pass
  both style floors, but 0/9 passes all safety gates.
- F2: node clipping preserves style but not structure/safety; uniform safe
  contraction passes structure/range for 9/9 but passes style/non-basic for
  0/9.
- Current real-film pools remain unidentified and forbidden for fitting,
  training, operator learning, and latent-mode discovery.

## Immutable evidence and split

Use every one of the 81 F1 `(reference, gold input)` LUT pairs. For each pair,
compose canonical and restyle LUTs on a uniform `17 x 17 x 17` RGB cube. The
optimizer may see only these synthetic coordinates and their immutable
composed outputs. It may not read source photographs, reference photographs,
rendered F1/F2 images, labels, scene features, or evaluation metrics.

The nine gold photographs are evaluation-only after the corresponding
explicit operator has been frozen. There is no confirmatory claim; this entire
leaf is development evidence.

## Explicit family

The frozen family is:

1. three independent strictly monotone piecewise-linear channel curves on nine
   uniform knots, each with exact endpoints `(0, 1)` and a positive minimum
   interval;
2. one non-negative row-stochastic `3 x 3` matrix applied after the curves;
3. matrix parameterization
   `M = (1 - alpha) I + alpha softmax_rows(L)`, with
   `0 <= alpha <= 0.45`.

This construction maps `[0,1]^3` into `[0,1]^3`. The matrix path cannot cross
singularity for `alpha < 0.5`; the complete fitted operator must nevertheless
pass an explicit determinant and baked-grid Jacobian audit.

No bias, residual LUT, spatial term, mask, semantic feature, image-conditioned
post-adjustment, clipping inside the operator, or effect layer is allowed.

## Frozen optimizer

- PyTorch float64 CPU;
- deterministic algorithms and fixed seed `20260723`;
- three fixed restarts;
- Adam, 600 full-grid steps per restart;
- learning rate `0.03`;
- objective: mean squared error to the clipped immutable composed target plus
  `1e-4` identity regularization on the matrix and curve knots;
- choose the lowest objective restart with deterministic index tie-break;
- no early stopping and no threshold or hyperparameter changes after any G0
  output is inspected.

Clipping the synthetic target defines a bounded approximation target; it does
not sanitize the source LUT and the original raw excursions remain reported.

## Structural and replay gates

Every fitted operator must have:

- finite parameters and outputs;
- curve interval at least `1e-4`;
- matrix entries non-negative within `1e-12`;
- row-sum error at most `1e-12`;
- matrix determinant at least `0.05`;
- baked 17-cube nodes in `[0,1]`;
- minimum corresponding-channel grid step at least `1e-7`;
- minimum tetrahedral Jacobian determinant at least `1e-8`;
- serialized-operator replay maximum absolute error at most `1e-12`;
- two complete passes with byte-identical manifests and outputs.

## Unchanged image frontier

Aggregate the nine fitted per-input operators by reference ID, exactly as F1
and F2 aggregate their policies:

- gold median style Delta E76 at least `7.0`;
- gold median non-basic residual Delta E76 at least `4.9`;
- worst gold new hard clipping at most `0.5%`;
- raw final out-of-range fraction exactly zero;
- all nine fitted operators structurally safe;
- reference-bank matched-input median pairwise Delta E76 at least `2.0`.

For a branch-level advantage over uniform contraction, at least one reference
must pass every gate and the median style across the nine G0 reference policies
must exceed the matched F2 `safe_contract_cap100` median by at least `0.25`
Delta E76.

Synthetic target RMSE is descriptive and cannot override image safety or style.

## Visual protocol

Only automatic survivors may be reviewed. Select at most one survivor per
provenance bucket and three total, ranked by non-basic residual then style.
Run three deterministic blind rounds against the frozen R2B anchor56 and R2E1
cyan-shadow/warm-highlight challengers. Then inspect all nine original-size
outputs for every shortlisted reference, including the ID11
red-speckle/posterization regression. Any confirmed severe artifact rejects the
candidate.

## Branches

- no automatic survivor: close this explicit family; do not increase capacity
  or tune after results;
- automatic survivor but no branch-level advantage: retain evidence only and
  close product promotion;
- survivor and advantage but visual failure: reject;
- survivor, advantage, and visual pass: retain as a B0 generic
  reference-conditioned explicit-ML challenger only.

## Forbidden fallbacks and claim ceiling

Forbidden:

- CanonCGT training/fine-tuning or new weights;
- fitting any current film or digital photograph;
- changing F1/F2/G0 gates or selecting a favorable subset after results;
- adding a residual LUT, spatial model, or direct RGB generator to rescue G0;
- calling the external reference categories stock truth;
- stock, process, scanner, exposure, push/pull, latent-mode, population
  preference, calibrated, or production claims.

Maximum claim:

`generic reference-conditioned external-ML transform approximated by a bounded
explicit operator on a synthetic colour grid`.

Ultimate remains active under every branch.
