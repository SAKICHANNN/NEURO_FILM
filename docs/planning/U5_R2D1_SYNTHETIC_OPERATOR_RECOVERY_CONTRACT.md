# U5.R2D1 synthetic known-operator recovery contract

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D1`  
Status: **frozen before generator, model or result implementation**

## Question

When the true colour operator is known exactly, which non-generative
conditioning representation can recover a visibly non-trivial, bounded
explicit operator under content-palette and operator-family shift?

This is a data-independent method-identifiability benchmark. It is not a film
experiment and cannot establish that an unpaired scan identifies a real
digital-to-film operator.

## Primary-source motivation

Recent explicit-colour work motivates three separable hypotheses:

- StatLUT predicts a residual 3D LUT from global Lab statistics, but U5.R2D
  proves those statistics remain palette-sensitive and do not uniquely
  identify unseen colours.
- MRStyle/IRStyle reports that interaction between content and reference
  features and a dual mapping outperform independent/direct mappings. Its
  image branch still learns from supervised and unpaired objectives and is not
  stock evidence.
- CanonCGT proposes an explicit canonical pivot before reference grading. The
  project has no validated real-film canonicalizer, so a synthetic
  neutral-reference delta is an oracle control, not a deployable fact.

The generative text branches of StatLUT/MRStyle and all direct-RGB generators
remain excluded.

## Exact generated truth

All pixels and operators are generated locally from the frozen seed. No image,
LUT pack, checkpoint or external dataset is used.

The encoded-sRGB operator is:

1. a positive, row-stochastic 3x3 matrix, which maps the unit cube into itself
   and preserves the neutral axis;
2. one shared strictly monotone five-knot tone curve with fixed endpoints;
3. no clipping, spatial mapping, texture synthesis or direct RGB generation.

The nine ground-truth parameters are six off-diagonal matrix entries and three
interior tone knots. Projection of predictions back into the same feasible
set is mandatory and is reported separately from raw regression error.

Three operator families are generated:

- `matrix_only`: non-identity matrix, identity curve;
- `tone_only`: identity matrix, non-identity curve;
- `combined`: both components non-identity.

Exactly 128 operators per family are generated. Operator ID is the leakage
group. Development, confirmation and stress contain 64/32/32 operators per
family, respectively. Within development, 48/16 per family are fit/validation
groups. Normalization, PCA, model selection and thresholds use development
only.

## Palette support

Each operator is observed through deterministic synthetic colour clouds, not
through spatial semantics. Development palettes cover balanced, warm, cool
and foliage-like support. Confirmation adds skin-like and low-key support.
Stress contains deliberately narrow supports that omit one or more saturated
colour regions.

Query-content and reference-style palette IDs differ. The same operator may
have several observations within a split, but no operator ID crosses a split.
All headline uncertainty is grouped by operator, never by observation.

## Frozen representations

Every representation uses the U5.R2D 2,304-D Lab descriptor.

1. `global_mean`: no image feature; mean development operator.
2. `source_content_only_negative`: query-content descriptor only.
3. `style_target_only`: transformed reference descriptor only.
4. `interaction_source_target`: concatenated query-content and transformed
   reference descriptors; deployable in a reference-guided setting but still
   unpaired.
5. `canonical_reference_delta_oracle`: transformed-reference descriptor minus
   the exact pre-transform reference descriptor.
6. `matched_query_delta_oracle`: transformed-query descriptor minus the exact
   query descriptor.
7. `shuffled_operator_negative`: operator targets permuted by operator group
   using the frozen seed.

The two delta lanes are synthetic oracles. Real deployment would require a
separately validated canonicalizer or matched neutral reference; neither
exists.

## Frozen predictors

- operator-group mean;
- standardized PCA(48) plus multi-output ridge, with alpha selected only on
  development validation from `[1e-4, 1e-2, 1, 100]`;
- standardized PCA(48) plus KNN regression with `k` selected only on
  development validation from `[1, 3, 5, 9]`.

All preprocessing is fit on development fit groups only. The chosen
hyperparameters are frozen before confirmation. No neural network, tree
ensemble, GPU or adaptive confirmatory tuning is allowed.

## Evaluation

Each predicted and true operator is rendered on:

- an exact 17-cube uniform probe grid;
- palette-observed probes;
- saturated and neutral-axis stress probes.

Report by split, family and palette-support lane:

- parameter RMSE;
- uniform-grid RGB RMSE, P95 and maximum error;
- CIE Delta E76 median/P95 after explicit encoded-sRGB conversion;
- identity-normalized error and captured-style fraction;
- output range, neutral-axis, determinant and monotonicity validity;
- raw-to-projected parameter movement;
- maximum group share and operator-group bootstrap intervals.

Family leave-one-out models are sensitivity evidence only. They do not share
preprocessing with the all-family model and cannot be tuned on the held-out
family.

## Frozen gates

All gates are evaluated on confirmation before stress is interpreted.

1. **Truth non-triviality:** median true-versus-identity uniform RGB RMSE
   `>= 0.02` in every family.
2. **Leakage:** zero operator IDs cross fit, validation, confirmation or
   stress; repeated audit is byte-identical.
3. **Validity:** 100% of projected predicted operators satisfy range,
   positive determinant, row-sum, monotonicity and neutral-axis checks.
4. **Negative content control:** `source_content_only_negative` may not improve
   median confirmation uniform RGB RMSE over `global_mean` by more than 5%.
5. **Shuffle control:** `shuffled_operator_negative` may not improve over
   `global_mean` by more than 2%.
6. **Canonical information gate:** the best
   `canonical_reference_delta_oracle` predictor must reduce median
   confirmation uniform RGB RMSE by at least 20% relative to the best
   `style_target_only` predictor.
7. **Deployable interaction gate:** the best
   `interaction_source_target` predictor must reduce median confirmation
   uniform RGB RMSE by at least 10% relative to `global_mean` and 5% relative
   to the best `style_target_only` predictor.
8. **Useful recovery:** the best non-oracle method must capture at least 50% of
   true style effect on confirmation, using
   `1 - predicted_error / identity_error`.
9. **Severe numerical veto:** any non-finite output, out-of-range value,
   non-positive determinant, non-monotone curve or neutral-axis error above
   `1e-10` rejects that prediction policy regardless of mean accuracy.

The gates are deliberately asymmetric. Oracle success with deployable
interaction failure means the operator is recoverable only with unavailable
canonical information. Failure of both closes this representation family.

## Branches

- all representation gates fail: close statistics-conditioned prediction and
  return to fixed global/retrieval operators;
- oracle passes, interaction fails: canonicalization is the missing
  information; open only a canonicalizer sensitivity leaf;
- interaction passes within family but collapses on family/palette shift:
  retain as a bounded in-distribution candidate, not a general film method;
- interaction and oracle pass with stable held-out probes: permit a separately
  frozen R2D2 comparison against hard operator retrieval and a small bounded
  parameter predictor;
- any severe numerical veto: reject the corresponding policy;
- no result changes current real-film `operator_fitting_allowed=false`.

## DoD

- contract/config committed before implementation;
- deterministic generator and split manifests have exact hashes;
- independent scalar operator reference and projection tests pass;
- confirmation remains untouched until preprocessing/model choices freeze;
- two final reports are byte-identical;
- full CPU suite and visual probe-grid inspection complete;
- results propagate with the exact claim ceiling;
- scoped commits are pushed and the Ultimate Goal remains ACTIVE.
