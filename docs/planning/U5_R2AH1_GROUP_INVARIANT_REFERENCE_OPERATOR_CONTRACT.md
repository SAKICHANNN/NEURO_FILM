# U5.R2AH1D — group-invariant reference-operator development contract

Date: 2026-07-28

Status: `contract_frozen_implementation_ready`

## Parent and hypothesis

AH0 permits one new generated-group ML experiment after W1's fixed descriptor
failed. The hypothesis is narrow:

> A hierarchical permutation-invariant encoder, trained with explicit
> same-operator/different-content invariance and anti-shortcut controls, can
> recover one unseen bounded O0 operator from four unrelated final reference
> colour sets.

The model predicts only a `4x4x4x3` stationary velocity grid. Existing O0
integration remains the sole RGB renderer. This is not generative AI and the
model never directly emits a final image.

## Frozen generated design

Training and development use new seeds, never the W1 population or its
reserved `28590/28591/28592` confirmation.

- 64 training directions and 16 unseen development directions;
- each split balances palette-score and independently smoothed random O0
  velocity generators;
- strengths `.35/.65/1.0`, plus 20% exact-identity episodes;
- eight content families, independent mixture draws and independent basic
  nuisance streams;
- 12 training and eight development groups per operator instance;
- each group contains four unrelated 256-colour reference sets.

Every operator is crossed with content and nuisance. No generator ID,
content label or nuisance label enters the operator head.

## Frozen model

The clean-room `hierarchical_deep_sets_v1` model has:

1. a shared two-layer point MLP over unordered RGB rows;
2. exact symmetric mean-and-max aggregation within each reference;
3. a shared reference MLP;
4. exact mean aggregation over the four same-look references;
5. a bounded radial-tanh head for 192 O0 velocity scalars;
6. discriminative gradient-reversal content heads on per-reference
   embeddings.

Trainable parameters must not exceed 150,000. Set order must be exactly
invariant. Attention, CNNs, image encoders, pretrained features, CLIP and
semantic labels are forbidden.

## Loss and execution

One local RTX 5070 Ti Laptop run uses deterministic PyTorch float32:

- seed 30131;
- AdamW, 5,000 steps, batch 32, learning rate `3e-4`;
- final-step checkpoint only;
- group and single-reference velocity-grid loss;
- a differentiable fixed `5^3` O0 render loss;
- same-operator reference-embedding agreement;
- VICReg-style variance/covariance terms;
- a gradient-reversal content-family loss.

Loss weights are in the machine-readable config. There is no search, early
stopping or best-checkpoint selection. Two independent processes must
reproduce the report byte-for-byte.

## Development evaluation

Primary scoring uses four references on unseen directions and unseen content.
Single-reference scoring is diagnostic.

The predictor must jointly:

- reach median/p90 operator-output RMSE at most `.05/.08`;
- improve median error by at least 25% over identity and global mean;
- improve by at least 10% over W1's four-reference ridge;
- keep same-look replicate RMSE at most `.04`;
- keep unseen identity-reference error at most `.01`;
- keep both held-out linear and two-layer content probes at balanced accuracy
  at most `.225` (chance `.125`);
- keep nuisance probes at most `.35` (chance `.25`);
- distinguish different looks on fixed content at least 75%;
- preserve the `53/55/56` one-direction strength path;
- pass O0 range, Jacobian determinant/norm, inverse, serialization and exact
  set-permutation gates.

Passing operator RMSE while leaking content still fails.

## Confirmation seal

The reserved confirmation uses a different
monotone-curves-plus-positive-matrix operator family, beta-copula content
family and seeds `30190/30191/30192`. AH1D code must not access it unless
every development gate passes. A development pass opens only a separately
frozen confirmation run.

## Branch

Any development failure closes the candidate without changing width, depth,
steps, loss weights, adversary, seeds or thresholds. A four-reference pass
with a single-reference failure retains a four-reference requirement. A
structural failure rejects the predictor but never reopens or invalidates O0.

No photograph, real film pixel, external code/checkpoint, unpaired real
operator fit, stock/mode claim, visual shortlist or production integration is
allowed. The maximum claim is generated development evidence for predicting
bounded explicit parameters.
